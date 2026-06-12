import polars as pl
import numpy as np
from dataclasses import dataclass, field
from datetime import date, timedelta

from demand_signal_kit.data.synthetic.causal.choice_model import MNLChoiceModel, ChoiceResult
from demand_signal_kit.data.synthetic.causal.parameters import ParameterStore, DEFAULT_ATTRIBUTE_NAMES
from demand_signal_kit.data.synthetic.missions.selector import MissionSelector
from demand_signal_kit.data.synthetic.config import CATEGORY_PROFILES, US_HOLIDAYS_2022_2024


@dataclass
class ScenarioResult:
    name: str
    market_shares: pl.DataFrame
    segment_shares: pl.DataFrame
    total_demand: dict[str, float]
    revenue: dict[str, float]
    parameters: dict
    mission_distribution: dict[str, float] | None = None
    elasticity_matrix: pl.DataFrame | None = None


class CausalDemandSimulator:
    """Full causal demand simulator with adjustable parameters.

    Combines:
    - MNL discrete choice model for product selection
    - Multi-dimensional customer segmentation with ontology
    - Shopping mission selection (weekly stockup, quick topup, etc.)
    - Parameter store with adjustable knobs
    - Scenario comparison engine
    """

    def __init__(
        self,
        products: pl.DataFrame,
        customers: pl.DataFrame | None = None,
        params: ParameterStore | None = None,
        mission_selector: MissionSelector | None = None,
        n_agents: int = 10000,
        seed: int = 42,
    ):
        self.products = products
        self.customers = customers
        self.params = params or ParameterStore()
        self.mission_selector = mission_selector or MissionSelector()
        self.n_agents = n_agents
        self._rng = np.random.default_rng(seed)
        self.choice_model = MNLChoiceModel()
        self._agents = None
        self._product_attrs = None

    def setup(self) -> "CausalDemandSimulator":
        """Initialize product attributes and agents from data."""
        self._product_attrs = self._build_product_attributes()
        self._agents = self._build_agents()
        return self

    def _build_product_attributes(self) -> pl.DataFrame:
        attrs = self.products.select([
            "product_id",
            "unit_price",
            "category",
            "margin",
            "base_demand",
        ]).rename({"unit_price": "price"})

        profile_map = {cat: prof for cat, prof in CATEGORY_PROFILES.items()}

        quality_scores = []
        brand_strengths = []
        seasonal_relevance = []

        for row in attrs.to_dicts():
            cat = row["category"]
            prof = profile_map.get(cat, {})
            quality_scores.append(float(self._rng.uniform(0.2, 0.9)))
            brand_strengths.append(float(self._rng.uniform(0.1, 0.8)))
            seasonal_relevance.append(float(prof.get("yearly_amplitude", 20)) / 50.0)

        attrs = attrs.with_columns([
            pl.Series("quality_score", quality_scores),
            pl.Series("brand_strength", brand_strengths),
            pl.Series("seasonal_relevance", seasonal_relevance),
            pl.lit(0.0).alias("promo_depth"),
        ])

        for pid, pparams in self.params.product_params.items():
            mask = pl.col("product_id") == pid
            for attr, val in pparams.items():
                if attr in attrs.columns:
                    attrs = attrs.with_columns(
                        pl.when(mask).then(pl.lit(val)).otherwise(pl.col(attr)).alias(attr)
                    )

        return attrs

    def _build_agents(self) -> list[dict]:
        return self.params.generate_agents(self.n_agents)

    def run_baseline(self, day: date | None = None) -> ChoiceResult:
        """Run choice simulation with current parameters."""
        if self._agents is None or self._product_attrs is None:
            self.setup()

        attrs = self._product_attrs.clone()
        if day is not None:
            attrs = self._apply_temporal_effects(attrs, day)

        return self.choice_model.predict_demand(self._agents, attrs)

    def run_scenario(
        self,
        name: str,
        price_overrides: dict[str, float] | None = None,
        promo_overrides: dict[str, float] | None = None,
        segment_shifts: dict[str, float] | None = None,
        product_adds: list[dict] | None = None,
        product_removes: list[str] | None = None,
    ) -> ScenarioResult:
        """Run a scenario with modified parameters.

        Args:
            name: Scenario name
            price_overrides: {product_id: new_price}
            promo_overrides: {product_id: promo_depth}
            segment_shifts: {segment_name: new_share}
            product_adds: list of dicts with product attributes to add
            product_removes: list of product_ids to remove
        """
        if self._agents is None or self._product_attrs is None:
            self.setup()

        snapshot = self.params.snapshot()

        attrs = self._product_attrs.clone()

        if price_overrides:
            for pid, price in price_overrides.items():
                attrs = attrs.with_columns(
                    pl.when(pl.col("product_id") == pid)
                    .then(pl.lit(price))
                    .otherwise(pl.col("price"))
                    .alias("price")
                )

        if promo_overrides:
            for pid, depth in promo_overrides.items():
                attrs = attrs.with_columns(
                    pl.when(pl.col("product_id") == pid)
                    .then(pl.lit(depth))
                    .otherwise(pl.col("promo_depth"))
                    .alias("promo_depth")
                )

        if product_removes:
            attrs = attrs.filter(~pl.col("product_id").is_in(product_removes))

        if product_adds:
            new_rows = pl.DataFrame(product_adds)
            attrs = pl.concat([attrs, new_rows])

        agents = self._agents
        if segment_shifts:
            self.params.shift_segment_shares(segment_shifts)
            agents = self.params.generate_agents(self.n_agents)

        result = self.choice_model.predict_demand(agents, attrs)

        prices = dict(zip(attrs["product_id"].to_list(), attrs["price"].to_list()))

        total_demand = {}
        revenue = {}
        for row in result.market_shares.to_dicts():
            pid = row["product_id"]
            total_demand[pid] = row["expected_demand"]
            revenue[pid] = row["expected_demand"] * prices.get(pid, 0)

        self.params.restore(snapshot)

        return ScenarioResult(
            name=name,
            market_shares=result.market_shares,
            segment_shares=result.segment_shares,
            total_demand=total_demand,
            revenue=revenue,
            parameters={
                "price_overrides": price_overrides or {},
                "promo_overrides": promo_overrides or {},
                "segment_shifts": segment_shifts or {},
            },
        )

    def compare_scenarios(
        self,
        scenarios: list[dict],
    ) -> list[ScenarioResult]:
        """Run baseline + multiple scenarios, return all results.

        Each scenario dict should have 'name' and any of:
        price_overrides, promo_overrides, segment_shifts
        """
        baseline = self.run_baseline()
        results = []

        baseline_result = ScenarioResult(
            name="baseline",
            market_shares=baseline.market_shares,
            segment_shares=baseline.segment_shares,
            total_demand=dict(zip(
                baseline.market_shares["product_id"].to_list(),
                baseline.market_shares["expected_demand"].to_list(),
            )),
            revenue={},
            parameters={},
        )
        results.append(baseline_result)

        for scenario in scenarios:
            result = self.run_scenario(**scenario)
            results.append(result)

        return results

    def compute_elasticity_matrix(
        self,
        product_ids: list[str] | None = None,
        price_delta_pct: float = 0.01,
    ) -> pl.DataFrame:
        """Compute full own-price and cross-price elasticity matrix."""
        if self._agents is None or self._product_attrs is None:
            self.setup()

        if product_ids is None:
            product_ids = self._product_attrs["product_id"].to_list()[:20]

        base = self.choice_model.predict_demand(self._agents, self._product_attrs)
        base_demands = dict(zip(
            base.market_shares["product_id"].to_list(),
            base.market_shares["expected_demand"].to_list(),
        ))

        rows = []
        for focal_id in product_ids:
            focal_base = base_demands.get(focal_id, 0)
            if focal_base == 0:
                continue

            original_price = self._product_attrs.filter(
                pl.col("product_id") == focal_id
            )["price"][0]
            new_price = original_price * (1 + price_delta_pct)

            modified = self._product_attrs.clone().with_columns(
                pl.when(pl.col("product_id") == focal_id)
                .then(pl.lit(new_price))
                .otherwise(pl.col("price"))
                .alias("price")
            )

            perturbed = self.choice_model.predict_demand(self._agents, modified)
            perturbed_demands = dict(zip(
                perturbed.market_shares["product_id"].to_list(),
                perturbed.market_shares["expected_demand"].to_list(),
            ))

            for other_id in product_ids:
                other_demand = perturbed_demands.get(other_id, 0)
                other_base = base_demands.get(other_id, 0)
                if other_base == 0:
                    continue

                pct_demand = (other_demand - other_base) / other_base
                elasticity = float(pct_demand / price_delta_pct)

                rows.append({
                    "focal_product": focal_id,
                    "affected_product": other_id,
                    "elasticity": elasticity,
                    "is_own": focal_id == other_id,
                })

        return pl.DataFrame(rows)

    def _apply_temporal_effects(self, attrs: pl.DataFrame, day: date) -> pl.DataFrame:
        """Apply day-of-week and seasonal modifiers to product attributes."""
        dow = day.weekday()
        is_weekend = dow >= 5

        weekend_mod = 1.05 if is_weekend else 1.0

        doy = day.timetuple().tm_yday
        seasonal = 1.0 + 0.1 * np.sin(2 * np.pi * doy / 365)

        is_holiday = day in US_HOLIDAYS_2022_2024
        holiday_mod = 0.7 if is_holiday else 1.0

        attrs = attrs.with_columns([
            (pl.col("quality_score") * weekend_mod * seasonal * holiday_mod).alias("quality_score"),
        ])

        return attrs
