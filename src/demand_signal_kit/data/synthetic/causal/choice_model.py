import numpy as np
import polars as pl
from dataclasses import dataclass

from demand_signal_kit.data.synthetic.causal.parameters import DEFAULT_ATTRIBUTE_NAMES


@dataclass
class ChoiceResult:
    market_shares: pl.DataFrame
    segment_shares: pl.DataFrame
    choice_probabilities: np.ndarray
    n_agents: int


class MNLChoiceModel:
    """Multinomial Logit discrete choice model.

    Each agent has utility for product j:
        U_ij = sum_k(beta_ik * x_jk) + epsilon_ij

    Where epsilon_ij follows Type-1 Extreme Value (Gumbel),
    the choice probability is:
        P(choose j) = exp(V_ij) / sum_k(exp(V_ik))
    """

    def __init__(self, attribute_names: list[str] | None = None):
        self.attribute_names = attribute_names or DEFAULT_ATTRIBUTE_NAMES

    def compute_utilities(
        self,
        agents: list[dict],
        product_attributes: pl.DataFrame,
    ) -> np.ndarray:
        """Compute utility matrix (n_agents x n_products).

        Returns deterministic utilities V_ij for each agent-product pair.
        """
        n_agents = len(agents)
        n_products = len(product_attributes)

        attr_matrix = product_attributes.select(self.attribute_names).to_numpy()

        utilities = np.zeros((n_agents, n_products))
        for i, agent in enumerate(agents):
            taste = agent["taste"]
            beta = np.array([taste.get(attr, 0.0) for attr in self.attribute_names])
            utilities[i] = attr_matrix @ beta

        return utilities

    def choice_probabilities(self, utilities: np.ndarray) -> np.ndarray:
        """Convert utilities to choice probabilities via softmax.

        Uses log-sum-exp trick for numerical stability.
        """
        shifted = utilities - utilities.max(axis=1, keepdims=True)
        exp_util = np.exp(shifted)
        return exp_util / exp_util.sum(axis=1, keepdims=True)

    def predict_demand(
        self,
        agents: list[dict],
        product_attributes: pl.DataFrame,
    ) -> ChoiceResult:
        """Run full choice simulation.

        Returns market shares, segment-level shares, and raw probabilities.
        """
        utilities = self.compute_utilities(agents, product_attributes)
        probs = self.choice_probabilities(utilities)

        market_shares = probs.mean(axis=0)

        segment_shares = {}
        seg_accumulators = {}
        for i, agent in enumerate(agents):
            seg = agent["segment"]
            if seg not in seg_accumulators:
                seg_accumulators[seg] = []
            seg_accumulators[seg].append(probs[i])

        for seg_name, prob_list in seg_accumulators.items():
            segment_shares[seg_name] = np.array(prob_list).mean(axis=0)

        product_ids = product_attributes["product_id"].to_list()
        market_df = pl.DataFrame({
            "product_id": product_ids,
            "market_share": market_shares.tolist(),
            "expected_demand": (market_shares * len(agents)).tolist(),
        })

        seg_rows = []
        for seg_name, shares in segment_shares.items():
            if isinstance(shares, np.ndarray) and shares.ndim > 1:
                shares = shares.mean(axis=0)
            for j, pid in enumerate(product_ids):
                seg_rows.append({
                    "segment": seg_name,
                    "product_id": pid,
                    "market_share": float(shares[j]),
                })
        segment_df = pl.DataFrame(seg_rows)

        return ChoiceResult(
            market_shares=market_df,
            segment_shares=segment_df,
            choice_probabilities=probs,
            n_agents=len(agents),
        )

    def simulate_price_change(
        self,
        agents: list[dict],
        product_attributes: pl.DataFrame,
        price_overrides: dict[str, float],
    ) -> ChoiceResult:
        """Simulate effect of price changes on demand."""
        modified = product_attributes.clone()
        for pid, new_price in price_overrides.items():
            modified = modified.with_columns(
                pl.when(pl.col("product_id") == pid)
                .then(pl.lit(new_price))
                .otherwise(pl.col("price"))
                .alias("price")
            )
        return self.predict_demand(agents, modified)

    def simulate_promo_change(
        self,
        agents: list[dict],
        product_attributes: pl.DataFrame,
        promo_overrides: dict[str, float],
    ) -> ChoiceResult:
        """Simulate effect of promotion changes on demand."""
        modified = product_attributes.clone()
        for pid, depth in promo_overrides.items():
            modified = modified.with_columns(
                pl.when(pl.col("product_id") == pid)
                .then(pl.lit(depth))
                .otherwise(pl.col("promo_depth"))
                .alias("promo_depth")
            )
        return self.predict_demand(agents, modified)

    def own_price_elasticity(
        self,
        agents: list[dict],
        product_attributes: pl.DataFrame,
        product_id: str,
        price_delta_pct: float = 0.01,
    ) -> float:
        """Compute own-price elasticity for a product.

        elasticity = % change in demand / % change in price
        """
        base = self.predict_demand(agents, product_attributes)
        base_demand = base.market_shares.filter(
            pl.col("product_id") == product_id
        )["expected_demand"][0]

        original_price = product_attributes.filter(
            pl.col("product_id") == product_id
        )["price"][0]
        new_price = original_price * (1 + price_delta_pct)

        modified = self.simulate_price_change(
            agents, product_attributes, {product_id: new_price}
        )
        new_demand = modified.market_shares.filter(
            pl.col("product_id") == product_id
        )["expected_demand"][0]

        pct_demand_change = (new_demand - base_demand) / base_demand
        return float(pct_demand_change / price_delta_pct)

    def cross_price_elasticity(
        self,
        agents: list[dict],
        product_attributes: pl.DataFrame,
        focal_id: str,
        other_id: str,
        price_delta_pct: float = 0.01,
    ) -> float:
        """Compute cross-price elasticity: effect of other's price change on focal product."""
        base = self.predict_demand(agents, product_attributes)
        base_demand_focal = base.market_shares.filter(
            pl.col("product_id") == focal_id
        )["expected_demand"][0]

        other_price = product_attributes.filter(
            pl.col("product_id") == other_id
        )["price"][0]
        new_price = other_price * (1 + price_delta_pct)

        modified = self.simulate_price_change(
            agents, product_attributes, {other_id: new_price}
        )
        new_demand_focal = modified.market_shares.filter(
            pl.col("product_id") == focal_id
        )["expected_demand"][0]

        pct_demand_change = (new_demand_focal - base_demand_focal) / base_demand_focal
        return float(pct_demand_change / price_delta_pct)
