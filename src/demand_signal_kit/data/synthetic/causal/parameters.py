from dataclasses import dataclass, field
import numpy as np

from demand_signal_kit.data.synthetic.dimensions import get_dimension, list_dimensions
from demand_signal_kit.data.synthetic.dimensions.ontology import OntologyMapper


DEFAULT_ATTRIBUTE_NAMES = ["price", "promo_depth", "quality_score", "brand_strength", "seasonal_relevance"]


class ParameterStore:
    """Adjustable parameters for the causal demand model.

    Supports multi-dimensional segmentation via the dimension registry
    and ontology mappings.
    """

    def __init__(
        self,
        ontology: OntologyMapper | None = None,
        dimension_overrides: dict[str, dict[str, float]] | None = None,
    ):
        self.ontology = ontology or OntologyMapper()
        self.dimension_overrides = dimension_overrides or {}
        self._rng = np.random.default_rng(42)
        self.product_params: dict[str, dict[str, float]] = {}

    def set_product_price(self, product_id: str, price: float):
        if product_id not in self.product_params:
            self.product_params[product_id] = {}
        self.product_params[product_id]["price"] = price

    def set_product_promo(self, product_id: str, promo_depth: float):
        if product_id not in self.product_params:
            self.product_params[product_id] = {}
        self.product_params[product_id]["promo_depth"] = promo_depth

    def override_prices(self, overrides: dict[str, float]):
        for pid, price in overrides.items():
            self.set_product_price(pid, price)

    def override_promos(self, overrides: dict[str, float]):
        for pid, depth in overrides.items():
            self.set_product_promo(pid, depth)

    def shift_dimension_weights(self, dimension: str, new_weights: dict[str, float]):
        self.dimension_overrides[dimension] = new_weights

    def generate_agents(self, n_agents: int = 10000) -> list[dict]:
        agents = []
        for _ in range(n_agents):
            dims = self._sample_dimensions()
            taste = self._compose_taste_vector(dims)
            agents.append({"segments": dims, "taste": taste})
        return agents

    def _sample_dimensions(self) -> dict[str, str]:
        dims = {}
        all_dims = list_dimensions()
        primary_dim = all_dims[0] if all_dims else None

        if primary_dim:
            weights = self.dimension_overrides.get(primary_dim, None)
            if weights is None:
                weights = get_dimension(primary_dim).weights()
            dims[primary_dim] = self._rng.choice(
                list(weights.keys()), p=list(weights.values())
            )

        for dim_name in all_dims[1:]:
            base_weights = self.dimension_overrides.get(dim_name, None)
            if base_weights is None:
                base_weights = get_dimension(dim_name).weights()

            adjusted = self.ontology.get_adjusted_weights(dim_name, dims, base_weights)
            values = list(adjusted.keys())
            weights = list(adjusted.values())
            dims[dim_name] = self._rng.choice(values, p=weights)

        return dims

    def _compose_taste_vector(self, dims: dict[str, str]) -> dict[str, float]:
        taste = {attr: 0.0 for attr in DEFAULT_ATTRIBUTE_NAMES}

        for dim_name, dim_val in dims.items():
            try:
                dim_cls = get_dimension(dim_name)
                segment = dim_cls.get(dim_val)
                params = segment.causal_params()
                for attr, (mean, std) in params.items():
                    if attr in taste:
                        taste[attr] += mean + float(self._rng.normal(0, std * 0.3))
            except (KeyError, ValueError):
                continue

        for attr in taste:
            taste[attr] = max(-3.0, min(3.0, taste[attr]))

        return taste

    def snapshot(self) -> dict:
        return {
            "product_params": dict(self.product_params),
            "dimension_overrides": dict(self.dimension_overrides),
            "ontology": self.ontology.snapshot(),
        }

    def restore(self, snapshot: dict):
        self.product_params = dict(snapshot.get("product_params", {}))
        self.dimension_overrides = dict(snapshot.get("dimension_overrides", {}))
        if "ontology" in snapshot:
            self.ontology.restore(snapshot["ontology"])
