from dataclasses import dataclass, field
import numpy as np


@dataclass
class SegmentParams:
    name: str
    share: float
    attribute_means: dict[str, float] = field(default_factory=dict)
    attribute_stds: dict[str, float] = field(default_factory=dict)


DEFAULT_ATTRIBUTE_NAMES = ["price", "promo_depth", "quality_score", "brand_strength", "seasonal_relevance"]

DEFAULT_SEGMENTS = [
    SegmentParams(
        name="budget",
        share=0.25,
        attribute_means={"price": -1.2, "promo_depth": 0.8, "quality_score": 0.1, "brand_strength": 0.05, "seasonal_relevance": 0.1},
        attribute_stds={"price": 0.3, "promo_depth": 0.2, "quality_score": 0.1, "brand_strength": 0.05, "seasonal_relevance": 0.05},
    ),
    SegmentParams(
        name="regular",
        share=0.35,
        attribute_means={"price": -0.6, "promo_depth": 0.4, "quality_score": 0.3, "brand_strength": 0.2, "seasonal_relevance": 0.15},
        attribute_stds={"price": 0.2, "promo_depth": 0.15, "quality_score": 0.15, "brand_strength": 0.1, "seasonal_relevance": 0.08},
    ),
    SegmentParams(
        name="premium",
        share=0.20,
        attribute_means={"price": -0.3, "promo_depth": 0.2, "quality_score": 0.6, "brand_strength": 0.5, "seasonal_relevance": 0.2},
        attribute_stds={"price": 0.15, "promo_depth": 0.1, "quality_score": 0.2, "brand_strength": 0.15, "seasonal_relevance": 0.1},
    ),
    SegmentParams(
        name="vip",
        share=0.10,
        attribute_means={"price": -0.15, "promo_depth": 0.1, "quality_score": 0.8, "brand_strength": 0.7, "seasonal_relevance": 0.25},
        attribute_stds={"price": 0.1, "promo_depth": 0.08, "quality_score": 0.15, "brand_strength": 0.1, "seasonal_relevance": 0.1},
    ),
    SegmentParams(
        name="promo_hunter",
        share=0.10,
        attribute_means={"price": -0.8, "promo_depth": 1.2, "quality_score": 0.05, "brand_strength": 0.02, "seasonal_relevance": 0.05},
        attribute_stds={"price": 0.25, "promo_depth": 0.3, "quality_score": 0.05, "brand_strength": 0.02, "seasonal_relevance": 0.03},
    ),
]


class ParameterStore:
    """Adjustable parameters for the causal demand model — the 'knobs'."""

    def __init__(
        self,
        product_params: dict[str, dict[str, float]] | None = None,
        segment_params: list[SegmentParams] | None = None,
        seasonal_multipliers: dict[int, float] | None = None,
    ):
        self.product_params = product_params or {}
        self.segments = segment_params or DEFAULT_SEGMENTS
        self.seasonal_multipliers = seasonal_multipliers or {}
        self._rng = np.random.default_rng(42)

    def get_product_attributes(self, product_id: str) -> dict[str, float]:
        if product_id in self.product_params:
            return self.product_params[product_id]
        return {
            "price": 10.0,
            "promo_depth": 0.0,
            "quality_score": 0.5,
            "brand_strength": 0.3,
            "seasonal_relevance": 0.2,
        }

    def set_product_price(self, product_id: str, price: float):
        if product_id not in self.product_params:
            self.product_params[product_id] = self.get_product_attributes(product_id)
        self.product_params[product_id]["price"] = price

    def set_product_promo(self, product_id: str, promo_depth: float):
        if product_id not in self.product_params:
            self.product_params[product_id] = self.get_product_attributes(product_id)
        self.product_params[product_id]["promo_depth"] = promo_depth

    def override_prices(self, overrides: dict[str, float]):
        for pid, price in overrides.items():
            self.set_product_price(pid, price)

    def override_promos(self, overrides: dict[str, float]):
        for pid, depth in overrides.items():
            self.set_product_promo(pid, depth)

    def shift_segment_shares(self, shifts: dict[str, float]):
        total = sum(s.share for s in self.segments)
        for seg in self.segments:
            if seg.name in shifts:
                seg.share = shifts[seg.name]
        new_total = sum(s.share for s in self.segments)
        if new_total > 0:
            for seg in self.segments:
                seg.share /= new_total

    def generate_agents(self, n_agents: int = 10000) -> list[dict]:
        agents = []
        for seg in self.segments:
            n = max(1, int(n_agents * seg.share))
            for _ in range(n):
                taste = {}
                for attr in DEFAULT_ATTRIBUTE_NAMES:
                    mean = seg.attribute_means.get(attr, 0.0)
                    std = seg.attribute_stds.get(attr, 0.1)
                    taste[attr] = float(self._rng.normal(mean, std))
                agents.append({"segment": seg.name, "taste": taste})
        return agents

    def snapshot(self) -> dict:
        return {
            "product_params": dict(self.product_params),
            "segment_shares": {s.name: s.share for s in self.segments},
        }

    def restore(self, snapshot: dict):
        self.product_params = dict(snapshot.get("product_params", {}))
        shares = snapshot.get("segment_shares", {})
        for seg in self.segments:
            if seg.name in shares:
                seg.share = shares[seg.name]
