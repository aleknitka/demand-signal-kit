import numpy as np
from dataclasses import dataclass, field


@dataclass
class Correlation:
    source_dim: str
    source_val: str
    target_dim: str
    target_weights: dict[str, float]


DEFAULT_CORRELATIONS = [
    Correlation("affluence", "luxury", "promo_sensitivity", {"low": 0.75, "medium": 0.20, "high": 0.05}),
    Correlation("affluence", "budget", "promo_sensitivity", {"low": 0.10, "medium": 0.30, "high": 0.60}),
    Correlation("affluence", "middle", "promo_sensitivity", {"low": 0.25, "medium": 0.50, "high": 0.25}),
    Correlation("affluence", "luxury", "geo_sociological", {"urban_professional": 0.60, "suburban_family": 0.25, "rural_value": 0.15}),
    Correlation("affluence", "budget", "geo_sociological", {"urban_professional": 0.20, "suburban_family": 0.40, "rural_value": 0.40}),
    Correlation("promo_sensitivity", "high", "affluence", {"budget": 0.50, "middle": 0.40, "luxury": 0.10}),
    Correlation("geo_sociological", "suburban_family", "affluence", {"budget": 0.20, "middle": 0.55, "luxury": 0.25}),
    Correlation("geo_sociological", "urban_professional", "affluence", {"budget": 0.15, "middle": 0.40, "luxury": 0.45}),
    Correlation("loyalty_engagement", "champion", "affluence", {"budget": 0.10, "middle": 0.40, "luxury": 0.50}),
    Correlation("loyalty_engagement", "dormant", "affluence", {"budget": 0.45, "middle": 0.40, "luxury": 0.15}),
]


class OntologyMapper:
    """Maps correlations between segmentation dimensions.

    Users can add/modify correlations at runtime.
    Given a primary dimension value, correlated dimensions are sampled
    from adjusted probability weights.
    """

    def __init__(self, correlations: list[Correlation] | None = None):
        self.correlations = correlations or list(DEFAULT_CORRELATIONS)
        self._index: dict[tuple[str, str], list[Correlation]] = {}
        self._build_index()

    def _build_index(self):
        self._index.clear()
        for c in self.correlations:
            key = (c.source_dim, c.source_val)
            if key not in self._index:
                self._index[key] = []
            self._index[key].append(c)

    def add_correlation(self, correlation: Correlation):
        self.correlations.append(correlation)
        self._build_index()

    def remove_correlation(self, source_dim: str, source_val: str, target_dim: str):
        self.correlations = [
            c for c in self.correlations
            if not (c.source_dim == source_dim and c.source_val == source_val and c.target_dim == target_dim)
        ]
        self._build_index()

    def get_adjusted_weights(
        self,
        target_dim: str,
        known_dimensions: dict[str, str],
        base_weights: dict[str, float],
    ) -> dict[str, float]:
        """Adjust base weights for target_dim given known dimension values.

        Uses Bayesian-style updating: correlations shift the base weights.
        """
        adjusted = dict(base_weights)
        sources_applied = 0

        for source_dim, source_val in known_dimensions.items():
            if source_dim == target_dim:
                continue
            key = (source_dim, source_val)
            if key not in self._index:
                continue

            for corr in self._index[key]:
                if corr.target_dim != target_dim:
                    continue
                strength = 0.6
                for target_val, corr_weight in corr.target_weights.items():
                    if target_val in adjusted:
                        base = adjusted[target_val]
                        adjusted[target_val] = base * (1 - strength) + corr_weight * strength
                sources_applied += 1

        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}

        return adjusted

    def sample_dimension_value(
        self,
        target_dim: str,
        known_dimensions: dict[str, str],
        base_weights: dict[str, float],
        rng: np.random.Generator,
    ) -> str:
        """Sample a value for target_dim given known dimensions + correlations."""
        adjusted = self.get_adjusted_weights(target_dim, known_dimensions, base_weights)
        values = list(adjusted.keys())
        weights = list(adjusted.values())
        return rng.choice(values, p=weights)

    def snapshot(self) -> list[dict]:
        return [
            {"source_dim": c.source_dim, "source_val": c.source_val, "target_dim": c.target_dim, "target_weights": c.target_weights}
            for c in self.correlations
        ]

    def restore(self, snapshot: list[dict]):
        self.correlations = [Correlation(**d) for d in snapshot]
        self._build_index()
