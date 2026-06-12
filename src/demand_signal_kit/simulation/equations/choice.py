import numpy as np
import polars as pl
from demand_signal_kit.data.synthetic.config import US_HOLIDAYS_2022_2024


def compute_utilities(
    taste_matrix: np.ndarray,
    product_attrs: np.ndarray,
) -> np.ndarray:
    """Compute utility matrix: n_agents x n_products.

    V_ij = sum_k(beta_ik * x_jk)
    """
    return taste_matrix @ product_attrs.T


def softmax(utilities: np.ndarray) -> np.ndarray:
    """Convert utilities to choice probabilities via softmax.

    Uses log-sum-exp trick for numerical stability.
    """
    shifted = utilities - utilities.max(axis=1, keepdims=True)
    exp_util = np.exp(shifted)
    return exp_util / exp_util.sum(axis=1, keepdims=True)


def predict_demand(
    probabilities: np.ndarray,
    agent_dims: list[dict],
    product_ids: list[str],
) -> dict:
    """Aggregate choice probabilities into demand results."""
    market_shares = probabilities.mean(axis=0)

    segment_shares = {}
    for i, agent in enumerate(agent_dims):
        seg = agent.get("affluence", "unknown")
        if seg not in segment_shares:
            segment_shares[seg] = []
        segment_shares[seg].append(probabilities[i])

    for seg in segment_shares:
        segment_shares[seg] = np.array(segment_shares[seg]).mean(axis=0)

    return {
        "market_shares": dict(zip(product_ids, market_shares.tolist())),
        "expected_demand": dict(zip(product_ids, (market_shares * len(agent_dims)).tolist())),
        "segment_shares": {seg: dict(zip(product_ids, shares.tolist())) for seg, shares in segment_shares.items()},
    }


def apply_temporal_effects(
    product_attrs: np.ndarray,
    day,
    attribute_names: list[str],
) -> np.ndarray:
    """Apply day-of-week, seasonal, and holiday modifiers."""
    modified = product_attrs.copy()
    dow = day.weekday()
    is_weekend = dow >= 5
    doy = day.timetuple().tm_yday
    is_holiday = day in US_HOLIDAYS_2022_2024

    quality_idx = attribute_names.index("quality_score") if "quality_score" in attribute_names else -1
    if quality_idx >= 0:
        weekend_mod = 1.05 if is_weekend else 1.0
        seasonal = 1.0 + 0.1 * np.sin(2 * np.pi * doy / 365)
        holiday_mod = 0.7 if is_holiday else 1.0
        modified[:, quality_idx] *= weekend_mod * seasonal * holiday_mod

    return modified
