from typing import Any
import numpy as np


def compose_taste_vector(
    agent_dims: dict[str, str],
    dimension_segments: dict[str, dict[str, Any]],
    attribute_names: list[str],
    rng: np.random.Generator,
) -> dict[str, float]:
    """Compose taste vector from dimension segments.

    Args:
        agent_dims: {dimension_name: segment_value}
        dimension_segments: {dimension_name: {segment_value: segment_obj}}
        attribute_names: list of attribute names for taste vector
        rng: random generator
    """
    taste = {attr: 0.0 for attr in attribute_names}

    for dim_name, dim_val in agent_dims.items():
        if dim_name not in dimension_segments:
            continue
        segment = dimension_segments[dim_name].get(dim_val)
        if segment is None:
            continue
        params = segment.causal_params()
        for attr, (mean, std) in params.items():
            if attr in taste:
                taste[attr] += mean + float(rng.normal(0, std * 0.3))

    for attr in taste:
        taste[attr] = max(-3.0, min(3.0, taste[attr]))

    return taste


def compose_taste_matrix(
    agents: list[dict],
    attribute_names: list[str],
) -> np.ndarray:
    """Convert list of agent dicts to taste matrix (n_agents x n_attributes)."""
    n = len(agents)
    matrix = np.zeros((n, len(attribute_names)))
    for i, agent in enumerate(agents):
        taste = agent.get("taste", {})
        for j, attr in enumerate(attribute_names):
            matrix[i, j] = taste.get(attr, 0.0)
    return matrix
