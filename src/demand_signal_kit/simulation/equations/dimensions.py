from demand_signal_kit.data.synthetic.dimensions import list_dimensions, get_dimension
from demand_signal_kit.data.synthetic.dimensions.ontology import OntologyMapper
import numpy as np


def sample_primary_dimension(
    dimension_weights: dict[str, dict[str, float]],
    rng: np.random.Generator,
) -> tuple[str, str]:
    """Sample the first dimension independently."""
    all_dims = list(dimension_weights.keys())
    if not all_dims:
        return "", ""
    primary = all_dims[0]
    weights = dimension_weights[primary]
    values = list(weights.keys())
    probs = list(weights.values())
    return primary, rng.choice(values, p=probs)


def sample_all_dimensions(
    dimension_weights: dict[str, dict[str, float]],
    ontology_rules: list[dict],
    rng: np.random.Generator,
) -> dict[str, str]:
    """Sample all dimensions with ontology-adjusted correlations."""
    ontology = OntologyMapper()
    if ontology_rules:
        from demand_signal_kit.data.synthetic.dimensions.ontology import Correlation
        ontology.correlations = [Correlation(**r) for r in ontology_rules]
        ontology._build_index()

    dims = {}
    all_dims = list(dimension_weights.keys())

    if not all_dims:
        return dims

    primary = all_dims[0]
    weights = dimension_weights[primary]
    dims[primary] = rng.choice(list(weights.keys()), p=list(weights.values()))

    for dim_name in all_dims[1:]:
        base_weights = dimension_weights.get(dim_name, {})
        adjusted = ontology.get_adjusted_weights(dim_name, dims, base_weights)
        values = list(adjusted.keys())
        probs = list(adjusted.values())
        dims[dim_name] = rng.choice(values, p=probs)

    return dims
