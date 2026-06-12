import numpy as np
from datetime import date

from demand_signal_kit.data.synthetic.missions import ALL_MISSIONS
from demand_signal_kit.data.synthetic.missions.base import BaseMission


def compute_mission_utilities(
    agent_dims: dict[str, str],
    day: date,
    mission_weights: dict[str, float],
    rng: np.random.Generator,
) -> dict[str, float]:
    """Compute utility for each mission given agent dimensions and day context."""
    is_weekend = day.weekday() >= 5
    hour = int(rng.integers(8, 21))
    is_lunch = 11 <= hour <= 13
    is_evening = 17 <= hour <= 20

    context = {
        "is_weekend": float(is_weekend),
        "is_weekday": float(not is_weekend),
        "is_lunch": float(is_lunch),
        "is_evening": float(is_evening),
        "is_holiday": 0.0,
        "luxury_affluence": float(agent_dims.get("affluence") == "luxury"),
        "budget_affluence": float(agent_dims.get("affluence") == "budget"),
        "promo_hunter": float(agent_dims.get("promo_sensitivity") == "high"),
        "family_geo": float(agent_dims.get("geo_sociological") == "suburban_family"),
        "urban_geo": float(agent_dims.get("geo_sociological") == "urban_professional"),
        "champion_loyalty": float(agent_dims.get("loyalty_engagement") == "champion"),
        "basket_large": 0.0,
        "basket_small": 0.0,
        "basket_very_large": 0.0,
        "premium_quality": 0.0,
        "impulse_prone": float(agent_dims.get("promo_sensitivity") == "high"),
    }

    utilities = {}
    for mission in ALL_MISSIONS:
        if not mission.is_available(day, rng):
            continue
        features = mission.utility_features()
        utility = sum(context.get(k, 0) * v for k, v in features.items())
        utility += float(rng.normal(0, 0.3))
        utilities[mission.name] = utility

    return utilities


def select_mission_from_utilities(
    utilities: dict[str, float],
    rng: np.random.Generator,
) -> str:
    """Sample a mission name from utilities via softmax."""
    if not utilities:
        return ALL_MISSIONS[0].name

    names = list(utilities.keys())
    utils = np.array([utilities[n] for n in names])
    shifted = utils - utils.max()
    exp_utils = np.exp(shifted)
    probs = exp_utils / exp_utils.sum()

    return rng.choice(names, p=probs)


def get_mission_params(mission_name: str) -> dict:
    """Get parameters for a selected mission."""
    for m in ALL_MISSIONS:
        if m.name == mission_name:
            return {
                "basket_size_range": m.basket_size_range(),
                "category_mix": m.category_mix(),
                "qty_range": m.qty_per_item_range(),
            }
    return ALL_MISSIONS[0].basket_size_range(), ALL_MISSIONS[0].category_mix(), ALL_MISSIONS[0].qty_per_item_range()
