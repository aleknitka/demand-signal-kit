import numpy as np
from datetime import date

from demand_signal_kit.data.synthetic.missions import ALL_MISSIONS, BaseMission


class MissionSelector:
    """Selects a shopping mission based on customer dimensions + context.

    Uses a logit model: utility of each mission is computed from
    customer features + context (day, season), then sampled via softmax.
    """

    def __init__(self, missions: list[BaseMission] | None = None):
        self.missions = missions or ALL_MISSIONS

    def compute_utilities(
        self,
        customer_dimensions: dict[str, str],
        day: date,
        rng: np.random.Generator,
    ) -> dict[str, float]:
        """Compute utility for each available mission."""
        utilities = {}
        is_weekend = day.weekday() >= 5
        hour = int(rng.integers(8, 21))
        is_lunch = 11 <= hour <= 13
        is_evening = 17 <= hour <= 20

        context = {
            "is_weekend": float(is_weekend),
            "is_weekday": float(not is_weekend),
            "is_lunch": float(is_lunch),
            "is_evening": float(is_evening),
            "luxury_affluence": float(customer_dimensions.get("affluence") == "luxury"),
            "budget_affluence": float(customer_dimensions.get("affluence") == "budget"),
            "promo_hunter": float(customer_dimensions.get("promo_sensitivity") == "high"),
            "family_geo": float(customer_dimensions.get("geo_sociological") == "suburban_family"),
            "urban_geo": float(customer_dimensions.get("geo_sociological") == "urban_professional"),
            "champion_loyalty": float(customer_dimensions.get("loyalty_engagement") == "champion"),
            "basket_large": 0.0,
            "basket_small": 0.0,
            "basket_very_large": 0.0,
            "premium_quality": 0.0,
            "impulse_prone": float(customer_dimensions.get("promo_sensitivity") == "high"),
        }

        for mission in self.missions:
            if not mission.is_available(day, rng):
                continue
            features = mission.utility_features()
            utility = sum(context.get(k, 0) * v for k, v in features.items())
            utility += float(rng.normal(0, 0.3))
            utilities[mission.name] = utility

        return utilities

    def select_mission(
        self,
        customer_dimensions: dict[str, str],
        day: date,
        rng: np.random.Generator,
    ) -> BaseMission:
        """Sample a mission based on customer dimensions + context."""
        utilities = self.compute_utilities(customer_dimensions, day, rng)
        if not utilities:
            return self.missions[0]

        names = list(utilities.keys())
        utils = np.array([utilities[n] for n in names])
        shifted = utils - utils.max()
        exp_utils = np.exp(shifted)
        probs = exp_utils / exp_utils.sum()

        chosen_name = rng.choice(names, p=probs)
        for m in self.missions:
            if m.name == chosen_name:
                return m
        return self.missions[0]

    def select_and_get_params(
        self,
        customer_dimensions: dict[str, str],
        day: date,
        rng: np.random.Generator,
    ) -> tuple[BaseMission, dict]:
        """Select mission and return it with its parameters."""
        mission = self.select_mission(customer_dimensions, day, rng)
        params = {
            "mission": mission.name,
            "basket_size": rng.integers(*mission.basket_size_range()),
            "category_mix": mission.category_mix(),
            "qty_range": mission.qty_per_item_range(),
        }
        return mission, params
