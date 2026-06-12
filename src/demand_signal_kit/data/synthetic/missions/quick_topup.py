from demand_signal_kit.data.synthetic.missions.base import BaseMission


class QuickTopupMission(BaseMission):
    name = "quick_topup"
    weight = 0.25

    def basket_size_range(self) -> tuple[int, int]:
        return (1, 5)

    def category_mix(self) -> dict[str, float]:
        return {"Food & Beverage": 0.70, "Electronics": 0.10, "Clothing": 0.10, "Home & Garden": 0.05, "Sports & Outdoors": 0.05}

    def qty_per_item_range(self) -> tuple[int, int]:
        return (1, 1)

    def utility_features(self) -> dict[str, float]:
        return {"is_weekday": 0.2, "is_lunch": 0.15, "urban_geo": 0.2, "basket_small": 0.4}

    def is_available(self, day, rng) -> bool:
        return True
