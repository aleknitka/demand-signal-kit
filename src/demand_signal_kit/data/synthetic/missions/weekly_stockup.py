from demand_signal_kit.data.synthetic.missions.base import BaseMission


class WeeklyStockupMission(BaseMission):
    name = "weekly_stockup"
    weight = 0.30

    def basket_size_range(self) -> tuple[int, int]:
        return (15, 30)

    def category_mix(self) -> dict[str, float]:
        return {"Food & Beverage": 0.55, "Home & Garden": 0.20, "Clothing": 0.10, "Electronics": 0.05, "Sports & Outdoors": 0.10}

    def qty_per_item_range(self) -> tuple[int, int]:
        return (2, 4)

    def utility_features(self) -> dict[str, float]:
        return {"is_weekend": 0.3, "is_evening": 0.1, "basket_large": 0.5, "family_geo": 0.2}
