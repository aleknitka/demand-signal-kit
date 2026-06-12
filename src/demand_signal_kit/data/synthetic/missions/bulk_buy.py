from demand_signal_kit.data.synthetic.missions.base import BaseMission


class BulkBuyMission(BaseMission):
    name = "bulk_buy"
    weight = 0.20

    def basket_size_range(self) -> tuple[int, int]:
        return (20, 40)

    def category_mix(self) -> dict[str, float]:
        return {"Food & Beverage": 0.50, "Home & Garden": 0.30, "Sports & Outdoors": 0.10, "Clothing": 0.05, "Electronics": 0.05}

    def qty_per_item_range(self) -> tuple[int, int]:
        return (4, 8)

    def utility_features(self) -> dict[str, float]:
        return {"is_weekend": 0.3, "promo_hunter": 0.4, "family_geo": 0.3, "basket_very_large": 0.5}
