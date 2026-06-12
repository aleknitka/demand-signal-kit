from demand_signal_kit.data.synthetic.missions.base import BaseMission


class ImpulseBrowseMission(BaseMission):
    name = "impulse_browse"
    weight = 0.10

    def basket_size_range(self) -> tuple[int, int]:
        return (2, 6)

    def category_mix(self) -> dict[str, float]:
        return {"Electronics": 0.25, "Clothing": 0.25, "Food & Beverage": 0.20, "Home & Garden": 0.15, "Sports & Outdoors": 0.15}

    def qty_per_item_range(self) -> tuple[int, int]:
        return (1, 1)

    def utility_features(self) -> dict[str, float]:
        return {"is_weekend": 0.2, "urban_geo": 0.15, "impulse_prone": 0.3}
