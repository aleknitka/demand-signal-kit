from demand_signal_kit.data.synthetic.missions.base import BaseMission
from demand_signal_kit.data.synthetic.config import US_HOLIDAYS_2022_2024


class SpecialOccasionMission(BaseMission):
    name = "special_occasion"
    weight = 0.15

    def basket_size_range(self) -> tuple[int, int]:
        return (5, 15)

    def category_mix(self) -> dict[str, float]:
        return {"Electronics": 0.30, "Clothing": 0.25, "Food & Beverage": 0.25, "Home & Garden": 0.10, "Sports & Outdoors": 0.10}

    def qty_per_item_range(self) -> tuple[int, int]:
        return (1, 2)

    def utility_features(self) -> dict[str, float]:
        return {"is_holiday": 0.4, "luxury_affluence": 0.3, "premium_quality": 0.2}

    def is_available(self, day, rng) -> bool:
        is_holiday = day in US_HOLIDAYS_2022_2024
        is_friday = day.weekday() == 4
        if is_holiday or is_friday:
            return True
        return rng.random() < 0.05
