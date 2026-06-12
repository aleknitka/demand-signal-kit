from demand_signal_kit.data.synthetic.dimensions.geo_sociological import GeoSociologicalDimension
from demand_signal_kit.data.synthetic.dimensions.base import BaseSegment


@GeoSociologicalDimension.register
class SuburbanFamilySegment(BaseSegment):
    value = "suburban_family"
    weight = 0.45

    def behavioral_profile(self) -> dict:
        return {
            "avg_basket_size_mod": 1.3,
            "visit_frequency_mod": 1.0,
            "category_prefs": {"Food & Beverage": 0.35, "Home & Garden": 0.25, "Clothing": 0.20, "Sports & Outdoors": 0.10, "Electronics": 0.10},
        }

    def causal_params(self) -> dict:
        return {"quality_score": (0.05, 0.05), "brand_strength": (0.05, 0.05)}
