from demand_signal_kit.data.synthetic.dimensions.geo_sociological import GeoSociologicalDimension
from demand_signal_kit.data.synthetic.dimensions.base import BaseSegment


@GeoSociologicalDimension.register
class RuralValueSegment(BaseSegment):
    value = "rural_value"
    weight = 0.25

    def behavioral_profile(self) -> dict:
        return {
            "avg_basket_size_mod": 1.1,
            "visit_frequency_mod": 0.8,
            "category_prefs": {"Food & Beverage": 0.40, "Home & Garden": 0.20, "Sports & Outdoors": 0.15, "Clothing": 0.15, "Electronics": 0.10},
        }

    def causal_params(self) -> dict:
        return {"quality_score": (-0.05, 0.05), "brand_strength": (-0.1, 0.05)}
