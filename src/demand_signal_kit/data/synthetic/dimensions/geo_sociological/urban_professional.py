from demand_signal_kit.data.synthetic.dimensions.geo_sociological import GeoSociologicalDimension
from demand_signal_kit.data.synthetic.dimensions.base import BaseSegment


@GeoSociologicalDimension.register
class UrbanProfessionalSegment(BaseSegment):
    value = "urban_professional"
    weight = 0.30

    def behavioral_profile(self) -> dict:
        return {
            "avg_basket_size_mod": 0.8,
            "visit_frequency_mod": 1.2,
            "category_prefs": {"Food & Beverage": 0.25, "Electronics": 0.25, "Clothing": 0.25, "Home & Garden": 0.15, "Sports & Outdoors": 0.10},
        }

    def causal_params(self) -> dict:
        return {"quality_score": (0.1, 0.05), "brand_strength": (0.15, 0.08)}
