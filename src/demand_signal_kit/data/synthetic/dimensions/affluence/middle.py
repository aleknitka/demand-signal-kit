from demand_signal_kit.data.synthetic.dimensions.affluence import AffluenceDimension
from demand_signal_kit.data.synthetic.dimensions.base import BaseSegment


@AffluenceDimension.register
class MiddleSegment(BaseSegment):
    value = "middle"
    weight = 0.45

    def behavioral_profile(self) -> dict:
        return {
            "avg_basket_size": (3, 8),
            "visit_frequency_days": (3, 10),
            "price_sensitivity": (0.7, 1.2),
            "promo_response": (0.4, 0.7),
            "category_prefs": {"Food & Beverage": 0.30, "Clothing": 0.20, "Home & Garden": 0.20, "Electronics": 0.15, "Sports & Outdoors": 0.15},
            "loyalty_tier_weights": {"bronze": 0.2, "silver": 0.4, "gold": 0.3, "platinum": 0.1},
        }

    def causal_params(self) -> dict:
        return {"price": (-0.6, 0.2), "promo_depth": (0.4, 0.15), "quality_score": (0.3, 0.15), "brand_strength": (0.2, 0.1), "seasonal_relevance": (0.15, 0.08)}
