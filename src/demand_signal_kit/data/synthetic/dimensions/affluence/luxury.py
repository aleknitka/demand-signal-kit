from demand_signal_kit.data.synthetic.dimensions.affluence import AffluenceDimension
from demand_signal_kit.data.synthetic.dimensions.base import BaseSegment


@AffluenceDimension.register
class LuxurySegment(BaseSegment):
    value = "luxury"
    weight = 0.25

    def behavioral_profile(self) -> dict:
        return {
            "avg_basket_size": (5, 15),
            "visit_frequency_days": (2, 7),
            "price_sensitivity": (0.2, 0.6),
            "promo_response": (0.1, 0.4),
            "category_prefs": {"Electronics": 0.30, "Clothing": 0.25, "Home & Garden": 0.20, "Sports & Outdoors": 0.15, "Food & Beverage": 0.10},
            "loyalty_tier_weights": {"bronze": 0.05, "silver": 0.10, "gold": 0.35, "platinum": 0.50},
        }

    def causal_params(self) -> dict:
        return {"price": (-0.15, 0.1), "promo_depth": (0.1, 0.08), "quality_score": (0.8, 0.15), "brand_strength": (0.7, 0.1), "seasonal_relevance": (0.25, 0.1)}
