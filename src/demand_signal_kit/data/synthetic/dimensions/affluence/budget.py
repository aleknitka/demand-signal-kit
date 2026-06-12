from demand_signal_kit.data.synthetic.dimensions.affluence import AffluenceDimension
from demand_signal_kit.data.synthetic.dimensions.base import BaseSegment


@AffluenceDimension.register
class BudgetSegment(BaseSegment):
    value = "budget"
    weight = 0.30

    def behavioral_profile(self) -> dict:
        return {
            "avg_basket_size": (1, 4),
            "visit_frequency_days": (7, 21),
            "price_sensitivity": (1.2, 1.8),
            "promo_response": (0.7, 1.0),
            "category_prefs": {"Food & Beverage": 0.45, "Clothing": 0.15, "Home & Garden": 0.15, "Electronics": 0.10, "Sports & Outdoors": 0.15},
            "loyalty_tier_weights": {"bronze": 0.6, "silver": 0.3, "gold": 0.08, "platinum": 0.02},
        }

    def causal_params(self) -> dict:
        return {"price": (-1.2, 0.3), "promo_depth": (0.8, 0.2), "quality_score": (0.1, 0.1), "brand_strength": (0.05, 0.05), "seasonal_relevance": (0.1, 0.05)}
