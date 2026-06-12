from demand_signal_kit.data.synthetic.dimensions.loyalty_engagement import LoyaltyEngagementDimension
from demand_signal_kit.data.synthetic.dimensions.base import BaseSegment


@LoyaltyEngagementDimension.register
class ChampionSegment(BaseSegment):
    value = "champion"
    weight = 0.30

    def behavioral_profile(self) -> dict:
        return {"visit_frequency_mod": 1.5, "loyalty_tier": "platinum", "avg_basket_size_mod": 1.2}

    def causal_params(self) -> dict:
        return {"brand_strength": (0.2, 0.1)}
