from demand_signal_kit.data.synthetic.dimensions.loyalty_engagement import LoyaltyEngagementDimension
from demand_signal_kit.data.synthetic.dimensions.base import BaseSegment


@LoyaltyEngagementDimension.register
class OccasionalSegment(BaseSegment):
    value = "occasional"
    weight = 0.50

    def behavioral_profile(self) -> dict:
        return {"visit_frequency_mod": 1.0, "loyalty_tier": "silver", "avg_basket_size_mod": 1.0}

    def causal_params(self) -> dict:
        return {"brand_strength": (0.05, 0.05)}
