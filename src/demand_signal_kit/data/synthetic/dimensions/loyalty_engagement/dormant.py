from demand_signal_kit.data.synthetic.dimensions.loyalty_engagement import LoyaltyEngagementDimension
from demand_signal_kit.data.synthetic.dimensions.base import BaseSegment


@LoyaltyEngagementDimension.register
class DormantSegment(BaseSegment):
    value = "dormant"
    weight = 0.20

    def behavioral_profile(self) -> dict:
        return {"visit_frequency_mod": 0.3, "loyalty_tier": "bronze", "avg_basket_size_mod": 0.7}

    def causal_params(self) -> dict:
        return {"brand_strength": (-0.1, 0.05)}
