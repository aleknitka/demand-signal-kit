from demand_signal_kit.data.synthetic.dimensions.promo_sensitivity import PromoSensitivityDimension
from demand_signal_kit.data.synthetic.dimensions.base import BaseSegment


@PromoSensitivityDimension.register
class HighPromoSegment(BaseSegment):
    value = "high"
    weight = 0.25

    def behavioral_profile(self) -> dict:
        return {"promo_response": (0.7, 1.0), "price_sensitivity_mod": 1.4}

    def causal_params(self) -> dict:
        return {"promo_depth": (1.2, 0.3)}
