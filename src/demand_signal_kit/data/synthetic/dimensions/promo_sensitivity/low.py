from demand_signal_kit.data.synthetic.dimensions.promo_sensitivity import PromoSensitivityDimension
from demand_signal_kit.data.synthetic.dimensions.base import BaseSegment


@PromoSensitivityDimension.register
class LowPromoSegment(BaseSegment):
    value = "low"
    weight = 0.30

    def behavioral_profile(self) -> dict:
        return {"promo_response": (0.0, 0.3), "price_sensitivity_mod": 0.7}

    def causal_params(self) -> dict:
        return {"promo_depth": (0.1, 0.08)}
