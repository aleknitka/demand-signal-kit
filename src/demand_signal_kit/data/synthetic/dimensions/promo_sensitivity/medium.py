from demand_signal_kit.data.synthetic.dimensions.promo_sensitivity import PromoSensitivityDimension
from demand_signal_kit.data.synthetic.dimensions.base import BaseSegment


@PromoSensitivityDimension.register
class MediumPromoSegment(BaseSegment):
    value = "medium"
    weight = 0.45

    def behavioral_profile(self) -> dict:
        return {"promo_response": (0.3, 0.7), "price_sensitivity_mod": 1.0}

    def causal_params(self) -> dict:
        return {"promo_depth": (0.5, 0.15)}
