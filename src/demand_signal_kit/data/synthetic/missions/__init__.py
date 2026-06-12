from demand_signal_kit.data.synthetic.missions.base import BaseMission
from demand_signal_kit.data.synthetic.missions.weekly_stockup import WeeklyStockupMission
from demand_signal_kit.data.synthetic.missions.quick_topup import QuickTopupMission
from demand_signal_kit.data.synthetic.missions.special_occasion import SpecialOccasionMission
from demand_signal_kit.data.synthetic.missions.bulk_buy import BulkBuyMission
from demand_signal_kit.data.synthetic.missions.impulse_browse import ImpulseBrowseMission

ALL_MISSIONS: list[BaseMission] = [
    WeeklyStockupMission(),
    QuickTopupMission(),
    SpecialOccasionMission(),
    BulkBuyMission(),
    ImpulseBrowseMission(),
]

__all__ = ["BaseMission", "ALL_MISSIONS"]
