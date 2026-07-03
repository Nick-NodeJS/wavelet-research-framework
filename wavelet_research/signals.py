from enum import Enum
from dataclasses import dataclass
from .wavelet_engine import WaveletPoint


class Signal(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass(frozen=True)
class SignalDecision:
    signal: Signal
    reason: str


def decide(point: WaveletPoint, threshold: float) -> SignalDecision:
    """
    v0 mean-reversion logic:
    - buy when price is materially below wavelet trend and trend slope is not collapsing
    - sell when price is materially above wavelet trend and trend slope is not exploding
    """
    if point.z_score <= -threshold and point.slope >= 0:
        return SignalDecision(Signal.BUY, "negative_z_with_non_negative_slope")

    if point.z_score >= threshold and point.slope <= 0:
        return SignalDecision(Signal.SELL, "positive_z_with_non_positive_slope")

    return SignalDecision(Signal.HOLD, "no_edge")
