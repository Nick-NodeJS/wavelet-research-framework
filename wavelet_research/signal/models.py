"""Domain models for the Signal Engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict


class Signal(str, Enum):
    """Trading signal direction."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass(frozen=True)
class SignalDecision:
    """Output of the Signal Engine for a single evaluation.

    Parameters
    ----------
    signal : Signal
        Trading direction: BUY, SELL, or HOLD.
    confidence : float
        Normalized confidence score in [0.0, 1.0].
    reason : str
        Human-readable explanation of the decision.
    z_score : float
        Z-score that triggered (or did not trigger) the signal.
    trend_slope : float
        Trend slope at the time of the decision.
    energy : float
        Wavelet energy at the time of the decision.
    noise : float
        Noise estimate at the time of the decision.
    metadata : dict[str, object]
        Additional metadata for debugging and analysis.
    """

    signal: Signal
    confidence: float
    reason: str
    z_score: float
    trend_slope: float
    energy: float
    noise: float
    metadata: Dict[str, object] = field(default_factory=dict)
