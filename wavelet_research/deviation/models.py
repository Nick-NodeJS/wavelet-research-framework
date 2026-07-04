"""Domain models for the Deviation Engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DeviationSide(str, Enum):
    """Price position relative to trend."""

    ABOVE = "above"
    BELOW = "below"
    NEAR = "near"


@dataclass(frozen=True)
class DeviationPoint:
    """Normalized deviation for a single tick/bar.

    All values are computed using only historical data.
    No lookahead.

    Parameters
    ----------
    raw_distance : float
        price - trend.
    relative_distance : float
        raw_distance / trend (or 0 if trend is zero).
    volatility_normalized : float
        raw_distance / rolling_volatility (z-score).
    z_score : float
        Alias for volatility_normalized (from WaveletEngine).
    side : DeviationSide
        Price position relative to trend.
    spread : float
        Current spread (for filter context).
    trend_slope : float
        Current trend slope.
    """

    raw_distance: float
    relative_distance: float
    volatility_normalized: float
    z_score: float
    side: DeviationSide
    spread: float
    trend_slope: float
