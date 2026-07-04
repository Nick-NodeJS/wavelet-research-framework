"""Deviation Engine — converts WaveletPoint + Tick into DeviationPoint."""

from __future__ import annotations

from wavelet_research.deviation.models import DeviationPoint, DeviationSide
from wavelet_research.deviation.normalization import (
    classify_side,
    relative_distance,
    volatility_normalized_distance,
)
from wavelet_research.engine.models import Tick, WaveletPoint


class DeviationEngine:
    """Converts WaveletPoint outputs into normalized DeviationPoint values.

    Pure adapter — no state, no lookahead, no trading logic.
    """

    def compute(self, wavelet_point: WaveletPoint, tick: Tick) -> DeviationPoint:
        """Compute normalized deviation for a single tick.

        Parameters
        ----------
        wavelet_point : WaveletPoint
            Output from WaveletEngine.update().
        tick : Tick
            The same tick passed to WaveletEngine.

        Returns
        -------
        DeviationPoint
            Normalized deviation measures.
        """
        raw = wavelet_point.deviation
        rel = relative_distance(raw, wavelet_point.trend)
        vol_norm = wavelet_point.z_score
        side_str = classify_side(vol_norm)
        side = DeviationSide(side_str)

        return DeviationPoint(
            raw_distance=raw,
            relative_distance=rel,
            volatility_normalized=vol_norm,
            z_score=vol_norm,
            side=side,
            spread=tick.spread,
            trend_slope=wavelet_point.slope,
        )
