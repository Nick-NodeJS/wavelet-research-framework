"""Wavelet computation for a batch of ticks.

Delegates entirely to the existing WaveletEngine.
This is the only layer that knows about engine internals.
"""

from __future__ import annotations

import pandas as pd

from wavelet_research.engine.config import WaveletEngineConfig
from wavelet_research.engine.core import WaveletEngine
from wavelet_research.engine.models import Tick
from wavelet_research.service.models import TickRequest, WaveletResponse


def process_ticks(
    tick_requests: tuple[TickRequest, ...],
    engine_config: WaveletEngineConfig,
) -> WaveletResponse:
    """Run the Wavelet Engine over a batch of ticks and return arrays.

    Positions before the engine warms up are filled with 0.0.
    All output arrays have length == len(tick_requests).

    Parameters
    ----------
    tick_requests : tuple[TickRequest, ...]
        Validated tick sequence.
    engine_config : WaveletEngineConfig
        Engine configuration (wavelet, window, level, vol_window).

    Returns
    -------
    WaveletResponse
        All output arrays with equal length.
    """
    engine = WaveletEngine(engine_config)

    trend: list[float] = []
    relative_deviation: list[float] = []
    z_score: list[float] = []
    energy: list[float] = []
    noise: list[float] = []

    for tr in tick_requests:
        tick = Tick(
            time=pd.Timestamp(tr.time) if tr.time else pd.Timestamp.now(),
            bid=tr.bid,
            ask=tr.ask,
            mid=tr.mid,
            spread=tr.ask - tr.bid,
        )
        point = engine.update(tick)

        if point is None:
            trend.append(0.0)
            relative_deviation.append(0.0)
            z_score.append(0.0)
            energy.append(0.0)
            noise.append(0.0)
        else:
            trend.append(point.trend)
            # relative_deviation = (mid - trend) / local_volatility = z_score
            relative_deviation.append(point.z_score)
            z_score.append(point.z_score)
            energy.append(point.energy)
            noise.append(point.noise)

    return WaveletResponse(
        trend=tuple(trend),
        relative_deviation=tuple(relative_deviation),
        z_score=tuple(z_score),
        energy=tuple(energy),
        noise=tuple(noise),
    )
