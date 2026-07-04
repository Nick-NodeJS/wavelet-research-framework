"""Wavelet computation for a batch of ticks.

Delegates entirely to the existing WaveletEngine.
This is the only layer that knows about engine internals.
"""

from __future__ import annotations

import pandas as pd

from wavelet_research.engine.config import WaveletEngineConfig
from wavelet_research.engine.core import WaveletEngine
from wavelet_research.engine.decomposition import TrendMode
from wavelet_research.engine.models import Tick
from wavelet_research.service.models import TickRequest, WaveletResponse


def _parse_timestamp(raw: str) -> pd.Timestamp:
    """Parse a timestamp string into a pd.Timestamp.

    Accepts:
    - ISO 8601 strings: ``"2026-07-04T12:00:00.123"``
    - Raw millisecond integers as strings: ``"1751670000000"`` (sent by MT5)

    Parameters
    ----------
    raw : str
        Raw time value from the request.

    Returns
    -------
    pd.Timestamp
        Parsed timestamp, or ``pd.Timestamp.now()`` if empty.
    """
    if not raw:
        return pd.Timestamp.now()
    # If the string is a pure integer it is a Unix millisecond timestamp
    if raw.isdigit():
        return pd.Timestamp(int(raw), unit="ms")
    return pd.Timestamp(raw)


def process_ticks(
    tick_requests: tuple[TickRequest, ...],
    engine_config: WaveletEngineConfig,
    trend_mode: TrendMode = TrendMode.A2,
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
    trend_mode : TrendMode
        Wavelet approximation level for trend reconstruction. Defaults to A2.

    Returns
    -------
    WaveletResponse
        All output arrays with equal length.
    """
    config_with_mode = WaveletEngineConfig(
        wavelet=engine_config.wavelet,
        window=engine_config.window,
        level=engine_config.level,
        volatility_window=engine_config.volatility_window,
        trend_mode=trend_mode,
    )
    engine = WaveletEngine(config_with_mode)

    trend: list[float] = []
    relative_deviation: list[float] = []
    z_score: list[float] = []
    energy: list[float] = []
    noise: list[float] = []

    for tr in tick_requests:
        tick = Tick(
            time=_parse_timestamp(tr.time),
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
