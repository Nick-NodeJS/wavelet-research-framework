"""Collect historical DeviationEvents from tick data offline."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from wavelet_research.deviation.core import DeviationEngine
from wavelet_research.deviation_stats.models import DeviationEvent
from wavelet_research.engine.config import WaveletEngineConfig
from wavelet_research.engine.core import WaveletEngine
from wavelet_research.engine.models import Tick

logger = logging.getLogger(__name__)

_NEAR_Z = 0.3
_FUTURE_WINDOWS = (1, 3, 5, 10, 20)
_VOL_LOW = 0.33
_VOL_HIGH = 0.67


def _volatility_bucket(z_score_history: list[float], current: float) -> str:
    if len(z_score_history) < 10:
        return "normal"
    p33 = float(np.percentile(z_score_history, 33))
    p67 = float(np.percentile(z_score_history, 67))
    if current <= p33:
        return "low"
    if current >= p67:
        return "high"
    return "normal"


class DeviationStatsCollector:
    """Build historical DeviationEvent records from offline tick data.

    Future labels are computed only during offline collection.
    No lookahead is used in current features.

    Parameters
    ----------
    engine_config : WaveletEngineConfig
        Engine configuration.
    symbol : str
        Symbol name for records.
    near_threshold : float
        z-score threshold to consider "returned to trend".
    """

    def __init__(
        self,
        engine_config: WaveletEngineConfig,
        symbol: str = "UNKNOWN",
        near_threshold: float = _NEAR_Z,
    ) -> None:
        self._engine_config = engine_config
        self._symbol = symbol
        self._near_threshold = near_threshold

    def collect(self, data: pd.DataFrame) -> list[DeviationEvent]:
        """Build event records from historical tick data.

        Parameters
        ----------
        data : pd.DataFrame
            Tick data with: time, bid, ask, mid, spread.

        Returns
        -------
        list[DeviationEvent]
            One event per post-warmup bar with future labels attached.
        """
        ticks = _to_ticks(data)
        wavelet_engine = WaveletEngine(self._engine_config)
        deviation_engine = DeviationEngine()

        points: list[tuple[int, float, float, float, float, str]] = []
        z_history: list[float] = []
        mid_series: list[float] = []
        timestamps: list[str] = []

        for i, tick in enumerate(ticks):
            wp = wavelet_engine.update(tick)
            if wp is None:
                continue
            dp = deviation_engine.compute(wp, tick)
            vol_bucket = _volatility_bucket(z_history, abs(dp.z_score))
            z_history.append(abs(dp.z_score))
            points.append((i, dp.z_score, wp.trend, dp.trend_slope, tick.spread, vol_bucket))
            mid_series.append(tick.mid)
            timestamps.append(str(tick.time))

        events: list[DeviationEvent] = []
        for k, (orig_idx, z_score, trend, slope, spread, vol_bucket) in enumerate(points):
            future_mids = mid_series[k + 1: k + 21]
            f_returns = _future_returns(mid_series[k], future_mids)
            returned, bars_to_return = _return_to_trend(
                mid_series[k:], trend, self._near_threshold
            )
            mfe, mae = _mfe_mae(mid_series[k:], mid_series[k], z_score, self._near_threshold)

            event = DeviationEvent(
                timestamp=timestamps[k],
                symbol=self._symbol,
                window=self._engine_config.window,
                trend_value=trend,
                price=mid_series[k],
                normalized_deviation=z_score,
                trend_slope=slope,
                volatility_bucket=vol_bucket,
                future_return_1=f_returns[0],
                future_return_3=f_returns[1],
                future_return_5=f_returns[2],
                future_return_10=f_returns[3],
                future_return_20=f_returns[4],
                returned_to_trend=returned,
                bars_to_return=bars_to_return,
                max_favorable_excursion=mfe,
                max_adverse_excursion=mae,
            )
            events.append(event)

        logger.info("Collected %d deviation events", len(events))
        return events


def _future_returns(price_now: float, future_mids: list[float]) -> tuple[float, ...]:
    padded = (future_mids + [price_now] * 20)[:20]
    horizons = (1, 3, 5, 10, 20)
    return tuple(padded[h - 1] - price_now for h in horizons)


def _return_to_trend(
    mids: list[float], trend: float, near_threshold: float
) -> tuple[bool, int]:
    for i, m in enumerate(mids[1:], 1):
        if abs(m - trend) / max(abs(trend), 1e-9) < near_threshold:
            return True, i
    return False, 0


def _mfe_mae(
    mids: list[float], entry: float, z_score: float, near_threshold: float
) -> tuple[float, float]:
    if len(mids) < 2:
        return 0.0, 0.0
    direction = 1.0 if z_score < 0 else -1.0  # below trend → expect up move
    moves = [(m - entry) * direction for m in mids[1:21]]
    mfe = max(moves) if moves else 0.0
    mae = abs(min(moves)) if moves else 0.0
    return mfe, mae


def _to_ticks(data: pd.DataFrame) -> list[Tick]:
    ticks: list[Tick] = []
    for row in data.itertuples(index=False):
        ticks.append(Tick(
            time=row.time,  # type: ignore[attr-defined]
            bid=float(row.bid),  # type: ignore[attr-defined]
            ask=float(row.ask),  # type: ignore[attr-defined]
            mid=float(row.mid),  # type: ignore[attr-defined]
            spread=float(row.spread),  # type: ignore[attr-defined]
        ))
    return ticks
