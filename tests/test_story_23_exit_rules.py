"""Tests for Story 23: Trend-relative exit rules."""

from __future__ import annotations

import pandas as pd
import pytest

from wavelet_research.backtest.config import BacktestConfig, ExitStrategy
from wavelet_research.backtest.position import OpenPosition, check_exit
from wavelet_research.engine.models import Tick, WaveletPoint
from wavelet_research.signal.models import Signal


def _tick(mid=1.102, spread=0.0001):
    return Tick(
        time=pd.Timestamp.now(),
        bid=mid - spread / 2,
        ask=mid + spread / 2,
        mid=mid,
        spread=spread,
    )


def _wavelet_point(trend=1.1, z_score=1.0, slope=0.0001):
    return WaveletPoint(
        trend=trend, deviation=0.002, z_score=z_score,
        slope=slope, energy=0.5, noise=0.1,
    )


def _open_position(side=Signal.BUY, entry_price=1.1, held=0):
    pos = OpenPosition(
        side=side,
        entry_index=0,
        entry_price=entry_price,
        entry_time=pd.Timestamp.now(),
        entry_trend=1.1,
    )
    pos._ticks_held = held
    return pos


class TestReturnToTrendExit:
    def _cfg(self):
        return BacktestConfig(
            pip_size=0.00001,
            exit_strategy=ExitStrategy.RETURN_TO_TREND,
            max_hold_ticks=100,
        )

    def test_buy_exits_when_price_reaches_trend(self):
        cfg = self._cfg()
        pos = _open_position(Signal.BUY)
        tick = _tick(mid=1.101)   # above trend
        wp = _wavelet_point(trend=1.1)
        reason = check_exit(pos, tick, cfg, wp, None)
        assert reason == "return_to_trend"

    def test_buy_does_not_exit_below_trend(self):
        cfg = self._cfg()
        pos = _open_position(Signal.BUY)
        tick = _tick(mid=1.09)
        wp = _wavelet_point(trend=1.1)
        reason = check_exit(pos, tick, cfg, wp, None)
        assert reason is None

    def test_sell_exits_when_price_reaches_trend(self):
        cfg = self._cfg()
        pos = _open_position(Signal.SELL)
        tick = _tick(mid=1.099)   # below trend
        wp = _wavelet_point(trend=1.1)
        reason = check_exit(pos, tick, cfg, wp, None)
        assert reason == "return_to_trend"

    def test_max_hold_overrides(self):
        cfg = self._cfg()
        pos = _open_position(Signal.BUY, held=100)
        tick = _tick(mid=1.09)
        wp = _wavelet_point(trend=1.1)
        reason = check_exit(pos, tick, cfg, wp, None)
        assert reason == "max_hold"


class TestDeviationNormalizedExit:
    def _cfg(self):
        return BacktestConfig(
            pip_size=0.00001,
            exit_strategy=ExitStrategy.DEVIATION_NORMALIZED,
            max_hold_ticks=100,
            exit_deviation_threshold=0.5,
        )

    def test_exits_when_deviation_small(self):
        cfg = self._cfg()
        pos = _open_position(Signal.BUY)
        tick = _tick(mid=1.1005)
        wp = _wavelet_point(trend=1.1, z_score=0.001)
        reason = check_exit(pos, tick, cfg, wp, None)
        assert reason == "deviation_normalized"


class TestMaxAdverseMoveExit:
    def _cfg(self):
        return BacktestConfig(
            pip_size=0.00001,
            exit_strategy=ExitStrategy.MAX_ADVERSE_MOVE,
            max_hold_ticks=100,
            max_adverse_normalized_deviation=2.0,
        )

    def test_exits_when_z_score_exceeds_threshold(self):
        cfg = self._cfg()
        pos = _open_position(Signal.BUY)
        tick = _tick()
        wp = _wavelet_point(z_score=2.5)
        reason = check_exit(pos, tick, cfg, wp, None)
        assert reason == "max_adverse_move"

    def test_no_exit_below_threshold(self):
        cfg = self._cfg()
        pos = _open_position(Signal.BUY)
        tick = _tick()
        wp = _wavelet_point(z_score=1.5)
        reason = check_exit(pos, tick, cfg, wp, None)
        assert reason is None


class TestTrendInvalidationExit:
    def _cfg(self):
        return BacktestConfig(
            pip_size=0.00001,
            exit_strategy=ExitStrategy.TREND_INVALIDATION,
            max_hold_ticks=100,
        )

    def test_buy_exits_on_negative_slope(self):
        cfg = self._cfg()
        pos = _open_position(Signal.BUY)
        tick = _tick()
        wp = _wavelet_point(slope=-0.001)
        reason = check_exit(pos, tick, cfg, wp, None)
        assert reason == "trend_invalidation"

    def test_sell_exits_on_positive_slope(self):
        cfg = self._cfg()
        pos = _open_position(Signal.SELL)
        tick = _tick()
        wp = _wavelet_point(slope=0.001)
        reason = check_exit(pos, tick, cfg, wp, None)
        assert reason == "trend_invalidation"


class TestCombinedExit:
    def _cfg(self):
        return BacktestConfig(
            pip_size=0.00001,
            exit_strategy=ExitStrategy.COMBINED,
            max_hold_ticks=100,
            exit_on_trend_touch=True,
            max_adverse_normalized_deviation=2.5,
        )

    def test_combined_return_to_trend(self):
        cfg = self._cfg()
        pos = _open_position(Signal.BUY)
        tick = _tick(mid=1.101)
        wp = _wavelet_point(trend=1.1, z_score=0.5)
        reason = check_exit(pos, tick, cfg, wp, None)
        assert reason == "return_to_trend"

    def test_combined_adverse_stop(self):
        cfg = self._cfg()
        pos = _open_position(Signal.BUY)
        tick = _tick(mid=1.09)
        wp = _wavelet_point(trend=1.1, z_score=3.0)
        reason = check_exit(pos, tick, cfg, wp, None)
        assert reason == "max_adverse_move"
