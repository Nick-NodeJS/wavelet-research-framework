"""Tests for the MT5 Indicator (Story 10)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wavelet_research.engine.config import WaveletEngineConfig
from wavelet_research.engine.models import Tick
from wavelet_research.mt5.indicator import IndicatorBuffer, MT5Indicator
from wavelet_research.signal.config import SignalConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tick(mid: float, index: int = 0) -> Tick:
    return Tick(
        time=pd.Timestamp("2026-06-01") + pd.Timedelta(seconds=index),
        bid=mid - 0.00005,
        ask=mid + 0.00005,
        mid=mid,
        spread=0.00010,
    )


def _make_ticks(n: int = 300, seed: int = 42) -> list[Tick]:
    rng = np.random.default_rng(seed)
    mid_values = 1.10 + np.cumsum(rng.normal(0, 0.00002, n))
    return [_make_tick(float(m), i) for i, m in enumerate(mid_values)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMT5Indicator:
    def test_on_tick_returns_buffer(self) -> None:
        ind = MT5Indicator(
            WaveletEngineConfig(wavelet="haar", window=256, level=2),
            SignalConfig(buy_z_threshold=2.0, sell_z_threshold=2.0),
        )
        tick = _make_tick(1.10)
        buf = ind.on_tick(tick)
        assert isinstance(buf, IndicatorBuffer)

    def test_warmup_produces_zero_buffers(self) -> None:
        ind = MT5Indicator(
            WaveletEngineConfig(wavelet="haar", window=256, level=2),
            SignalConfig(),
        )
        # During warmup, all values should be zero
        for i in range(10):
            buf = ind.on_tick(_make_tick(1.10, i))
        assert buf.trend == 0.0
        assert buf.signal == 0

    def test_produces_nonzero_after_warmup(self) -> None:
        ind = MT5Indicator(
            WaveletEngineConfig(wavelet="haar", window=256, level=2),
            SignalConfig(),
        )
        ticks = _make_ticks(300)
        buf = None
        for tick in ticks:
            buf = ind.on_tick(tick)
        assert buf is not None
        assert buf.trend != 0.0

    def test_signal_values(self) -> None:
        ind = MT5Indicator(
            WaveletEngineConfig(wavelet="haar", window=256, level=2),
            SignalConfig(buy_z_threshold=1.0, sell_z_threshold=1.0,
                        slope_filter_enabled=False),
        )
        ticks = _make_ticks(400)
        signals_seen = set()
        for tick in ticks:
            buf = ind.on_tick(tick)
            signals_seen.add(buf.signal)
        # Should see at least HOLD (0)
        assert 0 in signals_seen

    def test_history_length(self) -> None:
        ind = MT5Indicator(
            WaveletEngineConfig(wavelet="haar", window=256, level=2),
            SignalConfig(),
        )
        ticks = _make_ticks(50)
        for tick in ticks:
            ind.on_tick(tick)
        assert ind.buffer_count == 50
        assert len(ind.history) == 50

    def test_reset(self) -> None:
        ind = MT5Indicator(
            WaveletEngineConfig(wavelet="haar", window=256, level=2),
            SignalConfig(),
        )
        ticks = _make_ticks(50)
        for tick in ticks:
            ind.on_tick(tick)
        assert ind.buffer_count == 50
        ind.reset()
        assert ind.buffer_count == 0

    def test_deterministic(self) -> None:
        cfg_w = WaveletEngineConfig(wavelet="haar", window=256, level=2)
        cfg_s = SignalConfig(buy_z_threshold=1.5, sell_z_threshold=1.5,
                            slope_filter_enabled=False)
        ticks = _make_ticks(300)

        ind1 = MT5Indicator(cfg_w, cfg_s)
        ind2 = MT5Indicator(cfg_w, cfg_s)

        for tick in ticks:
            b1 = ind1.on_tick(tick)
            b2 = ind2.on_tick(tick)
            assert b1.trend == b2.trend
            assert b1.signal == b2.signal
            assert b1.confidence == b2.confidence

    def test_buffer_fields(self) -> None:
        ind = MT5Indicator(
            WaveletEngineConfig(wavelet="haar", window=256, level=2),
            SignalConfig(),
        )
        ticks = _make_ticks(300)
        for tick in ticks:
            buf = ind.on_tick(tick)
        # After warmup, check all fields are populated
        assert isinstance(buf.trend, float)
        assert isinstance(buf.deviation, float)
        assert isinstance(buf.z_score, float)
        assert isinstance(buf.energy, float)
        assert isinstance(buf.noise, float)
        assert buf.signal in (-1, 0, 1)
        assert 0.0 <= buf.confidence <= 1.0

    def test_no_business_logic_duplication(self) -> None:
        """Indicator delegates to WaveletEngine and SignalEngine."""
        ind = MT5Indicator(
            WaveletEngineConfig(wavelet="haar", window=256, level=2),
            SignalConfig(),
        )
        # Internal engines exist
        assert hasattr(ind, "_wavelet_engine")
        assert hasattr(ind, "_signal_engine")
