"""Tests for Story 22: Trend-relative entry rules."""

from __future__ import annotations

import pandas as pd
import pytest

from wavelet_research.deviation.models import DeviationPoint, DeviationSide
from wavelet_research.deviation_stats.models import DeviationQueryResult
from wavelet_research.engine.models import WaveletPoint
from wavelet_research.filters.models import FilterConfig, FilterResult, FilterSeverity
from wavelet_research.signal.config import SignalConfig
from wavelet_research.signal.core import SignalEngine
from wavelet_research.signal.models import Signal


def _wp(z_score=2.5, slope=0.0001, trend=1.1):
    return WaveletPoint(trend=trend, deviation=0.002, z_score=z_score,
                        slope=slope, energy=0.5, noise=0.1)


def _dp(z_score=2.5, spread=0.0001, slope=0.0001):
    return DeviationPoint(
        raw_distance=0.002, relative_distance=0.002,
        volatility_normalized=z_score, z_score=z_score,
        side=DeviationSide.ABOVE if z_score > 0 else DeviationSide.BELOW,
        spread=spread, trend_slope=slope,
    )


def _stats(sample=200, prob=0.70):
    return DeviationQueryResult(
        sample_size=sample, return_to_trend_probability=prob,
        median_bars_to_return=7.0, expected_return=0.001,
        expected_adverse_excursion=0.5, confidence_level="high",
    )


def _pass_filter():
    return FilterResult(can_trade=True, reasons=(), severity=FilterSeverity.PASS)


def _block_filter():
    from wavelet_research.filters.models import FilterReason
    return FilterResult(
        can_trade=False,
        reasons=(FilterReason.LOW_SAMPLE_SIZE,),
        severity=FilterSeverity.BLOCK,
    )


_CFG = SignalConfig(
    buy_z_threshold=2.0, sell_z_threshold=2.0,
    slope_filter_enabled=False,
    min_normalized_deviation=1.5,
    min_return_probability=0.55,
    min_stats_sample_size=50,
)


class TestDecideWithContext:
    def _engine(self):
        return SignalEngine(_CFG)

    def test_buy_below_trend(self):
        engine = self._engine()
        wp = _wp(z_score=-2.5, slope=-0.0001)
        dp = _dp(z_score=-2.5, slope=-0.0001)
        result = engine.decide_with_context(wp, dp, _stats(), _pass_filter())
        assert result.signal == Signal.BUY

    def test_sell_above_trend(self):
        engine = self._engine()
        wp = _wp(z_score=2.5, slope=0.0001)
        dp = _dp(z_score=2.5, slope=0.0001)
        result = engine.decide_with_context(wp, dp, _stats(), _pass_filter())
        assert result.signal == Signal.SELL

    def test_blocked_by_filter(self):
        engine = self._engine()
        wp = _wp(z_score=2.5)
        dp = _dp(z_score=2.5)
        result = engine.decide_with_context(wp, dp, _stats(), _block_filter())
        assert result.signal == Signal.HOLD
        assert result.reason == "filter_blocked"

    def test_blocked_by_low_probability(self):
        engine = self._engine()
        wp = _wp(z_score=2.5)
        dp = _dp(z_score=2.5)
        result = engine.decide_with_context(wp, dp, _stats(prob=0.3), _pass_filter())
        assert result.signal == Signal.HOLD
        assert result.reason == "low_return_probability"

    def test_blocked_by_deviation_too_small(self):
        engine = self._engine()
        wp = _wp(z_score=0.5)
        dp = _dp(z_score=0.5)
        result = engine.decide_with_context(wp, dp, _stats(), _pass_filter())
        assert result.signal == Signal.HOLD
        assert "deviation_too_small" in result.reason

    def test_blocked_by_insufficient_history(self):
        engine = self._engine()
        wp = _wp(z_score=2.5)
        dp = _dp(z_score=2.5)
        result = engine.decide_with_context(wp, dp, _stats(sample=5), _pass_filter())
        assert result.signal == Signal.HOLD
        assert result.reason == "insufficient_history"

    def test_metadata_contains_context(self):
        engine = self._engine()
        wp = _wp(z_score=2.5)
        dp = _dp(z_score=2.5)
        result = engine.decide_with_context(wp, dp, _stats(), _pass_filter())
        if result.signal in (Signal.BUY, Signal.SELL):
            assert "normalized_deviation" in result.metadata
            assert "historical_probability" in result.metadata
            assert "expected_bars_to_return" in result.metadata

    def test_confidence_is_valid_range(self):
        engine = self._engine()
        wp = _wp(z_score=2.5)
        dp = _dp(z_score=2.5)
        result = engine.decide_with_context(wp, dp, _stats(), _pass_filter())
        assert 0.0 <= result.confidence <= 1.0

    def test_deterministic(self):
        engine = self._engine()
        wp = _wp(z_score=2.5)
        dp = _dp(z_score=2.5)
        r1 = engine.decide_with_context(wp, dp, _stats(), _pass_filter())
        r2 = engine.decide_with_context(wp, dp, _stats(), _pass_filter())
        assert r1.signal == r2.signal
        assert r1.confidence == r2.confidence
