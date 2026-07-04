"""Tests for Story 21: No-Trade Filter Engine."""

from __future__ import annotations

import pandas as pd
import pytest

from wavelet_research.deviation.models import DeviationPoint, DeviationSide
from wavelet_research.deviation_stats.models import DeviationQueryResult
from wavelet_research.filters.engine import FilterEngine
from wavelet_research.filters.models import (
    FilterConfig,
    FilterReason,
    FilterResult,
    FilterSeverity,
)
from wavelet_research.filters.rules import (
    check_cooldown,
    check_deviation_too_small,
    check_high_adverse_excursion,
    check_low_return_probability,
    check_low_sample_size,
    check_spread_too_wide,
    check_trend_too_flat,
    check_trend_unstable,
)
from wavelet_research.trend_quality.models import TrendQualityState


def _dp(z_score=1.5, spread=0.0001, slope=0.0002):
    return DeviationPoint(
        raw_distance=0.002, relative_distance=0.002,
        volatility_normalized=z_score, z_score=z_score,
        side=DeviationSide.ABOVE if z_score > 0 else DeviationSide.BELOW,
        spread=spread, trend_slope=slope,
    )


def _stats(sample=200, prob=0.65, mae=0.5):
    return DeviationQueryResult(
        sample_size=sample, return_to_trend_probability=prob,
        median_bars_to_return=7.0, expected_return=0.001,
        expected_adverse_excursion=mae, confidence_level="high",
    )


def _tq_state(stable=True):
    return TrendQualityState(is_stable=stable, repaint_risk=0.001, slope_consistent=True)


_CFG = FilterConfig(
    min_sample_size=50, min_return_probability=0.55,
    max_adverse_excursion=2.0, max_spread=0.0003,
    min_normalized_deviation=1.0, min_slope_abs=1e-6,
    cooldown_bars=5,
)


class TestFilterRules:
    def test_low_sample_size_triggers(self):
        assert check_low_sample_size(_stats(sample=10), _CFG) is True

    def test_low_sample_size_passes(self):
        assert check_low_sample_size(_stats(sample=100), _CFG) is False

    def test_low_return_prob_triggers(self):
        assert check_low_return_probability(_stats(prob=0.3), _CFG) is True

    def test_low_return_prob_passes(self):
        assert check_low_return_probability(_stats(prob=0.7), _CFG) is False

    def test_low_return_prob_empty_stats(self):
        assert check_low_return_probability(_stats(sample=0), _CFG) is False

    def test_high_adverse_triggers(self):
        assert check_high_adverse_excursion(_stats(mae=3.0), _CFG) is True

    def test_high_adverse_passes(self):
        assert check_high_adverse_excursion(_stats(mae=1.0), _CFG) is False

    def test_trend_unstable_triggers(self):
        state = TrendQualityState(is_stable=False, repaint_risk=0.1, slope_consistent=False)
        assert check_trend_unstable(state) is True

    def test_trend_stable_passes(self):
        assert check_trend_unstable(_tq_state(stable=True)) is False

    def test_spread_too_wide_triggers(self):
        dp = _dp(spread=0.001)
        assert check_spread_too_wide(dp, _CFG) is True

    def test_spread_ok_passes(self):
        dp = _dp(spread=0.0001)
        assert check_spread_too_wide(dp, _CFG) is False

    def test_deviation_too_small_triggers(self):
        dp = _dp(z_score=0.5)
        assert check_deviation_too_small(dp, _CFG) is True

    def test_deviation_ok_passes(self):
        dp = _dp(z_score=1.5)
        assert check_deviation_too_small(dp, _CFG) is False

    def test_trend_too_flat_triggers(self):
        dp = _dp(slope=1e-10)
        assert check_trend_too_flat(dp, _CFG) is True

    def test_trend_not_flat_passes(self):
        dp = _dp(slope=0.001)
        assert check_trend_too_flat(dp, _CFG) is False

    def test_cooldown_triggers(self):
        assert check_cooldown(2, _CFG) is True

    def test_cooldown_passes(self):
        assert check_cooldown(10, _CFG) is False


class TestFilterEngine:
    def _engine(self):
        return FilterEngine(_CFG)

    def test_pass_case(self):
        engine = self._engine()
        engine._bars_since_last_signal = 999
        result = engine.evaluate(_dp(), _tq_state(), _stats())
        assert result.can_trade is True
        assert result.severity == FilterSeverity.PASS
        assert result.reasons == ()

    def test_blocked_by_low_sample(self):
        engine = self._engine()
        engine._bars_since_last_signal = 999
        result = engine.evaluate(_dp(), _tq_state(), _stats(sample=5))
        assert result.can_trade is False
        assert FilterReason.LOW_SAMPLE_SIZE in result.reasons

    def test_blocked_by_trend_unstable(self):
        engine = self._engine()
        engine._bars_since_last_signal = 999
        state = TrendQualityState(is_stable=False, repaint_risk=1.0, slope_consistent=False)
        result = engine.evaluate(_dp(), state, _stats())
        assert result.can_trade is False
        assert FilterReason.TREND_UNSTABLE in result.reasons

    def test_multiple_reasons(self):
        engine = self._engine()
        engine._bars_since_last_signal = 999
        state = TrendQualityState(is_stable=False, repaint_risk=1.0, slope_consistent=False)
        result = engine.evaluate(_dp(z_score=0.3), state, _stats(sample=5))
        assert len(result.reasons) >= 2

    def test_cooldown_blocks(self):
        engine = self._engine()
        engine._bars_since_last_signal = 2
        result = engine.evaluate(_dp(), _tq_state(), _stats())
        assert result.can_trade is False
        assert FilterReason.RECENT_SIGNAL_COOLDOWN in result.reasons

    def test_record_signal_resets_cooldown(self):
        engine = self._engine()
        engine._bars_since_last_signal = 999
        engine.record_signal()
        assert engine._bars_since_last_signal == 0

    def test_tick_increments_cooldown(self):
        engine = self._engine()
        engine._bars_since_last_signal = 3
        engine.tick()
        assert engine._bars_since_last_signal == 4

    def test_filter_result_serializable(self):
        engine = self._engine()
        engine._bars_since_last_signal = 999
        result = engine.evaluate(_dp(), _tq_state(), _stats())
        d = result.to_dict()
        assert "can_trade" in d
        assert "reasons" in d
        assert "severity" in d

    def test_warning_severity_on_non_blocking(self):
        engine = self._engine()
        engine._bars_since_last_signal = 999
        result = engine.evaluate(_dp(z_score=0.3), _tq_state(), _stats())
        assert result.severity in (FilterSeverity.WARNING, FilterSeverity.BLOCK)
