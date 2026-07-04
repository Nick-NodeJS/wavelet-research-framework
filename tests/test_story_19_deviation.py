"""Tests for Story 19: Deviation Engine."""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from wavelet_research.deviation.core import DeviationEngine
from wavelet_research.deviation.models import DeviationPoint, DeviationSide
from wavelet_research.deviation.normalization import (
    classify_side,
    relative_distance,
    volatility_normalized_distance,
)
from wavelet_research.engine.config import WaveletEngineConfig
from wavelet_research.engine.core import WaveletEngine
from wavelet_research.engine.models import Tick, WaveletPoint


def _make_wavelet_point(trend=1.1, deviation=0.002, z_score=1.5, slope=0.0001):
    return WaveletPoint(trend=trend, deviation=deviation, z_score=z_score,
                        slope=slope, energy=0.5, noise=0.1)


def _make_tick(mid=1.102, spread=0.0001):
    return Tick(time=pd.Timestamp.now(), bid=mid - spread/2,
                ask=mid + spread/2, mid=mid, spread=spread)


class TestNormalizationHelpers:
    def test_relative_distance_positive(self):
        result = relative_distance(0.002, 1.1)
        assert result == pytest.approx(0.002 / 1.1)

    def test_relative_distance_zero_trend(self):
        assert relative_distance(0.002, 0.0) == 0.0

    def test_volatility_normalized_positive(self):
        assert volatility_normalized_distance(0.002, 0.001) == pytest.approx(2.0)

    def test_volatility_normalized_zero_vol(self):
        assert volatility_normalized_distance(0.002, 0.0) == 0.0

    def test_volatility_normalized_negative(self):
        assert volatility_normalized_distance(-0.003, 0.001) == pytest.approx(-3.0)

    def test_classify_side_above(self):
        assert classify_side(1.5) == "above"

    def test_classify_side_below(self):
        assert classify_side(-1.5) == "below"

    def test_classify_side_near(self):
        assert classify_side(0.1) == "near"


class TestDeviationEngine:
    def test_compute_returns_deviation_point(self):
        engine = DeviationEngine()
        wp = _make_wavelet_point()
        tick = _make_tick()
        result = engine.compute(wp, tick)
        assert isinstance(result, DeviationPoint)

    def test_compute_above_trend(self):
        engine = DeviationEngine()
        wp = _make_wavelet_point(z_score=2.0)
        tick = _make_tick()
        result = engine.compute(wp, tick)
        assert result.side == DeviationSide.ABOVE

    def test_compute_below_trend(self):
        engine = DeviationEngine()
        wp = _make_wavelet_point(z_score=-2.0)
        tick = _make_tick()
        result = engine.compute(wp, tick)
        assert result.side == DeviationSide.BELOW

    def test_compute_near_trend(self):
        engine = DeviationEngine()
        wp = _make_wavelet_point(z_score=0.1)
        tick = _make_tick()
        result = engine.compute(wp, tick)
        assert result.side == DeviationSide.NEAR

    def test_compute_z_score_alias(self):
        engine = DeviationEngine()
        wp = _make_wavelet_point(z_score=1.8)
        tick = _make_tick()
        dp = engine.compute(wp, tick)
        assert dp.z_score == dp.volatility_normalized

    def test_compute_spread_preserved(self):
        engine = DeviationEngine()
        wp = _make_wavelet_point()
        tick = _make_tick(spread=0.0002)
        dp = engine.compute(wp, tick)
        assert dp.spread == pytest.approx(0.0002)

    def test_compute_no_lookahead(self):
        """Each call is purely based on passed-in objects — no state."""
        engine = DeviationEngine()
        wp1 = _make_wavelet_point(z_score=1.0)
        wp2 = _make_wavelet_point(z_score=-1.0)
        tick = _make_tick()
        dp1 = engine.compute(wp1, tick)
        dp2 = engine.compute(wp2, tick)
        assert dp1.side == DeviationSide.ABOVE
        assert dp2.side == DeviationSide.BELOW

    def test_compute_deterministic(self):
        engine = DeviationEngine()
        wp = _make_wavelet_point()
        tick = _make_tick()
        assert engine.compute(wp, tick) == engine.compute(wp, tick)

    def test_compute_with_live_engine(self):
        """Integration: compute from a real WaveletEngine output."""
        cfg = WaveletEngineConfig(wavelet="db4", window=256, level=2)
        wavelet_engine = WaveletEngine(cfg)
        dev_engine = DeviationEngine()

        rng = np.random.default_rng(7)
        mid = 1.1 + np.cumsum(rng.normal(0, 0.0001, 300))
        spread = 0.00005
        found = 0

        for m in mid:
            tick = Tick(
                time=pd.Timestamp.now(), bid=m - spread/2,
                ask=m + spread/2, mid=m, spread=spread
            )
            wp = wavelet_engine.update(tick)
            if wp is not None:
                dp = dev_engine.compute(wp, tick)
                assert isinstance(dp, DeviationPoint)
                assert dp.side in list(DeviationSide)
                found += 1
        assert found > 0, "Engine should have produced at least one point"
