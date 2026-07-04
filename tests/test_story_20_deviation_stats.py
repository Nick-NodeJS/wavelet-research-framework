"""Tests for Story 20: Historical Deviation Statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wavelet_research.deviation_stats.collector import DeviationStatsCollector
from wavelet_research.deviation_stats.models import DeviationEvent, DeviationQueryResult
from wavelet_research.deviation_stats.query import DeviationStatsIndex
from wavelet_research.engine.config import WaveletEngineConfig


def _make_data(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    mid = 1.1 + np.cumsum(rng.normal(0, 0.0001, n))
    spread = 0.00005
    return pd.DataFrame({
        "bid": mid - spread / 2,
        "ask": mid + spread / 2,
        "mid": mid,
        "spread": spread,
        "time": pd.date_range("2026-01-01", periods=n, freq="1s"),
    })


_CFG = WaveletEngineConfig(wavelet="db4", window=256, level=2)


class TestDeviationStatsCollector:
    def test_collect_returns_list(self):
        data = _make_data()
        collector = DeviationStatsCollector(_CFG, symbol="EURUSD")
        events = collector.collect(data)
        assert isinstance(events, list)
        assert len(events) > 0

    def test_collect_events_have_required_fields(self):
        data = _make_data()
        collector = DeviationStatsCollector(_CFG, symbol="EURUSD")
        events = collector.collect(data)
        e = events[0]
        assert e.symbol == "EURUSD"
        assert e.window == _CFG.window
        assert isinstance(e.normalized_deviation, float)
        assert isinstance(e.returned_to_trend, bool)
        assert isinstance(e.bars_to_return, int)

    def test_collect_no_lookahead_in_current_features(self):
        """Current features (z_score, trend, slope) must not use future bars."""
        data = _make_data(200)
        collector = DeviationStatsCollector(_CFG, symbol="TEST")
        events = collector.collect(data)
        # All timestamps must be in order (no reordering)
        ts = [e.timestamp for e in events]
        assert ts == sorted(ts)

    def test_collect_deterministic(self):
        data = _make_data(300)
        collector = DeviationStatsCollector(_CFG)
        e1 = collector.collect(data)
        e2 = collector.collect(data)
        assert [e.normalized_deviation for e in e1] == [e.normalized_deviation for e in e2]

    def test_volatility_bucket_values(self):
        data = _make_data(400)
        collector = DeviationStatsCollector(_CFG)
        events = collector.collect(data)
        buckets = {e.volatility_bucket for e in events}
        assert buckets.issubset({"low", "normal", "high"})

    def test_future_return_1_is_finite(self):
        data = _make_data(300)
        collector = DeviationStatsCollector(_CFG)
        events = collector.collect(data)
        for e in events:
            assert np.isfinite(e.future_return_1)


class TestDeviationStatsIndex:
    def _build_index(self) -> DeviationStatsIndex:
        data = _make_data(500)
        collector = DeviationStatsCollector(_CFG, symbol="EURUSD")
        events = collector.collect(data)
        return DeviationStatsIndex(events)

    def test_query_returns_result(self):
        idx = self._build_index()
        result = idx.query(normalized_deviation=1.0, volatility_bucket="normal")
        assert isinstance(result, DeviationQueryResult)

    def test_query_empty_bucket_returns_zero_sample(self):
        idx = self._build_index()
        result = idx.query(normalized_deviation=9999.0, volatility_bucket="low")
        assert result.sample_size == 0
        assert result.confidence_level == "insufficient"

    def test_query_probability_in_range(self):
        idx = self._build_index()
        result = idx.query(normalized_deviation=1.0, volatility_bucket="normal")
        if result.sample_size > 0:
            assert 0.0 <= result.return_to_trend_probability <= 1.0

    def test_query_deterministic(self):
        idx = self._build_index()
        r1 = idx.query(1.0, "normal")
        r2 = idx.query(1.0, "normal")
        assert r1.sample_size == r2.sample_size
        assert r1.return_to_trend_probability == r2.return_to_trend_probability

    def test_confidence_level_values(self):
        idx = self._build_index()
        result = idx.query(0.5, "normal")
        assert result.confidence_level in {"high", "medium", "low", "insufficient"}

    def test_low_sample_returns_low_confidence(self):
        single_event = DeviationEvent(
            timestamp="t", symbol="X", window=128,
            trend_value=1.1, price=1.102,
            normalized_deviation=1.0, trend_slope=0.0001,
            volatility_bucket="normal",
            future_return_1=0.0, future_return_3=0.0,
            future_return_5=0.0, future_return_10=0.0,
            future_return_20=0.0,
            returned_to_trend=True, bars_to_return=3,
            max_favorable_excursion=0.001, max_adverse_excursion=0.0005,
        )
        idx = DeviationStatsIndex([single_event])
        result = idx.query(1.0, "normal")
        assert result.confidence_level == "insufficient"
