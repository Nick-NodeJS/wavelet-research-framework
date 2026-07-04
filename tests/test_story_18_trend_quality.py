"""Tests for Story 18: Trend Quality Audit."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from wavelet_research.engine.config import WaveletEngineConfig
from wavelet_research.trend_quality.audit import TrendAuditor
from wavelet_research.trend_quality.metrics import (
    compute_cross_frequency,
    compute_direction_stability,
    compute_lag_estimate,
    compute_quality_score,
    compute_repaint,
    compute_smoothness,
)
from wavelet_research.trend_quality.models import TrendQualityReport, TrendVerdict


def _make_data(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    mid = 1.1 + np.cumsum(rng.normal(0, 0.0001, n))
    spread = 0.00005
    df = pd.DataFrame({
        "bid": mid - spread / 2,
        "ask": mid + spread / 2,
        "mid": mid,
        "spread": spread,
        "time": pd.date_range("2026-01-01", periods=n, freq="1s"),
    })
    return df


_ENGINE_CFG = WaveletEngineConfig(wavelet="db4", window=256, level=2)


class TestTrendQualityMetrics:
    def test_repaint_identical_returns_zero(self):
        vals = [1.0, 2.0, 3.0]
        max_r, mean_r = compute_repaint(vals, vals)
        assert max_r == 0.0
        assert mean_r == 0.0

    def test_repaint_measures_difference(self):
        a = [1.0, 2.0, 3.0]
        b = [1.1, 2.2, 3.3]
        max_r, mean_r = compute_repaint(a, b)
        assert max_r == pytest.approx(0.3, abs=1e-9)
        assert mean_r > 0.0

    def test_repaint_empty_returns_zero(self):
        max_r, mean_r = compute_repaint([], [])
        assert max_r == 0.0 and mean_r == 0.0

    def test_smoothness_flat_trend_is_low(self):
        price = np.array([1.0, 1.001, 0.999, 1.002, 0.998])
        trend = np.ones(5) * 1.0
        s = compute_smoothness(price, trend)
        assert s == pytest.approx(0.0, abs=1e-9)

    def test_smoothness_zero_price_std_returns_zero(self):
        price = np.ones(5)
        trend = np.ones(5)
        assert compute_smoothness(price, trend) == 0.0

    def test_direction_stability_all_same(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        assert compute_direction_stability(arr) == pytest.approx(1.0)

    def test_direction_stability_alternating(self):
        arr = np.array([1.0, 2.0, 1.0, 2.0, 1.0])
        result = compute_direction_stability(arr)
        assert result < 0.5

    def test_cross_frequency_no_crosses(self):
        price = np.array([2.0, 2.1, 2.2, 2.3])
        trend = np.ones(4)
        freq = compute_cross_frequency(price, trend)
        assert freq == pytest.approx(0.0)

    def test_cross_frequency_alternating(self):
        price = np.array([1.1, 0.9, 1.1, 0.9])
        trend = np.ones(4)
        freq = compute_cross_frequency(price, trend)
        assert freq > 0.5

    def test_lag_estimate_returns_int(self):
        rng = np.random.default_rng(0)
        price = rng.normal(0, 1, 100)
        trend = np.convolve(price, np.ones(5) / 5, mode="same")
        lag = compute_lag_estimate(price, trend, max_lag=20)
        assert isinstance(lag, int)
        assert 0 <= lag <= 20

    def test_quality_score_in_range(self):
        score = compute_quality_score(0.0001, 0.3, 0.8, 0.1)
        assert 0.0 <= score <= 1.0


class TestTrendAuditor:
    def test_audit_runs_on_data(self):
        data = _make_data(600)
        auditor = TrendAuditor(_ENGINE_CFG)
        report = auditor.audit(data)
        assert isinstance(report, TrendQualityReport)

    def test_audit_returns_verdict(self):
        data = _make_data(600)
        auditor = TrendAuditor(_ENGINE_CFG)
        report = auditor.audit(data)
        assert report.recommendation in (TrendVerdict.PASS, TrendVerdict.REVIEW, TrendVerdict.FAIL)

    def test_audit_repaint_max_is_nonnegative(self):
        data = _make_data(600)
        auditor = TrendAuditor(_ENGINE_CFG)
        report = auditor.audit(data)
        assert report.repaint_max >= 0.0

    def test_audit_score_in_range(self):
        data = _make_data(600)
        auditor = TrendAuditor(_ENGINE_CFG)
        report = auditor.audit(data)
        assert 0.0 <= report.trend_quality_score <= 1.0

    def test_audit_report_serializable(self):
        data = _make_data(600)
        auditor = TrendAuditor(_ENGINE_CFG)
        report = auditor.audit(data)
        d = report.to_dict()
        text = json.dumps(d)
        assert "trend_quality_score" in text
        assert "recommendation" in text
        assert "repaint_max" in text

    def test_audit_deterministic(self):
        data = _make_data(600)
        auditor = TrendAuditor(_ENGINE_CFG)
        r1 = auditor.audit(data)
        r2 = auditor.audit(data)
        assert r1.trend_quality_score == r2.trend_quality_score

    def test_audit_raises_on_empty_data(self):
        auditor = TrendAuditor(_ENGINE_CFG)
        with pytest.raises(ValueError, match="empty"):
            auditor.audit(pd.DataFrame())

    def test_audit_raises_on_insufficient_data(self):
        data = _make_data(50)
        auditor = TrendAuditor(_ENGINE_CFG)
        with pytest.raises(ValueError):
            auditor.audit(data)

    def test_assess_current_stable(self):
        auditor = TrendAuditor(_ENGINE_CFG)
        trends = [1.0, 1.001, 1.002, 1.003, 1.004]
        state = auditor.assess_current(trends)
        assert state.slope_consistent is True

    def test_assess_current_insufficient_data(self):
        auditor = TrendAuditor(_ENGINE_CFG)
        state = auditor.assess_current([1.0])
        assert state.is_stable is False


class TestCLITrendAudit:
    """Smoke test: CLI trend-audit imports and runs."""

    def test_cmd_trend_audit_import(self):
        from wavelet_research.cli_research import cmd_trend_audit
        assert callable(cmd_trend_audit)
