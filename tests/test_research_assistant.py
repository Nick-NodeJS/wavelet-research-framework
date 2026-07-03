"""Tests for the AI Research Assistant (Story 14)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wavelet_research.backtest.config import BacktestConfig, ExitStrategy
from wavelet_research.engine.config import WaveletEngineConfig
from wavelet_research.optimizer.config import ConstraintConfig
from wavelet_research.orchestrator.config import PipelineConfig
from wavelet_research.orchestrator.pipeline import run_pipeline
from wavelet_research.orchestrator.results import ExperimentReport
from wavelet_research.research_assistant.analyzer import ExperimentAnalyzer
from wavelet_research.research_assistant.comparator import ParameterComparator
from wavelet_research.research_assistant.models import (
    AnalysisReport,
    ComparisonReport,
    FailureExplanation,
    Recommendation,
)
from wavelet_research.signal.config import SignalConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mid = 1.10 + np.cumsum(rng.normal(0, 0.00002, n))
    spread = 0.00010
    return pd.DataFrame({
        "time": pd.date_range("2026-06-01", periods=n, freq="100ms"),
        "bid": mid - spread / 2,
        "ask": mid + spread / 2,
        "mid": mid,
        "spread": np.full(n, spread),
    })


def _make_config(buy_z: float = 1.5, sell_z: float = 1.5, max_hold: int = 10) -> PipelineConfig:
    return PipelineConfig(
        wavelet_config=WaveletEngineConfig(wavelet="haar", window=256, level=2),
        signal_config=SignalConfig(
            buy_z_threshold=buy_z, sell_z_threshold=sell_z,
            slope_filter_enabled=False,
        ),
        backtest_config=BacktestConfig(
            pip_size=0.00001,
            exit_strategy=ExitStrategy.MAX_HOLD,
            max_hold_ticks=max_hold,
        ),
    )


def _make_reports(data: pd.DataFrame, configs: list[PipelineConfig]) -> list[ExperimentReport]:
    return [run_pipeline(c, data) for c in configs]


# ---------------------------------------------------------------------------
# Analyzer Tests
# ---------------------------------------------------------------------------


class TestExperimentAnalyzer:
    def test_analyze_produces_report(self) -> None:
        data = _make_dataset()
        configs = [_make_config(buy_z=1.0), _make_config(buy_z=2.0)]
        reports = _make_reports(data, configs)
        analyzer = ExperimentAnalyzer(ConstraintConfig(min_trades=0))
        result = analyzer.analyze(reports)
        assert isinstance(result, AnalysisReport)
        assert result.total_experiments == 2

    def test_identifies_failures(self) -> None:
        data = _make_dataset()
        configs = [_make_config(buy_z=1.5)]
        reports = _make_reports(data, configs)
        # Set min_trades very high so everything fails
        analyzer = ExperimentAnalyzer(ConstraintConfig(min_trades=10000))
        result = analyzer.analyze(reports)
        assert result.failed_count == 1
        assert len(result.failure_explanations) == 1
        assert "Too few trades" in result.failure_explanations[0].reasons[0]

    def test_failure_suggestions(self) -> None:
        data = _make_dataset()
        configs = [_make_config(buy_z=1.5)]
        reports = _make_reports(data, configs)
        analyzer = ExperimentAnalyzer(ConstraintConfig(min_trades=10000))
        result = analyzer.analyze(reports)
        assert result.failure_explanations[0].suggestion != ""

    def test_generates_recommendations(self) -> None:
        data = _make_dataset()
        configs = [_make_config(buy_z=1.0), _make_config(buy_z=2.0)]
        reports = _make_reports(data, configs)
        analyzer = ExperimentAnalyzer(ConstraintConfig(min_trades=0))
        result = analyzer.analyze(reports)
        assert len(result.recommendations) > 0

    def test_generates_insights(self) -> None:
        data = _make_dataset()
        configs = [_make_config(buy_z=1.0)]
        reports = _make_reports(data, configs)
        analyzer = ExperimentAnalyzer(ConstraintConfig(min_trades=0))
        result = analyzer.analyze(reports)
        assert len(result.insights) > 0

    def test_all_fail_recommendation(self) -> None:
        data = _make_dataset()
        configs = [_make_config(buy_z=1.5)]
        reports = _make_reports(data, configs)
        analyzer = ExperimentAnalyzer(ConstraintConfig(min_trades=10000))
        result = analyzer.analyze(reports)
        # Should recommend broadening search
        high_priority = [r for r in result.recommendations if r.priority == "high"]
        assert len(high_priority) > 0

    def test_empty_raises(self) -> None:
        analyzer = ExperimentAnalyzer(ConstraintConfig())
        with pytest.raises(ValueError, match="empty"):
            analyzer.analyze([])

    def test_does_not_modify_reports(self) -> None:
        data = _make_dataset()
        configs = [_make_config(buy_z=1.0)]
        reports = _make_reports(data, configs)
        original_pnl = reports[0].report.total_pnl
        analyzer = ExperimentAnalyzer(ConstraintConfig(min_trades=0))
        analyzer.analyze(reports)
        assert reports[0].report.total_pnl == original_pnl


# ---------------------------------------------------------------------------
# Comparator Tests
# ---------------------------------------------------------------------------


class TestParameterComparator:
    def test_compare_produces_report(self) -> None:
        data = _make_dataset()
        r1 = run_pipeline(_make_config(buy_z=1.0), data)
        r2 = run_pipeline(_make_config(buy_z=2.0), data)
        comparator = ParameterComparator()
        result = comparator.compare(r1, r2)
        assert isinstance(result, ComparisonReport)
        assert result.winner in (r1.config.identifier, r2.config.identifier)

    def test_identifies_winner(self) -> None:
        data = _make_dataset()
        r1 = run_pipeline(_make_config(buy_z=1.0), data)
        r2 = run_pipeline(_make_config(buy_z=2.0), data)
        comparator = ParameterComparator()
        result = comparator.compare(r1, r2)
        assert result.winner != ""
        assert result.explanation != ""

    def test_pnl_difference(self) -> None:
        data = _make_dataset()
        r1 = run_pipeline(_make_config(buy_z=1.0), data)
        r2 = run_pipeline(_make_config(buy_z=2.0), data)
        comparator = ParameterComparator()
        result = comparator.compare(r1, r2)
        expected_diff = r1.report.total_pnl - r2.report.total_pnl
        assert result.pnl_difference == pytest.approx(expected_diff)

    def test_deterministic(self) -> None:
        data = _make_dataset()
        r1 = run_pipeline(_make_config(buy_z=1.0), data)
        r2 = run_pipeline(_make_config(buy_z=2.0), data)
        comparator = ParameterComparator()
        c1 = comparator.compare(r1, r2)
        c2 = comparator.compare(r1, r2)
        assert c1.winner == c2.winner
        assert c1.pnl_difference == c2.pnl_difference

    def test_does_not_modify_reports(self) -> None:
        data = _make_dataset()
        r1 = run_pipeline(_make_config(buy_z=1.0), data)
        r2 = run_pipeline(_make_config(buy_z=2.0), data)
        pnl1 = r1.report.total_pnl
        pnl2 = r2.report.total_pnl
        comparator = ParameterComparator()
        comparator.compare(r1, r2)
        assert r1.report.total_pnl == pnl1
        assert r2.report.total_pnl == pnl2
