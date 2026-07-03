"""Tests for the Strategy Validation & Walk-Forward Framework (Story 8)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wavelet_research.backtest.config import BacktestConfig, ExitStrategy
from wavelet_research.backtest.models import BacktestReport
from wavelet_research.engine.config import WaveletEngineConfig
from wavelet_research.orchestrator.config import PipelineConfig
from wavelet_research.orchestrator.pipeline import run_pipeline
from wavelet_research.orchestrator.results import ExperimentReport
from wavelet_research.signal.config import SignalConfig
from wavelet_research.validation.core import WalkForwardValidator
from wavelet_research.validation.metrics import (
    ExtendedMetrics,
    compute_extended_metrics,
    compute_recovery_factor,
    compute_sharpe,
    compute_sortino,
    compute_stability_score,
)
from wavelet_research.validation.models import (
    ParameterSensitivityReport,
    RobustnessReport,
    ValidationReport,
)
from wavelet_research.validation.monte_carlo import (
    MonteCarloResult,
    run_monte_carlo,
)
from wavelet_research.validation.robustness import (
    RobustnessMetrics,
    compute_robustness,
)
from wavelet_research.validation.sensitivity import (
    ParameterSensitivity,
    analyze_parameter_sensitivity,
)
from wavelet_research.validation.splits import (
    SplitConfig,
    generate_rolling_splits,
    generate_walk_forward_splits,
    split_in_out_of_sample,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset(n: int = 600, seed: int = 42) -> pd.DataFrame:
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


def _make_pipeline_config(
    buy_z: float = 1.5, sell_z: float = 1.5, max_hold: int = 10,
) -> PipelineConfig:
    return PipelineConfig(
        wavelet_config=WaveletEngineConfig(
            wavelet="haar", window=256, level=2,
        ),
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


def _make_experiment_reports(
    data: pd.DataFrame, configs: list[PipelineConfig]
) -> list[ExperimentReport]:
    return [run_pipeline(c, data) for c in configs]


# ---------------------------------------------------------------------------
# SplitConfig Tests
# ---------------------------------------------------------------------------


class TestSplitConfig:
    def test_valid(self) -> None:
        cfg = SplitConfig()
        assert cfg.in_sample_ratio == 0.7
        assert cfg.n_folds == 3

    def test_invalid_ratio_low(self) -> None:
        with pytest.raises(ValueError, match="in_sample_ratio"):
            SplitConfig(in_sample_ratio=0.0)

    def test_invalid_ratio_high(self) -> None:
        with pytest.raises(ValueError, match="in_sample_ratio"):
            SplitConfig(in_sample_ratio=1.0)

    def test_invalid_folds(self) -> None:
        with pytest.raises(ValueError, match="n_folds"):
            SplitConfig(n_folds=0)


# ---------------------------------------------------------------------------
# Split Tests
# ---------------------------------------------------------------------------


class TestSplits:
    def test_is_oos_split(self) -> None:
        data = _make_dataset(100)
        is_data, oos_data = split_in_out_of_sample(data, 0.7)
        assert len(is_data) == 70
        assert len(oos_data) == 30
        assert len(is_data) + len(oos_data) == len(data)

    def test_is_oos_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            split_in_out_of_sample(pd.DataFrame(), 0.7)

    def test_is_oos_invalid_ratio(self) -> None:
        data = _make_dataset(100)
        with pytest.raises(ValueError, match="ratio"):
            split_in_out_of_sample(data, 0.0)

    def test_walk_forward_splits(self) -> None:
        data = _make_dataset(300)
        cfg = SplitConfig(in_sample_ratio=0.7, n_folds=3)
        splits = generate_walk_forward_splits(data, cfg)
        assert len(splits) == 3
        # Each fold has IS + OOS data
        for is_data, oos_data in splits:
            assert len(is_data) > 0
            assert len(oos_data) > 0

    def test_walk_forward_empty_raises(self) -> None:
        cfg = SplitConfig()
        with pytest.raises(ValueError, match="empty"):
            generate_walk_forward_splits(pd.DataFrame(), cfg)

    def test_walk_forward_expanding_is(self) -> None:
        data = _make_dataset(300)
        cfg = SplitConfig(in_sample_ratio=0.7, n_folds=3)
        splits = generate_walk_forward_splits(data, cfg)
        # IS windows should be expanding
        is_sizes = [len(is_data) for is_data, _ in splits]
        for i in range(len(is_sizes) - 1):
            assert is_sizes[i] <= is_sizes[i + 1]

    def test_rolling_splits(self) -> None:
        data = _make_dataset(200)
        splits = generate_rolling_splits(data, window_size=50, step_size=20)
        assert len(splits) > 0
        for train, test in splits:
            assert len(train) == 50
            assert len(test) == 20

    def test_rolling_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            generate_rolling_splits(pd.DataFrame(), 50, 20)

    def test_rolling_too_small_raises(self) -> None:
        data = _make_dataset(10)
        with pytest.raises(ValueError, match="too small"):
            generate_rolling_splits(data, 50, 20)

    def test_splits_reproducible(self) -> None:
        data = _make_dataset(300)
        cfg = SplitConfig(in_sample_ratio=0.7, n_folds=3)
        s1 = generate_walk_forward_splits(data, cfg)
        s2 = generate_walk_forward_splits(data, cfg)
        for (is1, oos1), (is2, oos2) in zip(s1, s2):
            assert len(is1) == len(is2)
            assert len(oos1) == len(oos2)


# ---------------------------------------------------------------------------
# Extended Metrics Tests
# ---------------------------------------------------------------------------


class TestExtendedMetrics:
    def test_sharpe_positive(self) -> None:
        pnl = np.array([5.0, 3.0, 7.0, -1.0, 4.0])
        s = compute_sharpe(pnl)
        assert s > 0

    def test_sharpe_insufficient(self) -> None:
        assert compute_sharpe(np.array([5.0])) == 0.0

    def test_sharpe_zero_std(self) -> None:
        assert compute_sharpe(np.array([5.0, 5.0, 5.0])) == 0.0

    def test_sortino_positive(self) -> None:
        pnl = np.array([5.0, 3.0, 7.0, -1.0, 4.0])
        s = compute_sortino(pnl)
        assert s > 0

    def test_sortino_no_losses(self) -> None:
        pnl = np.array([5.0, 3.0, 7.0])
        s = compute_sortino(pnl)
        assert s == float("inf")

    def test_recovery_factor(self) -> None:
        assert compute_recovery_factor(100.0, 20.0) == pytest.approx(5.0)

    def test_recovery_factor_zero_dd(self) -> None:
        assert compute_recovery_factor(50.0, 0.0) == float("inf")

    def test_stability_score_linear(self) -> None:
        equity = np.arange(1.0, 11.0)
        score = compute_stability_score(equity)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_stability_score_noisy(self) -> None:
        rng = np.random.default_rng(42)
        equity = np.cumsum(rng.normal(0, 1, 100))
        score = compute_stability_score(equity)
        assert 0.0 <= score <= 1.0

    def test_compute_extended_metrics(self) -> None:
        pnl = [10.0, -5.0, 8.0, -3.0, 12.0]
        equity = list(np.cumsum(pnl))
        mae = [2.0, 5.0, 1.0, 3.0, 0.5]
        mfe = [12.0, 1.0, 10.0, 0.5, 14.0]
        metrics = compute_extended_metrics(pnl, equity, mae, mfe, 5.0)
        assert isinstance(metrics, ExtendedMetrics)
        assert metrics.trades == 5
        assert metrics.total_pnl == pytest.approx(22.0)
        assert metrics.sharpe_ratio > 0
        assert 0.0 <= metrics.stability_score <= 1.0


# ---------------------------------------------------------------------------
# Monte Carlo Tests
# ---------------------------------------------------------------------------


class TestMonteCarlo:
    def test_run_monte_carlo(self) -> None:
        pnl = [10.0, -5.0, 8.0, -3.0, 12.0, -2.0, 7.0]
        results = run_monte_carlo(pnl, n_simulations=100, seed=42)
        assert len(results) == 100
        for r in results:
            assert isinstance(r, MonteCarloResult)

    def test_reproducible(self) -> None:
        pnl = [10.0, -5.0, 8.0, -3.0, 12.0]
        r1 = run_monte_carlo(pnl, n_simulations=50, seed=42)
        r2 = run_monte_carlo(pnl, n_simulations=50, seed=42)
        for a, b in zip(r1, r2):
            assert a.total_pnl == b.total_pnl
            assert a.max_drawdown == b.max_drawdown

    def test_different_seeds(self) -> None:
        pnl = [10.0, -5.0, 8.0, -3.0, 12.0]
        r1 = run_monte_carlo(pnl, n_simulations=50, seed=42)
        r2 = run_monte_carlo(pnl, n_simulations=50, seed=99)
        # Drawdown varies by order, so at least some should differ
        diffs = sum(1 for a, b in zip(r1, r2) if a.max_drawdown != b.max_drawdown)
        assert diffs > 0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            run_monte_carlo([], 100)

    def test_invalid_simulations(self) -> None:
        with pytest.raises(ValueError, match="n_simulations"):
            run_monte_carlo([10.0], 0)

    def test_total_pnl_preserved(self) -> None:
        pnl = [10.0, -5.0, 8.0]
        results = run_monte_carlo(pnl, n_simulations=100, seed=42)
        # All simulations should have the same total PnL (same trades, different order)
        expected = sum(pnl)
        for r in results:
            assert r.total_pnl == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Robustness Tests
# ---------------------------------------------------------------------------


class TestRobustness:
    def test_compute_robustness(self) -> None:
        pnl = [10.0, -5.0, 8.0, -3.0, 12.0]
        mc = run_monte_carlo(pnl, n_simulations=100, seed=42)
        rob = compute_robustness(sum(pnl), mc)
        assert isinstance(rob, RobustnessMetrics)
        assert 0.0 <= rob.probability_of_profit <= 1.0
        assert rob.percentile_5_pnl <= rob.percentile_95_pnl

    def test_empty_mc_results(self) -> None:
        rob = compute_robustness(10.0, [])
        assert rob.probability_of_profit == 0.0


# ---------------------------------------------------------------------------
# Sensitivity Tests
# ---------------------------------------------------------------------------


class TestSensitivity:
    def test_parameter_sensitivity(self) -> None:
        data = _make_dataset(n=400)
        configs = [
            _make_pipeline_config(buy_z=1.0, sell_z=1.0),
            _make_pipeline_config(buy_z=2.0, sell_z=2.0),
            _make_pipeline_config(buy_z=3.0, sell_z=3.0),
        ]
        reports = _make_experiment_reports(data, configs)
        sens = analyze_parameter_sensitivity(reports, "buy_z_threshold")
        assert isinstance(sens, ParameterSensitivity)
        assert sens.parameter_name == "buy_z_threshold"
        assert len(sens.values) == 3

    def test_empty_reports_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            analyze_parameter_sensitivity([], "wavelet")

    def test_unknown_parameter_raises(self) -> None:
        data = _make_dataset(n=400)
        configs = [_make_pipeline_config()]
        reports = _make_experiment_reports(data, configs)
        with pytest.raises(ValueError, match="Unknown"):
            analyze_parameter_sensitivity(reports, "nonexistent")


# ---------------------------------------------------------------------------
# WalkForwardValidator Tests
# ---------------------------------------------------------------------------


class TestWalkForwardValidator:
    def test_validate_produces_report(self) -> None:
        data = _make_dataset(n=600)
        config = _make_pipeline_config()
        split_cfg = SplitConfig(in_sample_ratio=0.7, n_folds=2)
        validator = WalkForwardValidator(config, split_cfg)
        report = validator.validate(data)
        assert isinstance(report, ValidationReport)
        assert report.in_sample_metrics.trades >= 0
        assert report.out_of_sample_metrics.trades >= 0
        assert len(report.walk_forward_results) == 2

    def test_validate_deterministic(self) -> None:
        data = _make_dataset(n=600)
        config = _make_pipeline_config()
        split_cfg = SplitConfig(in_sample_ratio=0.7, n_folds=2)
        r1 = WalkForwardValidator(config, split_cfg).validate(data)
        r2 = WalkForwardValidator(config, split_cfg).validate(data)
        assert r1.in_sample_metrics.total_pnl == r2.in_sample_metrics.total_pnl
        assert r1.out_of_sample_metrics.total_pnl == r2.out_of_sample_metrics.total_pnl
        assert r1.mean_oos_efficiency == r2.mean_oos_efficiency

    def test_validate_empty_raises(self) -> None:
        config = _make_pipeline_config()
        split_cfg = SplitConfig()
        validator = WalkForwardValidator(config, split_cfg)
        with pytest.raises(ValueError, match="empty"):
            validator.validate(pd.DataFrame())

    def test_walk_forward_folds(self) -> None:
        data = _make_dataset(n=600)
        config = _make_pipeline_config()
        split_cfg = SplitConfig(in_sample_ratio=0.7, n_folds=3)
        validator = WalkForwardValidator(config, split_cfg)
        report = validator.validate(data)
        assert len(report.walk_forward_results) == 3
        for wf in report.walk_forward_results:
            assert wf.in_sample.metrics.trades >= 0
            assert wf.out_of_sample.metrics.trades >= 0

    def test_monte_carlo_analysis(self) -> None:
        data = _make_dataset(n=500)
        config = _make_pipeline_config(buy_z=1.0, sell_z=1.0, max_hold=5)
        split_cfg = SplitConfig()
        validator = WalkForwardValidator(config, split_cfg)

        # Need to ensure we get trades
        report = run_pipeline(config, data)
        if report.report.trades > 0:
            mc_report = validator.monte_carlo(data, n_simulations=50, seed=42)
            assert isinstance(mc_report, RobustnessReport)
            assert mc_report.n_simulations == 50

    def test_monte_carlo_reproducible(self) -> None:
        data = _make_dataset(n=500)
        config = _make_pipeline_config(buy_z=1.0, sell_z=1.0, max_hold=5)
        split_cfg = SplitConfig()
        validator = WalkForwardValidator(config, split_cfg)

        report = run_pipeline(config, data)
        if report.report.trades > 0:
            r1 = validator.monte_carlo(data, n_simulations=50, seed=42)
            r2 = validator.monte_carlo(data, n_simulations=50, seed=42)
            assert r1.robustness.percentile_50_pnl == r2.robustness.percentile_50_pnl

    def test_monte_carlo_empty_raises(self) -> None:
        config = _make_pipeline_config()
        split_cfg = SplitConfig()
        validator = WalkForwardValidator(config, split_cfg)
        with pytest.raises(ValueError):
            validator.monte_carlo(pd.DataFrame())


# ---------------------------------------------------------------------------
# Regression Tests
# ---------------------------------------------------------------------------


class TestRegression:
    def test_is_oos_no_data_leak(self) -> None:
        """IS and OOS should have no overlapping indices."""
        data = _make_dataset(300)
        is_data, oos_data = split_in_out_of_sample(data, 0.7)
        # After reset_index, check original row content doesn't overlap
        assert len(is_data) + len(oos_data) == len(data)
        # IS ends before OOS starts in original data
        is_last_mid = is_data.iloc[-1]["mid"]
        oos_first_mid = oos_data.iloc[0]["mid"]
        # They should be consecutive entries from original data
        original_is_end = int(len(data) * 0.7)
        assert data.iloc[original_is_end - 1]["mid"] == is_last_mid
        assert data.iloc[original_is_end]["mid"] == oos_first_mid

    def test_walk_forward_no_future_leak(self) -> None:
        """OOS data must come after IS data in each fold."""
        data = _make_dataset(300)
        cfg = SplitConfig(in_sample_ratio=0.7, n_folds=3)
        splits = generate_walk_forward_splits(data, cfg)
        for is_data, oos_data in splits:
            is_last_time = is_data.iloc[-1]["time"]
            oos_first_time = oos_data.iloc[0]["time"]
            assert is_last_time <= oos_first_time
