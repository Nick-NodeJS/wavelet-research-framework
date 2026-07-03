"""Tests for the Parameter Optimization Engine (Story 9)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wavelet_research.backtest.config import BacktestConfig, ExitStrategy
from wavelet_research.backtest.models import BacktestReport
from wavelet_research.engine.config import WaveletEngineConfig
from wavelet_research.optimizer.config import (
    ConstraintConfig,
    ObjectiveConfig,
    OptimizerConfig,
    SearchMethod,
)
from wavelet_research.optimizer.core import ParameterOptimizer
from wavelet_research.optimizer.models import (
    OptimizationReport,
    ScoredConfig,
)
from wavelet_research.optimizer.scoring import (
    compute_objective_score,
    passes_constraints,
)
from wavelet_research.optimizer.search import (
    generate_grid_configs,
    generate_random_configs,
)
from wavelet_research.orchestrator.config import PipelineConfig
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


def _make_report(
    trades: int = 10, pf: float = 1.5, exp: float = 2.0,
    dd: float = 20.0, total: float = 50.0,
) -> BacktestReport:
    from wavelet_research.backtest.models import EquityCurve, TradeJournal
    return BacktestReport(
        trades=trades, win_rate=0.6, profit_factor=pf,
        expectancy=exp, total_pnl=total, max_drawdown=dd,
        average_mae=3.0, average_mfe=5.0,
        average_trade=exp, average_holding_time=5.0,
        journal=TradeJournal(), equity_curve=EquityCurve(),
    )


# ---------------------------------------------------------------------------
# Config Tests
# ---------------------------------------------------------------------------


class TestOptimizerConfig:
    def test_valid(self) -> None:
        cfg = OptimizerConfig()
        assert cfg.search_method == SearchMethod.GRID
        assert cfg.max_iterations == 100

    def test_invalid_max_iterations(self) -> None:
        with pytest.raises(ValueError, match="max_iterations"):
            OptimizerConfig(max_iterations=0)

    def test_empty_wavelets(self) -> None:
        with pytest.raises(ValueError, match="wavelets"):
            OptimizerConfig(wavelets=())

    def test_empty_windows(self) -> None:
        with pytest.raises(ValueError, match="windows"):
            OptimizerConfig(windows=())


class TestObjectiveConfig:
    def test_valid(self) -> None:
        cfg = ObjectiveConfig()
        assert cfg.profit_factor_weight == 0.3

    def test_negative_weight(self) -> None:
        with pytest.raises(ValueError):
            ObjectiveConfig(profit_factor_weight=-1.0)


class TestConstraintConfig:
    def test_valid(self) -> None:
        cfg = ConstraintConfig()
        assert cfg.min_trades == 5

    def test_invalid_min_trades(self) -> None:
        with pytest.raises(ValueError, match="min_trades"):
            ConstraintConfig(min_trades=-1)

    def test_invalid_max_drawdown(self) -> None:
        with pytest.raises(ValueError, match="max_drawdown"):
            ConstraintConfig(max_drawdown=0.0)


# ---------------------------------------------------------------------------
# Scoring Tests
# ---------------------------------------------------------------------------


class TestScoring:
    def test_passes_all_constraints(self) -> None:
        report = _make_report(trades=10, pf=1.5, exp=2.0, dd=20.0)
        constraints = ConstraintConfig(
            min_trades=5, max_drawdown=50.0,
            min_profit_factor=1.0, min_expectancy=0.0,
        )
        assert passes_constraints(report, constraints) is True

    def test_fails_min_trades(self) -> None:
        report = _make_report(trades=3)
        constraints = ConstraintConfig(min_trades=5)
        assert passes_constraints(report, constraints) is False

    def test_fails_max_drawdown(self) -> None:
        report = _make_report(dd=60.0)
        constraints = ConstraintConfig(max_drawdown=50.0)
        assert passes_constraints(report, constraints) is False

    def test_fails_min_profit_factor(self) -> None:
        report = _make_report(pf=0.8)
        constraints = ConstraintConfig(min_profit_factor=1.0)
        assert passes_constraints(report, constraints) is False

    def test_fails_min_expectancy(self) -> None:
        report = _make_report(exp=-1.0)
        constraints = ConstraintConfig(min_expectancy=0.0)
        assert passes_constraints(report, constraints) is False

    def test_objective_score(self) -> None:
        report = _make_report(pf=2.0, exp=5.0, dd=10.0, total=50.0)
        obj = ObjectiveConfig()
        score = compute_objective_score(report, obj)
        assert isinstance(score, float)

    def test_higher_pf_higher_score(self) -> None:
        obj = ObjectiveConfig()
        r1 = _make_report(pf=1.5, exp=2.0, dd=20.0, total=50.0)
        r2 = _make_report(pf=3.0, exp=2.0, dd=20.0, total=50.0)
        assert compute_objective_score(r2, obj) > compute_objective_score(r1, obj)


# ---------------------------------------------------------------------------
# Search Tests
# ---------------------------------------------------------------------------


class TestSearch:
    def test_grid_search_count(self) -> None:
        cfg = OptimizerConfig(
            wavelets=("haar", "db4"),
            windows=(256,),
            levels=(2,),
            buy_z_thresholds=(1.5, 2.0),
            sell_z_thresholds=(1.5,),
            max_hold_ticks_list=(10,),
            slope_filter_options=(False,),
        )
        configs = generate_grid_configs(cfg)
        expected = 2 * 1 * 1 * 1 * 2 * 1 * 1 * 1 * 1
        assert len(configs) == expected

    def test_grid_produces_valid_configs(self) -> None:
        cfg = OptimizerConfig(
            wavelets=("haar",), windows=(256,), levels=(2,),
            buy_z_thresholds=(1.5,), sell_z_thresholds=(1.5,),
            max_hold_ticks_list=(10,), slope_filter_options=(False,),
        )
        configs = generate_grid_configs(cfg)
        assert len(configs) == 1
        assert isinstance(configs[0], PipelineConfig)

    def test_random_search_count(self) -> None:
        cfg = OptimizerConfig(
            search_method=SearchMethod.RANDOM,
            max_iterations=25,
        )
        configs = generate_random_configs(cfg)
        assert len(configs) == 25

    def test_random_search_deterministic(self) -> None:
        cfg = OptimizerConfig(
            search_method=SearchMethod.RANDOM,
            max_iterations=10, seed=42,
        )
        c1 = generate_random_configs(cfg)
        c2 = generate_random_configs(cfg)
        for a, b in zip(c1, c2):
            assert a.identifier == b.identifier

    def test_random_different_seed(self) -> None:
        cfg1 = OptimizerConfig(
            search_method=SearchMethod.RANDOM,
            max_iterations=10, seed=42,
            buy_z_thresholds=(1.0, 1.5, 2.0, 2.5, 3.0),
        )
        cfg2 = OptimizerConfig(
            search_method=SearchMethod.RANDOM,
            max_iterations=10, seed=99,
            buy_z_thresholds=(1.0, 1.5, 2.0, 2.5, 3.0),
        )
        c1 = generate_random_configs(cfg1)
        c2 = generate_random_configs(cfg2)
        diffs = sum(1 for a, b in zip(c1, c2) if a.identifier != b.identifier)
        assert diffs > 0


# ---------------------------------------------------------------------------
# Optimizer Integration Tests
# ---------------------------------------------------------------------------


class TestParameterOptimizer:
    def test_grid_optimization(self) -> None:
        data = _make_dataset(n=400)
        cfg = OptimizerConfig(
            search_method=SearchMethod.GRID,
            wavelets=("haar",), windows=(256,), levels=(2,),
            buy_z_thresholds=(1.5, 2.5),
            sell_z_thresholds=(1.5,),
            max_hold_ticks_list=(10,),
            slope_filter_options=(False,),
            constraints=ConstraintConfig(min_trades=0),
        )
        optimizer = ParameterOptimizer(cfg)
        report = optimizer.optimize(data)
        assert isinstance(report, OptimizationReport)
        assert report.total_evaluated == 2
        assert report.search_method == "grid"

    def test_random_optimization(self) -> None:
        data = _make_dataset(n=400)
        cfg = OptimizerConfig(
            search_method=SearchMethod.RANDOM,
            wavelets=("haar",), windows=(256,), levels=(2,),
            buy_z_thresholds=(1.0, 1.5, 2.0, 2.5),
            sell_z_thresholds=(1.0, 1.5, 2.0),
            max_hold_ticks_list=(5, 10),
            slope_filter_options=(False,),
            max_iterations=5,
            constraints=ConstraintConfig(min_trades=0),
        )
        optimizer = ParameterOptimizer(cfg)
        report = optimizer.optimize(data)
        assert report.total_evaluated == 5

    def test_constraint_filtering(self) -> None:
        data = _make_dataset(n=400)
        cfg = OptimizerConfig(
            wavelets=("haar",), windows=(256,), levels=(2,),
            buy_z_thresholds=(1.5,), sell_z_thresholds=(1.5,),
            max_hold_ticks_list=(10,), slope_filter_options=(False,),
            constraints=ConstraintConfig(min_trades=1000),
        )
        optimizer = ParameterOptimizer(cfg)
        report = optimizer.optimize(data)
        assert report.total_passed == 0
        assert report.history.filtered_count == report.total_evaluated

    def test_deterministic(self) -> None:
        data = _make_dataset(n=400)
        cfg = OptimizerConfig(
            wavelets=("haar",), windows=(256,), levels=(2,),
            buy_z_thresholds=(1.5, 2.0),
            sell_z_thresholds=(1.5,),
            max_hold_ticks_list=(10,),
            slope_filter_options=(False,),
            constraints=ConstraintConfig(min_trades=0),
        )
        r1 = ParameterOptimizer(cfg).optimize(data)
        r2 = ParameterOptimizer(cfg).optimize(data)
        assert r1.total_evaluated == r2.total_evaluated
        assert r1.total_passed == r2.total_passed
        if r1.best_configs and r2.best_configs:
            assert r1.best_configs[0].score == r2.best_configs[0].score

    def test_empty_data_raises(self) -> None:
        cfg = OptimizerConfig()
        optimizer = ParameterOptimizer(cfg)
        with pytest.raises(ValueError, match="empty"):
            optimizer.optimize(pd.DataFrame())

    def test_best_configs_sorted(self) -> None:
        data = _make_dataset(n=400)
        cfg = OptimizerConfig(
            wavelets=("haar",), windows=(256,), levels=(2,),
            buy_z_thresholds=(1.0, 1.5, 2.0, 2.5),
            sell_z_thresholds=(1.0,),
            max_hold_ticks_list=(5, 10),
            slope_filter_options=(False,),
            constraints=ConstraintConfig(min_trades=0),
        )
        report = ParameterOptimizer(cfg).optimize(data)
        if len(report.best_configs) > 1:
            scores = [s.score for s in report.best_configs]
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i + 1]
