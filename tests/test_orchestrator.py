"""Tests for the Experiment Orchestrator (Story 7)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wavelet_research.backtest.config import BacktestConfig, ExitStrategy
from wavelet_research.backtest.core import BacktestEngine
from wavelet_research.backtest.models import BacktestReport
from wavelet_research.engine.config import WaveletEngineConfig
from wavelet_research.engine.core import WaveletEngine
from wavelet_research.engine.models import Tick, WaveletPoint
from wavelet_research.orchestrator.config import PipelineConfig
from wavelet_research.orchestrator.core import ExperimentOrchestrator
from wavelet_research.orchestrator.matrix import ParameterMatrix
from wavelet_research.orchestrator.pipeline import run_pipeline
from wavelet_research.orchestrator.results import (
    ExperimentReport,
    load_results,
    rank_experiments,
    report_to_row,
    save_results,
)
from wavelet_research.signal.config import SignalConfig
from wavelet_research.signal.core import SignalEngine
from wavelet_research.signal.models import Signal, SignalDecision


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


def _make_pipeline_config(
    wavelet: str = "haar",
    window: int = 256,
    level: int = 2,
    buy_z: float = 1.5,
    sell_z: float = 1.5,
    max_hold: int = 10,
) -> PipelineConfig:
    return PipelineConfig(
        wavelet_config=WaveletEngineConfig(
            wavelet=wavelet, window=window, level=level,
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


# ---------------------------------------------------------------------------
# PipelineConfig Tests
# ---------------------------------------------------------------------------


class TestPipelineConfig:
    def test_identifier_unique(self) -> None:
        c1 = _make_pipeline_config(wavelet="haar", buy_z=1.5)
        c2 = _make_pipeline_config(wavelet="db4", buy_z=2.0)
        assert c1.identifier != c2.identifier

    def test_immutable(self) -> None:
        c = _make_pipeline_config()
        with pytest.raises(Exception):
            c.wavelet_config = None  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Parameter Matrix Tests
# ---------------------------------------------------------------------------


class TestParameterMatrix:
    def test_single_combination(self) -> None:
        matrix = ParameterMatrix()
        configs = matrix.generate()
        assert len(configs) == 1
        assert matrix.total_combinations == 1

    def test_multiple_combinations(self) -> None:
        matrix = ParameterMatrix(
            wavelets=("haar", "db4"),
            windows=(256, 512),
            levels=(2, 3),
            buy_z_thresholds=(1.5, 2.5),
        )
        configs = matrix.generate()
        expected = 2 * 2 * 2 * 1 * 2 * 1 * 1 * 1 * 1  # all combos
        assert len(configs) == expected
        assert matrix.total_combinations == expected

    def test_all_configs_valid(self) -> None:
        matrix = ParameterMatrix(
            wavelets=("haar", "db4"),
            windows=(256,),
            levels=(2,),
        )
        configs = matrix.generate()
        for cfg in configs:
            assert isinstance(cfg, PipelineConfig)
            assert cfg.wavelet_config.wavelet in ("haar", "db4")

    def test_empty_wavelets_raises(self) -> None:
        with pytest.raises(ValueError, match="wavelets"):
            ParameterMatrix(wavelets=())

    def test_empty_windows_raises(self) -> None:
        with pytest.raises(ValueError, match="windows"):
            ParameterMatrix(windows=())

    def test_exit_strategy_combinations(self) -> None:
        matrix = ParameterMatrix(
            exit_strategies=(ExitStrategy.MAX_HOLD, ExitStrategy.FIXED_TP_SL),
            max_hold_ticks_list=(10, 50),
        )
        configs = matrix.generate()
        assert len(configs) == 4
        strategies = {c.backtest_config.exit_strategy for c in configs}
        assert strategies == {ExitStrategy.MAX_HOLD, ExitStrategy.FIXED_TP_SL}

    def test_costs_propagated(self) -> None:
        base = BacktestConfig(commission_pips=1.5, slippage_pips=0.5)
        matrix = ParameterMatrix(backtest_config=base)
        configs = matrix.generate()
        for cfg in configs:
            assert cfg.backtest_config.commission_pips == 1.5
            assert cfg.backtest_config.slippage_pips == 0.5


# ---------------------------------------------------------------------------
# Pipeline Execution Tests
# ---------------------------------------------------------------------------


class TestPipeline:
    def test_single_pipeline(self) -> None:
        data = _make_dataset(n=400)
        config = _make_pipeline_config()
        report = run_pipeline(config, data)
        assert isinstance(report, ExperimentReport)
        assert isinstance(report.report, BacktestReport)
        assert report.config == config

    def test_pipeline_deterministic(self) -> None:
        data = _make_dataset(n=400)
        config = _make_pipeline_config()
        r1 = run_pipeline(config, data)
        r2 = run_pipeline(config, data)
        assert r1.report.trades == r2.report.trades
        assert r1.report.total_pnl == r2.report.total_pnl
        assert r1.report.profit_factor == r2.report.profit_factor


# ---------------------------------------------------------------------------
# Result Aggregation Tests
# ---------------------------------------------------------------------------


class TestResults:
    def test_report_to_row(self) -> None:
        data = _make_dataset(n=400)
        config = _make_pipeline_config()
        report = run_pipeline(config, data)
        row = report_to_row(report)
        assert "wavelet" in row
        assert "profit_factor" in row
        assert "trades" in row
        assert "exit_strategy" in row
        assert row["wavelet"] == "haar"

    def test_rank_experiments(self) -> None:
        data = _make_dataset(n=400)
        configs = [
            _make_pipeline_config(buy_z=1.0, sell_z=1.0),
            _make_pipeline_config(buy_z=2.0, sell_z=2.0),
            _make_pipeline_config(buy_z=3.0, sell_z=3.0),
        ]
        reports = [run_pipeline(c, data) for c in configs]
        ranked = rank_experiments(reports)
        assert len(ranked) == 3
        # Best profit factor first
        for i in range(len(ranked) - 1):
            assert ranked[i].report.profit_factor >= ranked[i + 1].report.profit_factor

    def test_rank_empty(self) -> None:
        assert rank_experiments([]) == []

    def test_save_and_load(self, tmp_path: Path) -> None:
        data = _make_dataset(n=400)
        config = _make_pipeline_config()
        report = run_pipeline(config, data)
        filepath = save_results([report], tmp_path)
        assert filepath.exists()
        loaded = load_results(tmp_path)
        assert len(loaded) == 1
        assert "profit_factor" in loaded.columns

    def test_save_empty_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="No experiments"):
            save_results([], tmp_path)

    def test_load_nonexistent_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_results(tmp_path, "nonexistent")


# ---------------------------------------------------------------------------
# Orchestrator Tests
# ---------------------------------------------------------------------------


class TestOrchestrator:
    def test_single_experiment(self) -> None:
        data = _make_dataset(n=400)
        config = _make_pipeline_config()
        orch = ExperimentOrchestrator([config])
        results = orch.run(data, persist=False)
        assert len(results) == 1
        assert results[0].report.trades >= 0

    def test_multiple_experiments(self) -> None:
        data = _make_dataset(n=400)
        configs = [
            _make_pipeline_config(buy_z=1.0, sell_z=1.0),
            _make_pipeline_config(buy_z=2.0, sell_z=2.0),
            _make_pipeline_config(buy_z=3.0, sell_z=3.0),
        ]
        orch = ExperimentOrchestrator(configs)
        results = orch.run(data, persist=False)
        assert len(results) == 3

    def test_results_ranked(self) -> None:
        data = _make_dataset(n=400)
        configs = [
            _make_pipeline_config(buy_z=1.0, sell_z=1.0),
            _make_pipeline_config(buy_z=2.0, sell_z=2.0),
        ]
        orch = ExperimentOrchestrator(configs)
        results = orch.run(data, persist=False)
        for i in range(len(results) - 1):
            assert results[i].report.profit_factor >= results[i + 1].report.profit_factor

    def test_persistence(self, tmp_path: Path) -> None:
        data = _make_dataset(n=400)
        config = _make_pipeline_config()
        orch = ExperimentOrchestrator([config], output_path=tmp_path)
        orch.run(data, persist=True)
        loaded = load_results(tmp_path)
        assert len(loaded) == 1

    def test_empty_configs_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one"):
            ExperimentOrchestrator([])

    def test_empty_data_raises(self) -> None:
        config = _make_pipeline_config()
        orch = ExperimentOrchestrator([config])
        with pytest.raises(ValueError, match="empty"):
            orch.run(pd.DataFrame(), persist=False)

    def test_deterministic_replay(self) -> None:
        data = _make_dataset(n=400)
        configs = [
            _make_pipeline_config(buy_z=1.5, sell_z=1.5),
            _make_pipeline_config(buy_z=2.5, sell_z=2.5),
        ]
        r1 = ExperimentOrchestrator(configs).run(data, persist=False)
        r2 = ExperimentOrchestrator(configs).run(data, persist=False)
        for a, b in zip(r1, r2):
            assert a.report.trades == b.report.trades
            assert a.report.total_pnl == b.report.total_pnl

    def test_matrix_integration(self) -> None:
        data = _make_dataset(n=400)
        matrix = ParameterMatrix(
            wavelets=("haar",),
            windows=(256,),
            levels=(2,),
            buy_z_thresholds=(1.5, 2.5),
            sell_z_thresholds=(1.5,),
            max_hold_ticks_list=(10,),
            slope_filter_options=(False,),
        )
        configs = matrix.generate()
        assert len(configs) == 2
        orch = ExperimentOrchestrator(configs)
        results = orch.run(data, persist=False)
        assert len(results) == 2

    def test_total_experiments(self) -> None:
        configs = [_make_pipeline_config(), _make_pipeline_config(buy_z=2.0)]
        orch = ExperimentOrchestrator(configs)
        assert orch.total_experiments == 2


# ---------------------------------------------------------------------------
# Contract Tests: Interface Compatibility
# ---------------------------------------------------------------------------


class TestContracts:
    """Contract tests validating interfaces between engines."""

    def test_wavelet_engine_produces_wavelet_point(self) -> None:
        """WaveletEngine.update(tick) returns WaveletPoint."""
        cfg = WaveletEngineConfig(wavelet="haar", window=256, level=2)
        engine = WaveletEngine(cfg)
        rng = np.random.default_rng(42)
        values = 1.10 + np.cumsum(rng.normal(0, 0.00001, 256))
        result = None
        for i, v in enumerate(values):
            tick = Tick(
                time=pd.Timestamp("2026-06-01") + pd.Timedelta(seconds=i),
                bid=v - 0.00005, ask=v + 0.00005, mid=v, spread=0.00010,
            )
            result = engine.update(tick)
        assert isinstance(result, WaveletPoint)
        assert hasattr(result, "trend")
        assert hasattr(result, "deviation")
        assert hasattr(result, "z_score")
        assert hasattr(result, "slope")
        assert hasattr(result, "energy")
        assert hasattr(result, "noise")

    def test_signal_engine_accepts_wavelet_point(self) -> None:
        """SignalEngine.decide(WaveletPoint) returns SignalDecision."""
        point = WaveletPoint(
            trend=1.10, deviation=-0.003, z_score=-3.0,
            slope=0.001, energy=0.01, noise=0.001,
        )
        cfg = SignalConfig(buy_z_threshold=2.0, sell_z_threshold=2.0)
        engine = SignalEngine(cfg)
        decision = engine.decide(point)
        assert isinstance(decision, SignalDecision)
        assert decision.signal in (Signal.BUY, Signal.SELL, Signal.HOLD)
        assert hasattr(decision, "confidence")
        assert hasattr(decision, "reason")

    def test_backtest_engine_uses_wavelet_and_signal(self) -> None:
        """BacktestEngine.run() correctly uses WaveletEngine and SignalEngine."""
        data = _make_dataset(n=400)
        wavelet_cfg = WaveletEngineConfig(wavelet="haar", window=256, level=2)
        signal_cfg = SignalConfig(
            buy_z_threshold=1.5, sell_z_threshold=1.5,
            slope_filter_enabled=False,
        )
        backtest_cfg = BacktestConfig(
            pip_size=0.00001, max_hold_ticks=10,
        )
        report = BacktestEngine(backtest_cfg).run(
            data, WaveletEngine(wavelet_cfg), SignalEngine(signal_cfg),
        )
        assert isinstance(report, BacktestReport)
        assert hasattr(report, "trades")
        assert hasattr(report, "journal")
        assert hasattr(report, "equity_curve")

    def test_full_pipeline_contract(self) -> None:
        """Full pipeline: Dataset → WaveletEngine → SignalEngine → BacktestEngine."""
        data = _make_dataset(n=400)
        config = _make_pipeline_config()
        report = run_pipeline(config, data)
        assert isinstance(report, ExperimentReport)
        assert isinstance(report.report, BacktestReport)
        assert report.report.journal.count == report.report.trades

    def test_wavelet_point_fields_match_signal_expectations(self) -> None:
        """WaveletPoint fields are exactly what SignalEngine rules expect."""
        point = WaveletPoint(
            trend=1.10, deviation=0.01, z_score=2.5,
            slope=-0.001, energy=0.01, noise=0.002,
        )
        # Signal rules access these directly
        assert isinstance(point.z_score, float)
        assert isinstance(point.slope, float)
        assert isinstance(point.energy, float)
        assert isinstance(point.noise, float)

    def test_signal_decision_fields_for_backtest(self) -> None:
        """SignalDecision has the fields BacktestEngine needs."""
        decision = SignalDecision(
            signal=Signal.BUY, confidence=0.8, reason="test",
            z_score=-3.0, trend_slope=0.001, energy=0.01, noise=0.001,
        )
        # BacktestEngine checks .signal
        assert decision.signal in (Signal.BUY, Signal.SELL, Signal.HOLD)
        assert isinstance(decision.confidence, float)
