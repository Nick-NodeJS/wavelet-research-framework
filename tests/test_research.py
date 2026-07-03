"""Tests for the Wavelet Research Framework (Story 3)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pytest

from wavelet_research.research.config import ExperimentConfig
from wavelet_research.research.experiment import ExperimentResult, run_experiment
from wavelet_research.research.metrics import MetricsCollector, TradeMetrics
from wavelet_research.research.ranking import (
    DEFAULT_RANKING_CRITERIA,
    RankingCriterion,
    SortDirection,
    rank_results,
)
from wavelet_research.research.runner import ResearchRunner
from wavelet_research.research.storage import ResultStorage, StorageFormat


def _has_parquet_engine() -> bool:
    """Check if a parquet engine is available."""
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import fastparquet  # noqa: F401
        return True
    except ImportError:
        pass
    return False


# ---------------------------------------------------------------------------
# Test Strategy (mock)
# ---------------------------------------------------------------------------


class MockStrategy:
    """A deterministic mock strategy for testing."""

    def __init__(self, pnl_values: Sequence[float]) -> None:
        self._pnl_values = list(pnl_values)

    def execute(
        self, config: ExperimentConfig, data: pd.DataFrame
    ) -> Sequence[float]:
        return self._pnl_values


class ConfigDependentStrategy:
    """Strategy that returns different PnLs based on config."""

    def execute(
        self, config: ExperimentConfig, data: pd.DataFrame
    ) -> Sequence[float]:
        # Use threshold as a multiplier for deterministic variation
        base = [10.0, -5.0, 8.0, -3.0, 12.0]
        return [v * config.signal_threshold for v in base]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_config() -> ExperimentConfig:
    return ExperimentConfig(
        wavelet="haar",
        level=2,
        window=256,
        volatility_window=128,
        normalization="zscore",
        signal_threshold=2.5,
        max_hold=100,
        pip_size=0.00001,
    )


@pytest.fixture
def sample_configs() -> list[ExperimentConfig]:
    return [
        ExperimentConfig(
            wavelet="haar",
            level=2,
            window=256,
            volatility_window=128,
            normalization="zscore",
            signal_threshold=2.0,
        ),
        ExperimentConfig(
            wavelet="db4",
            level=3,
            window=512,
            volatility_window=256,
            normalization="zscore",
            signal_threshold=3.0,
        ),
        ExperimentConfig(
            wavelet="haar",
            level=1,
            window=128,
            volatility_window=64,
            normalization="minmax",
            signal_threshold=1.5,
        ),
    ]


@pytest.fixture
def sample_data() -> pd.DataFrame:
    n = 1000
    rng = np.random.default_rng(42)
    bid = 1.10 + np.cumsum(rng.normal(0, 0.00001, n))
    ask = bid + 0.00005
    return pd.DataFrame({
        "time": pd.date_range("2026-06-01", periods=n, freq="100ms"),
        "bid": bid,
        "ask": ask,
        "mid": (bid + ask) / 2,
    })


@pytest.fixture
def winning_pnl() -> list[float]:
    return [10.0, 5.0, -3.0, 8.0, -2.0, 12.0, -1.0, 7.0]


@pytest.fixture
def losing_pnl() -> list[float]:
    return [-10.0, -5.0, -3.0, -8.0, -2.0, -12.0, -1.0, -7.0]


# ---------------------------------------------------------------------------
# Config Tests
# ---------------------------------------------------------------------------


class TestExperimentConfig:
    def test_valid_config(self, sample_config: ExperimentConfig) -> None:
        assert sample_config.wavelet == "haar"
        assert sample_config.level == 2
        assert sample_config.window == 256

    def test_invalid_level(self) -> None:
        with pytest.raises(ValueError, match="level must be >= 1"):
            ExperimentConfig(
                wavelet="haar", level=0, window=256,
                volatility_window=128, normalization="zscore",
                signal_threshold=2.5,
            )

    def test_invalid_window(self) -> None:
        with pytest.raises(ValueError, match="window must be >= 2"):
            ExperimentConfig(
                wavelet="haar", level=2, window=1,
                volatility_window=128, normalization="zscore",
                signal_threshold=2.5,
            )

    def test_invalid_volatility_window(self) -> None:
        with pytest.raises(ValueError, match="volatility_window must be >= 1"):
            ExperimentConfig(
                wavelet="haar", level=2, window=256,
                volatility_window=0, normalization="zscore",
                signal_threshold=2.5,
            )

    def test_invalid_signal_threshold(self) -> None:
        with pytest.raises(ValueError, match="signal_threshold must be > 0"):
            ExperimentConfig(
                wavelet="haar", level=2, window=256,
                volatility_window=128, normalization="zscore",
                signal_threshold=-1.0,
            )

    def test_invalid_max_hold(self) -> None:
        with pytest.raises(ValueError, match="max_hold must be >= 1"):
            ExperimentConfig(
                wavelet="haar", level=2, window=256,
                volatility_window=128, normalization="zscore",
                signal_threshold=2.5, max_hold=0,
            )

    def test_invalid_pip_size(self) -> None:
        with pytest.raises(ValueError, match="pip_size must be > 0"):
            ExperimentConfig(
                wavelet="haar", level=2, window=256,
                volatility_window=128, normalization="zscore",
                signal_threshold=2.5, pip_size=-0.001,
            )

    def test_identifier_unique(self, sample_configs: list[ExperimentConfig]) -> None:
        identifiers = [c.identifier for c in sample_configs]
        assert len(identifiers) == len(set(identifiers))

    def test_immutable(self, sample_config: ExperimentConfig) -> None:
        with pytest.raises(Exception):
            sample_config.level = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Metrics Tests
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_empty_pnl(self) -> None:
        collector = MetricsCollector()
        metrics = collector.calculate([])
        assert metrics.trades == 0
        assert metrics.win_rate == 0.0
        assert metrics.total_pnl == 0.0

    def test_all_winners(self) -> None:
        collector = MetricsCollector()
        metrics = collector.calculate([10.0, 5.0, 8.0])
        assert metrics.trades == 3
        assert metrics.win_rate == 1.0
        assert metrics.profit_factor == float("inf")
        assert metrics.total_pnl == pytest.approx(23.0)
        assert metrics.max_drawdown == 0.0

    def test_all_losers(self, losing_pnl: list[float]) -> None:
        collector = MetricsCollector()
        metrics = collector.calculate(losing_pnl)
        assert metrics.trades == 8
        assert metrics.win_rate == 0.0
        assert metrics.profit_factor == 0.0
        assert metrics.average_win == 0.0

    def test_mixed_pnl(self, winning_pnl: list[float]) -> None:
        collector = MetricsCollector()
        metrics = collector.calculate(winning_pnl)
        assert metrics.trades == 8
        assert metrics.win_rate == pytest.approx(5 / 8)
        assert metrics.profit_factor > 1.0
        assert metrics.expectancy > 0
        assert metrics.max_drawdown >= 0

    def test_drawdown_computation(self) -> None:
        collector = MetricsCollector()
        # Equity curve: 10, 5, 13, 10 => max drawdown at 10->5 = 5
        metrics = collector.calculate([10.0, -5.0, 8.0, -3.0])
        assert metrics.max_drawdown == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Experiment Tests
# ---------------------------------------------------------------------------


class TestExperiment:
    def test_single_experiment(
        self, sample_config: ExperimentConfig, sample_data: pd.DataFrame
    ) -> None:
        strategy = MockStrategy([10.0, -5.0, 8.0])
        result = run_experiment(sample_config, sample_data, strategy)
        assert isinstance(result, ExperimentResult)
        assert result.config == sample_config
        assert result.metrics.trades == 3
        assert result.pnl_values == (10.0, -5.0, 8.0)

    def test_experiment_empty_trades(
        self, sample_config: ExperimentConfig, sample_data: pd.DataFrame
    ) -> None:
        strategy = MockStrategy([])
        result = run_experiment(sample_config, sample_data, strategy)
        assert result.metrics.trades == 0
        assert result.metrics.total_pnl == 0.0

    def test_experiment_deterministic(
        self, sample_config: ExperimentConfig, sample_data: pd.DataFrame
    ) -> None:
        strategy = MockStrategy([10.0, -5.0, 8.0, -2.0])
        result1 = run_experiment(sample_config, sample_data, strategy)
        result2 = run_experiment(sample_config, sample_data, strategy)
        assert result1.metrics == result2.metrics
        assert result1.pnl_values == result2.pnl_values


# ---------------------------------------------------------------------------
# Runner Tests
# ---------------------------------------------------------------------------


class TestRunner:
    def test_single_config(
        self, sample_config: ExperimentConfig, sample_data: pd.DataFrame
    ) -> None:
        strategy = MockStrategy([10.0, -5.0])
        runner = ResearchRunner([sample_config], strategy)
        results = runner.run(sample_data)
        assert len(results) == 1
        assert results[0].config == sample_config

    def test_multiple_configs(
        self, sample_configs: list[ExperimentConfig], sample_data: pd.DataFrame
    ) -> None:
        strategy = ConfigDependentStrategy()
        runner = ResearchRunner(sample_configs, strategy)
        results = runner.run(sample_data)
        assert len(results) == 3
        # Each result should have different metrics due to different thresholds
        pnls = [r.metrics.total_pnl for r in results]
        assert len(set(pnls)) == 3  # All different

    def test_empty_configs_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one"):
            ResearchRunner([], MockStrategy([]))

    def test_empty_data_raises(
        self, sample_config: ExperimentConfig
    ) -> None:
        strategy = MockStrategy([10.0])
        runner = ResearchRunner([sample_config], strategy)
        with pytest.raises(ValueError, match="Dataset is empty"):
            runner.run(pd.DataFrame())

    def test_deterministic_execution(
        self, sample_configs: list[ExperimentConfig], sample_data: pd.DataFrame
    ) -> None:
        strategy = ConfigDependentStrategy()
        runner = ResearchRunner(sample_configs, strategy)
        results1 = runner.run(sample_data)
        results2 = runner.run(sample_data)
        for r1, r2 in zip(results1, results2):
            assert r1.metrics == r2.metrics
            assert r1.pnl_values == r2.pnl_values


# ---------------------------------------------------------------------------
# Ranking Tests
# ---------------------------------------------------------------------------


class TestRanking:
    def test_rank_by_default_criteria(
        self, sample_configs: list[ExperimentConfig], sample_data: pd.DataFrame
    ) -> None:
        strategy = ConfigDependentStrategy()
        runner = ResearchRunner(sample_configs, strategy)
        results = runner.run(sample_data)
        ranked = rank_results(results)
        # First result should have best profit factor
        for i in range(len(ranked) - 1):
            assert ranked[i].metrics.profit_factor >= ranked[i + 1].metrics.profit_factor

    def test_rank_empty_results(self) -> None:
        ranked = rank_results([])
        assert ranked == []

    def test_rank_custom_criteria(
        self, sample_configs: list[ExperimentConfig], sample_data: pd.DataFrame
    ) -> None:
        strategy = ConfigDependentStrategy()
        runner = ResearchRunner(sample_configs, strategy)
        results = runner.run(sample_data)

        # Rank by total_pnl descending only
        criteria = [RankingCriterion("total_pnl", SortDirection.DESCENDING)]
        ranked = rank_results(results, criteria)
        for i in range(len(ranked) - 1):
            assert ranked[i].metrics.total_pnl >= ranked[i + 1].metrics.total_pnl

    def test_rank_ascending_drawdown(
        self, sample_configs: list[ExperimentConfig], sample_data: pd.DataFrame
    ) -> None:
        strategy = ConfigDependentStrategy()
        runner = ResearchRunner(sample_configs, strategy)
        results = runner.run(sample_data)

        criteria = [RankingCriterion("max_drawdown", SortDirection.ASCENDING)]
        ranked = rank_results(results, criteria)
        for i in range(len(ranked) - 1):
            assert ranked[i].metrics.max_drawdown <= ranked[i + 1].metrics.max_drawdown


# ---------------------------------------------------------------------------
# Storage Tests
# ---------------------------------------------------------------------------


class TestStorage:
    def test_save_csv(
        self,
        tmp_path: Path,
        sample_config: ExperimentConfig,
        sample_data: pd.DataFrame,
    ) -> None:
        strategy = MockStrategy([10.0, -5.0, 8.0])
        result = run_experiment(sample_config, sample_data, strategy)
        storage = ResultStorage(tmp_path, StorageFormat.CSV)
        filepath = storage.save([result])
        assert filepath.exists()
        assert filepath.suffix == ".csv"
        df = pd.read_csv(filepath)
        assert len(df) == 1
        assert "profit_factor" in df.columns
        assert "wavelet" in df.columns

    @pytest.mark.skipif(
        not _has_parquet_engine(), reason="No parquet engine available"
    )
    def test_save_parquet(
        self,
        tmp_path: Path,
        sample_config: ExperimentConfig,
        sample_data: pd.DataFrame,
    ) -> None:
        strategy = MockStrategy([10.0, -5.0, 8.0])
        result = run_experiment(sample_config, sample_data, strategy)
        storage = ResultStorage(tmp_path, StorageFormat.PARQUET)
        filepath = storage.save([result])
        assert filepath.exists()
        assert filepath.suffix == ".parquet"
        df = pd.read_parquet(filepath)
        assert len(df) == 1

    def test_save_multiple_results(
        self,
        tmp_path: Path,
        sample_configs: list[ExperimentConfig],
        sample_data: pd.DataFrame,
    ) -> None:
        strategy = ConfigDependentStrategy()
        runner = ResearchRunner(sample_configs, strategy)
        results = runner.run(sample_data)
        storage = ResultStorage(tmp_path, StorageFormat.CSV)
        filepath = storage.save(results)
        df = pd.read_csv(filepath)
        assert len(df) == 3

    def test_save_empty_raises(self, tmp_path: Path) -> None:
        storage = ResultStorage(tmp_path, StorageFormat.CSV)
        with pytest.raises(ValueError, match="No results"):
            storage.save([])

    def test_load_csv(
        self,
        tmp_path: Path,
        sample_config: ExperimentConfig,
        sample_data: pd.DataFrame,
    ) -> None:
        strategy = MockStrategy([10.0, -5.0, 8.0])
        result = run_experiment(sample_config, sample_data, strategy)
        storage = ResultStorage(tmp_path, StorageFormat.CSV)
        storage.save([result])
        loaded = storage.load()
        assert len(loaded) == 1

    @pytest.mark.skipif(
        not _has_parquet_engine(), reason="No parquet engine available"
    )
    def test_load_parquet(
        self,
        tmp_path: Path,
        sample_config: ExperimentConfig,
        sample_data: pd.DataFrame,
    ) -> None:
        strategy = MockStrategy([10.0, -5.0, 8.0])
        result = run_experiment(sample_config, sample_data, strategy)
        storage = ResultStorage(tmp_path, StorageFormat.PARQUET)
        storage.save([result])
        loaded = storage.load()
        assert len(loaded) == 1

    def test_load_nonexistent_raises(self, tmp_path: Path) -> None:
        storage = ResultStorage(tmp_path, StorageFormat.CSV)
        with pytest.raises(FileNotFoundError):
            storage.load("nonexistent")
