"""Domain models for the optimization engine."""

from __future__ import annotations

from dataclasses import dataclass

from wavelet_research.backtest.models import BacktestReport
from wavelet_research.orchestrator.config import PipelineConfig


@dataclass(frozen=True)
class ScoredConfig:
    """A pipeline configuration with its optimization score.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration.
    report : BacktestReport
        Backtest report.
    score : float
        Composite objective score.
    """

    config: PipelineConfig
    report: BacktestReport
    score: float


@dataclass(frozen=True)
class OptimizationHistory:
    """History of all evaluated configurations.

    Parameters
    ----------
    evaluated : tuple[ScoredConfig, ...]
        All evaluated configurations (including filtered).
    passed : tuple[ScoredConfig, ...]
        Configurations that passed constraints.
    filtered_count : int
        Number filtered by constraints.
    """

    evaluated: tuple[ScoredConfig, ...]
    passed: tuple[ScoredConfig, ...]
    filtered_count: int


@dataclass(frozen=True)
class OptimizationReport:
    """Complete optimization report.

    Parameters
    ----------
    best_configs : tuple[ScoredConfig, ...]
        Top configurations ranked by score.
    history : OptimizationHistory
        Full optimization history.
    search_method : str
        Search method used.
    total_evaluated : int
        Total configurations evaluated.
    total_passed : int
        Configurations passing constraints.
    """

    best_configs: tuple[ScoredConfig, ...]
    history: OptimizationHistory
    search_method: str
    total_evaluated: int
    total_passed: int
