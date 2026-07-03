"""Experiment execution and result model.

Defines the ExperimentResult and the Strategy protocol for pluggable
experiment execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import pandas as pd

from wavelet_research.research.config import ExperimentConfig
from wavelet_research.research.metrics import MetricsCollector, TradeMetrics


@dataclass(frozen=True)
class ExperimentResult:
    """Result of a single experiment execution.

    Parameters
    ----------
    config : ExperimentConfig
        The configuration that produced this result.
    metrics : TradeMetrics
        Computed trading metrics.
    pnl_values : tuple[float, ...]
        Per-trade PnL values (stored for reproducibility).
    """

    config: ExperimentConfig
    metrics: TradeMetrics
    pnl_values: tuple[float, ...]


class Strategy(Protocol):
    """Protocol for experiment strategies.

    Any callable or object implementing this protocol can be used
    as a strategy in the research framework. This keeps the framework
    independent of any specific wavelet or trading implementation.
    """

    def execute(
        self, config: ExperimentConfig, data: pd.DataFrame
    ) -> Sequence[float]:
        """Execute a strategy and return per-trade PnL values.

        Parameters
        ----------
        config : ExperimentConfig
            Experiment configuration to use.
        data : pd.DataFrame
            Tick data with at minimum: time, bid, ask, mid columns.

        Returns
        -------
        Sequence[float]
            Per-trade PnL values in pips.
        """
        ...


def run_experiment(
    config: ExperimentConfig,
    data: pd.DataFrame,
    strategy: Strategy,
    metrics_collector: MetricsCollector | None = None,
) -> ExperimentResult:
    """Execute a single experiment.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration.
    data : pd.DataFrame
        Tick data.
    strategy : Strategy
        Strategy implementing the execution logic.
    metrics_collector : MetricsCollector or None
        Metrics collector instance. Uses default if None.

    Returns
    -------
    ExperimentResult
        The result containing config, metrics, and PnL values.
    """
    if metrics_collector is None:
        metrics_collector = MetricsCollector()

    pnl_values = strategy.execute(config, data)
    pnl_tuple = tuple(float(v) for v in pnl_values)
    metrics = metrics_collector.calculate(pnl_tuple)

    return ExperimentResult(
        config=config,
        metrics=metrics,
        pnl_values=pnl_tuple,
    )
