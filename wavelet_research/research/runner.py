"""Research runner for batch experiment execution.

Orchestrates execution of multiple experiment configurations on a dataset.
"""

from __future__ import annotations

import logging
from typing import Sequence

import pandas as pd

from wavelet_research.research.config import ExperimentConfig
from wavelet_research.research.experiment import (
    ExperimentResult,
    Strategy,
    run_experiment,
)
from wavelet_research.research.metrics import MetricsCollector

logger = logging.getLogger(__name__)


class ResearchRunner:
    """Batch runner for research experiments.

    Executes a list of configurations against a dataset using a provided
    strategy, collecting results for each.

    Parameters
    ----------
    configs : Sequence[ExperimentConfig]
        List of experiment configurations to evaluate.
    strategy : Strategy
        Strategy implementation for experiment execution.
    metrics_collector : MetricsCollector or None
        Shared metrics collector. Uses default if None.
    """

    def __init__(
        self,
        configs: Sequence[ExperimentConfig],
        strategy: Strategy,
        metrics_collector: MetricsCollector | None = None,
    ) -> None:
        if not configs:
            raise ValueError("At least one ExperimentConfig is required.")
        self._configs = list(configs)
        self._strategy = strategy
        self._metrics_collector = metrics_collector or MetricsCollector()

    @property
    def configs(self) -> list[ExperimentConfig]:
        """Configured experiment list."""
        return list(self._configs)

    def run(self, data: pd.DataFrame) -> list[ExperimentResult]:
        """Execute all configured experiments on the provided dataset.

        Parameters
        ----------
        data : pd.DataFrame
            Tick data with at minimum: time, bid, ask, mid columns.

        Returns
        -------
        list[ExperimentResult]
            Results for all experiments (in configuration order).

        Raises
        ------
        ValueError
            If data is empty.
        """
        if data.empty:
            raise ValueError("Dataset is empty.")

        results: list[ExperimentResult] = []

        for index, config in enumerate(self._configs):
            logger.info(
                "Running experiment %d/%d: %s",
                index + 1,
                len(self._configs),
                config.identifier,
            )
            result = run_experiment(
                config=config,
                data=data,
                strategy=self._strategy,
                metrics_collector=self._metrics_collector,
            )
            results.append(result)
            logger.info(
                "Experiment %s complete: %d trades, PF=%.2f, expectancy=%.2f",
                config.identifier,
                result.metrics.trades,
                result.metrics.profit_factor,
                result.metrics.expectancy,
            )

        logger.info("All %d experiments complete.", len(results))
        return results
