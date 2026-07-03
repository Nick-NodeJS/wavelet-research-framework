"""Core Experiment Orchestrator for batch pipeline execution.

Executes multiple PipelineConfigs over a dataset, collects results,
ranks them, and persists to disk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import pandas as pd

from wavelet_research.orchestrator.config import PipelineConfig
from wavelet_research.orchestrator.pipeline import run_pipeline
from wavelet_research.orchestrator.results import (
    ExperimentReport,
    rank_experiments,
    save_results,
)

logger = logging.getLogger(__name__)


class ExperimentOrchestrator:
    """Orchestrates batch execution of research experiments.

    Runs each PipelineConfig through the full pipeline, collects
    results, ranks them, and optionally persists to disk.

    Parameters
    ----------
    configs : Sequence[PipelineConfig]
        Pipeline configurations to execute.
    output_path : Path or str or None
        Optional output directory for result persistence.

    Raises
    ------
    ValueError
        If configs is empty.
    """

    def __init__(
        self,
        configs: Sequence[PipelineConfig],
        output_path: Path | str | None = None,
    ) -> None:
        if not configs:
            raise ValueError("At least one PipelineConfig is required.")
        self._configs = list(configs)
        self._output_path = Path(output_path) if output_path else None

    @property
    def configs(self) -> list[PipelineConfig]:
        """Configured pipeline list."""
        return list(self._configs)

    @property
    def total_experiments(self) -> int:
        """Number of experiments to run."""
        return len(self._configs)

    def run(
        self, data: pd.DataFrame, persist: bool = True
    ) -> list[ExperimentReport]:
        """Execute all experiments and return ranked results.

        Parameters
        ----------
        data : pd.DataFrame
            Normalized tick data.
        persist : bool
            If True and output_path is set, save results to disk.

        Returns
        -------
        list[ExperimentReport]
            Ranked experiment reports (best first).

        Raises
        ------
        ValueError
            If data is empty.
        """
        if data.empty:
            raise ValueError("Dataset is empty.")

        results: list[ExperimentReport] = []

        for index, config in enumerate(self._configs):
            logger.info(
                "Running experiment %d/%d: %s",
                index + 1, len(self._configs), config.identifier,
            )
            try:
                report = run_pipeline(config, data)
                results.append(report)
                logger.info(
                    "Experiment %d complete: %d trades, PF=%.2f, PnL=%.2f",
                    index + 1, report.report.trades,
                    report.report.profit_factor, report.report.total_pnl,
                )
            except Exception:
                logger.exception(
                    "Experiment %d failed: %s", index + 1, config.identifier,
                )
                raise

        ranked = rank_experiments(results)

        if persist and self._output_path is not None and ranked:
            filepath = save_results(ranked, self._output_path)
            logger.info("Results saved to %s", filepath)

        logger.info(
            "All %d experiments complete. Best PF=%.2f",
            len(ranked),
            ranked[0].report.profit_factor if ranked else 0.0,
        )

        return ranked
