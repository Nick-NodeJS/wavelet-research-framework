"""Core Parameter Optimizer."""

from __future__ import annotations

import logging

import pandas as pd

from wavelet_research.optimizer.config import OptimizerConfig, SearchMethod
from wavelet_research.optimizer.models import (
    OptimizationHistory,
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
from wavelet_research.orchestrator.pipeline import run_pipeline

logger = logging.getLogger(__name__)


class ParameterOptimizer:
    """Deterministic parameter optimization engine.

    Searches the parameter space via grid or random search,
    evaluates each configuration, filters by constraints,
    and ranks by composite objective score.

    Parameters
    ----------
    config : OptimizerConfig
        Optimizer configuration.
    """

    def __init__(self, config: OptimizerConfig) -> None:
        self._config = config

    @property
    def config(self) -> OptimizerConfig:
        """Optimizer configuration."""
        return self._config

    def optimize(self, data: pd.DataFrame) -> OptimizationReport:
        """Run optimization over the dataset.

        Parameters
        ----------
        data : pd.DataFrame
            Normalized tick dataset.

        Returns
        -------
        OptimizationReport
            Complete optimization report.

        Raises
        ------
        ValueError
            If data is empty.
        """
        if data.empty:
            raise ValueError("Dataset is empty.")

        if self._config.search_method == SearchMethod.GRID:
            configs = generate_grid_configs(self._config)
        else:
            configs = generate_random_configs(self._config)

        logger.info(
            "Starting %s optimization with %d configurations",
            self._config.search_method.value, len(configs),
        )

        evaluated: list[ScoredConfig] = []
        passed: list[ScoredConfig] = []

        for idx, pipeline_config in enumerate(configs):
            report = run_pipeline(pipeline_config, data).report
            score = compute_objective_score(report, self._config.objective)

            scored = ScoredConfig(
                config=pipeline_config, report=report, score=score
            )
            evaluated.append(scored)

            if passes_constraints(report, self._config.constraints):
                passed.append(scored)

            if (idx + 1) % 10 == 0:
                logger.debug(
                    "Evaluated %d/%d configs, %d passed",
                    idx + 1, len(configs), len(passed),
                )

        # Rank by score descending
        passed_sorted = sorted(passed, key=lambda s: -s.score)

        history = OptimizationHistory(
            evaluated=tuple(evaluated),
            passed=tuple(passed_sorted),
            filtered_count=len(evaluated) - len(passed),
        )

        best = tuple(passed_sorted[:10])

        logger.info(
            "Optimization complete: %d evaluated, %d passed, best score=%.4f",
            len(evaluated), len(passed),
            best[0].score if best else 0.0,
        )

        return OptimizationReport(
            best_configs=best,
            history=history,
            search_method=self._config.search_method.value,
            total_evaluated=len(evaluated),
            total_passed=len(passed),
        )
