"""Core Walk-Forward Validator.

Orchestrates validation: IS/OOS splits, walk-forward folds,
metrics computation, Monte Carlo, and robustness analysis.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from wavelet_research.backtest.models import BacktestReport
from wavelet_research.orchestrator.config import PipelineConfig
from wavelet_research.orchestrator.pipeline import run_pipeline
from wavelet_research.validation.metrics import compute_extended_metrics, ExtendedMetrics
from wavelet_research.validation.models import (
    RobustnessReport,
    ValidationReport,
    ValidationSplit,
    WalkForwardResult,
)
from wavelet_research.validation.monte_carlo import run_monte_carlo
from wavelet_research.validation.robustness import compute_robustness
from wavelet_research.validation.splits import (
    SplitConfig,
    generate_walk_forward_splits,
    split_in_out_of_sample,
)

logger = logging.getLogger(__name__)


def _report_to_extended_metrics(report: BacktestReport) -> ExtendedMetrics:
    """Extract ExtendedMetrics from a BacktestReport.

    Parameters
    ----------
    report : BacktestReport
        Backtest report.

    Returns
    -------
    ExtendedMetrics
        Extended metrics.
    """
    pnl_values = report.journal.pnl_values()
    equity_values = list(report.equity_curve.values)
    mae_values = [t.mae_pips for t in report.journal.trades]
    mfe_values = [t.mfe_pips for t in report.journal.trades]

    return compute_extended_metrics(
        pnl_values=pnl_values,
        equity_values=equity_values,
        mae_values=mae_values,
        mfe_values=mfe_values,
        max_drawdown=report.max_drawdown,
    )


def _empty_extended_metrics() -> ExtendedMetrics:
    """Create empty ExtendedMetrics for no-trade scenarios."""
    return ExtendedMetrics(
        sharpe_ratio=0.0, sortino_ratio=0.0,
        recovery_factor=0.0, stability_score=0.0,
        profit_factor=0.0, expectancy=0.0,
        max_drawdown=0.0, total_pnl=0.0,
        trades=0, win_rate=0.0,
        average_mae=0.0, average_mfe=0.0,
    )


class WalkForwardValidator:
    """Walk-forward validation framework.

    Runs a PipelineConfig through IS/OOS splits and walk-forward
    folds, computing extended metrics for each.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration to validate.
    split_config : SplitConfig
        Split configuration.
    """

    def __init__(
        self, config: PipelineConfig, split_config: SplitConfig
    ) -> None:
        self._config = config
        self._split_config = split_config

    @property
    def config(self) -> PipelineConfig:
        """Pipeline configuration."""
        return self._config

    def validate(self, data: pd.DataFrame) -> ValidationReport:
        """Run full validation: IS/OOS + walk-forward.

        Parameters
        ----------
        data : pd.DataFrame
            Full normalized tick dataset.

        Returns
        -------
        ValidationReport
            Complete validation report.

        Raises
        ------
        ValueError
            If data is empty.
        """
        if data.empty:
            raise ValueError("Data is empty.")

        # IS/OOS split
        is_data, oos_data = split_in_out_of_sample(
            data, self._split_config.in_sample_ratio
        )

        is_report = run_pipeline(self._config, is_data)
        is_metrics = _report_to_extended_metrics(is_report.report)

        oos_report = run_pipeline(self._config, oos_data)
        oos_metrics = _report_to_extended_metrics(oos_report.report)

        logger.info(
            "IS: %d trades, PnL=%.2f | OOS: %d trades, PnL=%.2f",
            is_metrics.trades, is_metrics.total_pnl,
            oos_metrics.trades, oos_metrics.total_pnl,
        )

        # Walk-forward folds
        splits = generate_walk_forward_splits(data, self._split_config)
        wf_results: list[WalkForwardResult] = []

        for fold_idx, (fold_is, fold_oos) in enumerate(splits):
            fold_is_report = run_pipeline(self._config, fold_is)
            fold_is_metrics = _report_to_extended_metrics(fold_is_report.report)

            fold_oos_report = run_pipeline(self._config, fold_oos)
            fold_oos_metrics = _report_to_extended_metrics(fold_oos_report.report)

            oos_eff = (
                fold_oos_metrics.total_pnl / fold_is_metrics.total_pnl
                if fold_is_metrics.total_pnl != 0 else 0.0
            )

            wf_results.append(WalkForwardResult(
                fold_index=fold_idx,
                in_sample=ValidationSplit(
                    split_name=f"fold_{fold_idx}_is", metrics=fold_is_metrics
                ),
                out_of_sample=ValidationSplit(
                    split_name=f"fold_{fold_idx}_oos", metrics=fold_oos_metrics
                ),
                oos_efficiency=oos_eff,
            ))

            logger.debug(
                "Fold %d: IS PnL=%.2f, OOS PnL=%.2f, efficiency=%.2f",
                fold_idx, fold_is_metrics.total_pnl,
                fold_oos_metrics.total_pnl, oos_eff,
            )

        mean_eff = (
            float(np.mean([r.oos_efficiency for r in wf_results]))
            if wf_results else 0.0
        )

        return ValidationReport(
            in_sample_metrics=is_metrics,
            out_of_sample_metrics=oos_metrics,
            walk_forward_results=tuple(wf_results),
            mean_oos_efficiency=mean_eff,
        )

    def monte_carlo(
        self,
        data: pd.DataFrame,
        n_simulations: int = 1000,
        seed: int = 42,
    ) -> RobustnessReport:
        """Run Monte Carlo analysis on the strategy.

        Parameters
        ----------
        data : pd.DataFrame
            Full dataset.
        n_simulations : int
            Number of simulations.
        seed : int
            Random seed.

        Returns
        -------
        RobustnessReport
            Robustness report.

        Raises
        ------
        ValueError
            If data is empty or no trades produced.
        """
        if data.empty:
            raise ValueError("Data is empty.")

        report = run_pipeline(self._config, data)
        pnl_values = report.report.journal.pnl_values()

        if not pnl_values:
            raise ValueError("No trades produced for Monte Carlo analysis.")

        mc_results = run_monte_carlo(pnl_values, n_simulations, seed)
        robustness = compute_robustness(report.report.total_pnl, mc_results)

        return RobustnessReport(
            robustness=robustness,
            n_simulations=n_simulations,
        )
