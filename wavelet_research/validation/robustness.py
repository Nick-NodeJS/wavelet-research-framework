"""Robustness analysis from Monte Carlo simulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wavelet_research.validation.monte_carlo import MonteCarloResult


def compute_robustness(
    original_pnl: float,
    mc_results: list[MonteCarloResult],
) -> RobustnessMetrics:
    """Analyze robustness of a strategy from Monte Carlo results.

    Parameters
    ----------
    original_pnl : float
        Original strategy total PnL.
    mc_results : list[MonteCarloResult]
        Monte Carlo simulation results.

    Returns
    -------
    RobustnessMetrics
        Robustness analysis.
    """
    if not mc_results:
        return RobustnessMetrics(
            percentile_5_pnl=0.0,
            percentile_25_pnl=0.0,
            percentile_50_pnl=0.0,
            percentile_75_pnl=0.0,
            percentile_95_pnl=0.0,
            mean_drawdown=0.0,
            worst_drawdown=0.0,
            probability_of_profit=0.0,
            original_vs_median=0.0,
        )

    pnls = np.asarray([r.total_pnl for r in mc_results], dtype=np.float64)
    dds = np.asarray([r.max_drawdown for r in mc_results], dtype=np.float64)

    median_pnl = float(np.percentile(pnls, 50))
    original_vs_median = (
        (original_pnl - median_pnl) / abs(median_pnl)
        if median_pnl != 0 else 0.0
    )

    return RobustnessMetrics(
        percentile_5_pnl=float(np.percentile(pnls, 5)),
        percentile_25_pnl=float(np.percentile(pnls, 25)),
        percentile_50_pnl=float(np.percentile(pnls, 50)),
        percentile_75_pnl=float(np.percentile(pnls, 75)),
        percentile_95_pnl=float(np.percentile(pnls, 95)),
        mean_drawdown=float(np.mean(dds)),
        worst_drawdown=float(np.max(dds)),
        probability_of_profit=float(np.mean(pnls > 0)),
        original_vs_median=original_vs_median,
    )


@dataclass(frozen=True)
class RobustnessMetrics:
    """Robustness metrics derived from Monte Carlo analysis.

    Parameters
    ----------
    percentile_5_pnl : float
        5th percentile of shuffled PnL distribution.
    percentile_25_pnl : float
        25th percentile.
    percentile_50_pnl : float
        Median (50th percentile).
    percentile_75_pnl : float
        75th percentile.
    percentile_95_pnl : float
        95th percentile.
    mean_drawdown : float
        Mean max drawdown across simulations.
    worst_drawdown : float
        Worst max drawdown across simulations.
    probability_of_profit : float
        Fraction of simulations with positive PnL.
    original_vs_median : float
        How the original PnL compares to median shuffle.
    """

    percentile_5_pnl: float
    percentile_25_pnl: float
    percentile_50_pnl: float
    percentile_75_pnl: float
    percentile_95_pnl: float
    mean_drawdown: float
    worst_drawdown: float
    probability_of_profit: float
    original_vs_median: float
