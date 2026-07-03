"""Multi-objective scoring and constraint filtering."""

from __future__ import annotations

import numpy as np

from wavelet_research.backtest.models import BacktestReport
from wavelet_research.optimizer.config import ConstraintConfig, ObjectiveConfig


def passes_constraints(report: BacktestReport, constraints: ConstraintConfig) -> bool:
    """Check if a report meets all hard constraints.

    Parameters
    ----------
    report : BacktestReport
        Backtest results.
    constraints : ConstraintConfig
        Hard constraints.

    Returns
    -------
    bool
        True if all constraints are satisfied.
    """
    if report.trades < constraints.min_trades:
        return False
    if report.max_drawdown > constraints.max_drawdown:
        return False
    if report.profit_factor < constraints.min_profit_factor:
        return False
    if report.expectancy < constraints.min_expectancy:
        return False
    return True


def compute_objective_score(
    report: BacktestReport, objective: ObjectiveConfig
) -> float:
    """Compute composite objective score.

    Higher is better. Drawdown is penalized.

    Parameters
    ----------
    report : BacktestReport
        Backtest results.
    objective : ObjectiveConfig
        Objective weights.

    Returns
    -------
    float
        Composite score.
    """
    pf = min(report.profit_factor, 10.0) if np.isfinite(report.profit_factor) else 10.0
    exp = report.expectancy
    dd_penalty = -report.max_drawdown
    stability = 1.0 - (report.max_drawdown / max(abs(report.total_pnl), 1.0))
    stability = max(0.0, min(1.0, stability))

    score = (
        objective.profit_factor_weight * pf
        + objective.expectancy_weight * exp
        + objective.drawdown_weight * dd_penalty
        + objective.stability_weight * stability
    )
    return float(score)
