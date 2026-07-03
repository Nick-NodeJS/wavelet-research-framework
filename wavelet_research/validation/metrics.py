"""Extended validation metrics: Sharpe, Sortino, recovery factor, stability."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExtendedMetrics:
    """Extended performance metrics for validation.

    Parameters
    ----------
    sharpe_ratio : float
        Risk-adjusted return (annualized concept, here per-trade).
    sortino_ratio : float
        Downside-risk-adjusted return.
    recovery_factor : float
        Total PnL / max drawdown.
    stability_score : float
        R-squared of equity curve vs linear fit [0.0, 1.0].
    profit_factor : float
        Gross profit / gross loss.
    expectancy : float
        Average PnL per trade.
    max_drawdown : float
        Maximum drawdown.
    total_pnl : float
        Total PnL.
    trades : int
        Number of trades.
    win_rate : float
        Win rate.
    average_mae : float
        Average MAE.
    average_mfe : float
        Average MFE.
    """

    sharpe_ratio: float
    sortino_ratio: float
    recovery_factor: float
    stability_score: float
    profit_factor: float
    expectancy: float
    max_drawdown: float
    total_pnl: float
    trades: int
    win_rate: float
    average_mae: float
    average_mfe: float


def compute_sharpe(pnl_values: np.ndarray) -> float:
    """Compute Sharpe ratio from per-trade PnL.

    Parameters
    ----------
    pnl_values : np.ndarray
        Per-trade PnL values.

    Returns
    -------
    float
        Sharpe ratio (mean / std). Returns 0.0 if insufficient data.
    """
    if len(pnl_values) < 2:
        return 0.0
    std = float(np.std(pnl_values, ddof=1))
    if std == 0:
        return 0.0
    return float(np.mean(pnl_values)) / std


def compute_sortino(pnl_values: np.ndarray) -> float:
    """Compute Sortino ratio from per-trade PnL.

    Parameters
    ----------
    pnl_values : np.ndarray
        Per-trade PnL values.

    Returns
    -------
    float
        Sortino ratio (mean / downside_std). Returns 0.0 if no downside.
    """
    if len(pnl_values) < 2:
        return 0.0
    downside = pnl_values[pnl_values < 0]
    if len(downside) == 0:
        return float("inf") if float(np.mean(pnl_values)) > 0 else 0.0
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else abs(float(downside[0]))
    if downside_std == 0:
        return 0.0
    return float(np.mean(pnl_values)) / downside_std


def compute_recovery_factor(total_pnl: float, max_drawdown: float) -> float:
    """Compute recovery factor.

    Parameters
    ----------
    total_pnl : float
        Total PnL.
    max_drawdown : float
        Max drawdown (positive value).

    Returns
    -------
    float
        total_pnl / max_drawdown.
    """
    if max_drawdown == 0:
        return float("inf") if total_pnl > 0 else 0.0
    return total_pnl / max_drawdown


def compute_stability_score(equity_values: np.ndarray) -> float:
    """Compute stability score as R-squared of equity curve vs linear fit.

    Parameters
    ----------
    equity_values : np.ndarray
        Cumulative equity values.

    Returns
    -------
    float
        R-squared in [0.0, 1.0]. Higher = more stable.
    """
    if len(equity_values) < 2:
        return 0.0
    x = np.arange(len(equity_values), dtype=np.float64)
    y = np.asarray(equity_values, dtype=np.float64)

    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0:
        return 1.0

    coeffs = np.polyfit(x, y, 1)
    y_fit = np.polyval(coeffs, x)
    ss_res = float(np.sum((y - y_fit) ** 2))
    r_squared = 1.0 - ss_res / ss_tot
    return max(0.0, min(1.0, r_squared))


def compute_extended_metrics(
    pnl_values: list[float],
    equity_values: list[float],
    mae_values: list[float],
    mfe_values: list[float],
    max_drawdown: float,
) -> ExtendedMetrics:
    """Compute all extended metrics from trade data.

    Parameters
    ----------
    pnl_values : list[float]
        Per-trade PnL.
    equity_values : list[float]
        Cumulative equity.
    mae_values : list[float]
        Per-trade MAE.
    mfe_values : list[float]
        Per-trade MFE.
    max_drawdown : float
        Max drawdown.

    Returns
    -------
    ExtendedMetrics
        Complete extended metrics.
    """
    pnl = np.asarray(pnl_values, dtype=np.float64)
    equity = np.asarray(equity_values, dtype=np.float64)

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_loss = abs(float(np.sum(losses))) if len(losses) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    total_pnl = float(np.sum(pnl)) if len(pnl) > 0 else 0.0

    return ExtendedMetrics(
        sharpe_ratio=compute_sharpe(pnl),
        sortino_ratio=compute_sortino(pnl),
        recovery_factor=compute_recovery_factor(total_pnl, max_drawdown),
        stability_score=compute_stability_score(equity),
        profit_factor=float(profit_factor),
        expectancy=float(np.mean(pnl)) if len(pnl) > 0 else 0.0,
        max_drawdown=max_drawdown,
        total_pnl=total_pnl,
        trades=len(pnl),
        win_rate=float(len(wins) / len(pnl)) if len(pnl) > 0 else 0.0,
        average_mae=float(np.mean(mae_values)) if mae_values else 0.0,
        average_mfe=float(np.mean(mfe_values)) if mfe_values else 0.0,
    )
