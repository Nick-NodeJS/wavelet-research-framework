"""Metrics collection for research experiments.

Collects standard trading metrics from a sequence of PnL values.
Extensible: add new metrics by subclassing MetricsCollector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class TradeMetrics:
    """Computed trading metrics for a single experiment.

    Parameters
    ----------
    trades : int
        Total number of trades.
    win_rate : float
        Fraction of winning trades.
    profit_factor : float
        Gross profit / gross loss.
    expectancy : float
        Average PnL per trade (in pips).
    max_drawdown : float
        Maximum peak-to-trough drawdown (in pips).
    total_pnl : float
        Total PnL (in pips).
    average_trade : float
        Average PnL per trade (same as expectancy, explicit).
    average_win : float
        Average PnL of winning trades.
    average_loss : float
        Average PnL of losing trades.
    """

    trades: int
    win_rate: float
    profit_factor: float
    expectancy: float
    max_drawdown: float
    total_pnl: float
    average_trade: float
    average_win: float
    average_loss: float


class MetricsCalculator(Protocol):
    """Protocol for metrics calculation strategies."""

    def calculate(self, pnl_values: Sequence[float]) -> TradeMetrics:
        """Calculate metrics from PnL values."""
        ...


def _compute_max_drawdown(pnl_values: np.ndarray) -> float:
    """Compute maximum drawdown from a series of PnL values.

    Parameters
    ----------
    pnl_values : np.ndarray
        Array of per-trade PnL values.

    Returns
    -------
    float
        Maximum peak-to-trough drawdown (positive value).
    """
    equity = np.cumsum(pnl_values)
    running_max = np.maximum.accumulate(equity)
    drawdown = running_max - equity
    return float(np.max(drawdown)) if len(drawdown) > 0 else 0.0


class MetricsCollector:
    """Standard metrics collector for research experiments.

    Computes all required trading metrics from a list of PnL values.
    """

    def calculate(self, pnl_values: Sequence[float]) -> TradeMetrics:
        """Calculate trading metrics from per-trade PnL values.

        Parameters
        ----------
        pnl_values : Sequence[float]
            Per-trade PnL values in pips.

        Returns
        -------
        TradeMetrics
            Computed metrics.
        """
        if len(pnl_values) == 0:
            return TradeMetrics(
                trades=0,
                win_rate=0.0,
                profit_factor=0.0,
                expectancy=0.0,
                max_drawdown=0.0,
                total_pnl=0.0,
                average_trade=0.0,
                average_win=0.0,
                average_loss=0.0,
            )

        pnl = np.asarray(pnl_values, dtype=np.float64)
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]

        gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
        gross_loss = abs(float(np.sum(losses))) if len(losses) > 0 else 0.0

        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else float("inf")
        )

        return TradeMetrics(
            trades=len(pnl),
            win_rate=float(len(wins) / len(pnl)),
            profit_factor=float(profit_factor),
            expectancy=float(np.mean(pnl)),
            max_drawdown=_compute_max_drawdown(pnl),
            total_pnl=float(np.sum(pnl)),
            average_trade=float(np.mean(pnl)),
            average_win=float(np.mean(wins)) if len(wins) > 0 else 0.0,
            average_loss=float(np.mean(losses)) if len(losses) > 0 else 0.0,
        )
