"""Metrics computation for the Backtesting Engine.

Computes trading metrics from a list of Trade records.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from wavelet_research.backtest.models import (
    BacktestReport,
    EquityCurve,
    Trade,
    TradeJournal,
)


def _compute_max_drawdown(equity_values: Sequence[float]) -> float:
    """Compute maximum drawdown from an equity curve.

    Parameters
    ----------
    equity_values : Sequence[float]
        Cumulative equity values.

    Returns
    -------
    float
        Maximum peak-to-trough drawdown (positive value).
    """
    if not equity_values:
        return 0.0
    equity = np.asarray(equity_values, dtype=np.float64)
    running_max = np.maximum.accumulate(equity)
    drawdown = running_max - equity
    return float(np.max(drawdown))


def compute_report(trades: list[Trade]) -> BacktestReport:
    """Compute a complete backtest report from trade records.

    Parameters
    ----------
    trades : list[Trade]
        Completed trades.

    Returns
    -------
    BacktestReport
        Full report with metrics, journal, and equity curve.
    """
    journal = TradeJournal(trades=tuple(trades))

    if not trades:
        return BacktestReport(
            trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            total_pnl=0.0,
            max_drawdown=0.0,
            average_mae=0.0,
            average_mfe=0.0,
            average_trade=0.0,
            average_holding_time=0.0,
            journal=journal,
            equity_curve=EquityCurve(),
        )

    pnl_values = np.asarray([t.pnl_pips for t in trades], dtype=np.float64)
    equity_values = np.cumsum(pnl_values)
    equity_curve = EquityCurve(values=tuple(float(v) for v in equity_values))

    wins = pnl_values[pnl_values > 0]
    losses = pnl_values[pnl_values < 0]

    gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_loss = abs(float(np.sum(losses))) if len(losses) > 0 else 0.0
    profit_factor = (
        gross_profit / gross_loss if gross_loss > 0 else float("inf")
    )

    mae_values = [t.mae_pips for t in trades]
    mfe_values = [t.mfe_pips for t in trades]
    holding_values = [t.holding_ticks for t in trades]

    return BacktestReport(
        trades=len(trades),
        win_rate=float(len(wins) / len(pnl_values)),
        profit_factor=float(profit_factor),
        expectancy=float(np.mean(pnl_values)),
        total_pnl=float(np.sum(pnl_values)),
        max_drawdown=_compute_max_drawdown(equity_values.tolist()),
        average_mae=float(np.mean(mae_values)),
        average_mfe=float(np.mean(mfe_values)),
        average_trade=float(np.mean(pnl_values)),
        average_holding_time=float(np.mean(holding_values)),
        journal=journal,
        equity_curve=equity_curve,
    )
