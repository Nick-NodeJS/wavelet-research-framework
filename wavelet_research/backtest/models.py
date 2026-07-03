"""Domain models for the Backtesting Engine."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class Trade:
    """A completed trade record.

    Parameters
    ----------
    entry_index : int
        Tick index at entry.
    exit_index : int
        Tick index at exit.
    side : str
        Trade direction: 'BUY' or 'SELL'.
    entry_price : float
        Execution price at entry.
    exit_price : float
        Execution price at exit.
    pnl_pips : float
        Realized PnL in pips (after costs).
    gross_pnl_pips : float
        Gross PnL in pips (before costs).
    mae_pips : float
        Maximum Adverse Excursion in pips.
    mfe_pips : float
        Maximum Favorable Excursion in pips.
    holding_ticks : int
        Number of ticks the position was held.
    exit_reason : str
        Reason for exit.
    entry_time : pd.Timestamp | None
        Entry timestamp.
    exit_time : pd.Timestamp | None
        Exit timestamp.
    """

    entry_index: int
    exit_index: int
    side: str
    entry_price: float
    exit_price: float
    pnl_pips: float
    gross_pnl_pips: float
    mae_pips: float
    mfe_pips: float
    holding_ticks: int
    exit_reason: str
    entry_time: pd.Timestamp | None = None
    exit_time: pd.Timestamp | None = None


@dataclass(frozen=True)
class TradeJournal:
    """Collection of all trades from a backtest run.

    Parameters
    ----------
    trades : tuple[Trade, ...]
        All completed trades in chronological order.
    """

    trades: tuple[Trade, ...] = ()

    @property
    def count(self) -> int:
        """Number of trades."""
        return len(self.trades)

    def pnl_values(self) -> list[float]:
        """Extract PnL values from all trades.

        Returns
        -------
        list[float]
            Per-trade PnL in pips.
        """
        return [t.pnl_pips for t in self.trades]


@dataclass(frozen=True)
class EquityCurve:
    """Equity curve from a backtest run.

    Parameters
    ----------
    values : tuple[float, ...]
        Cumulative equity values after each trade.
    """

    values: tuple[float, ...] = ()

    @property
    def peak(self) -> float:
        """Maximum equity reached."""
        return max(self.values) if self.values else 0.0

    @property
    def final(self) -> float:
        """Final equity value."""
        return self.values[-1] if self.values else 0.0


@dataclass(frozen=True)
class BacktestReport:
    """Complete report from a backtest run.

    Parameters
    ----------
    trades : int
        Total number of trades.
    win_rate : float
        Fraction of winning trades.
    profit_factor : float
        Gross profit / gross loss.
    expectancy : float
        Average PnL per trade in pips.
    total_pnl : float
        Total PnL in pips.
    max_drawdown : float
        Maximum peak-to-trough drawdown in pips.
    average_mae : float
        Average Maximum Adverse Excursion in pips.
    average_mfe : float
        Average Maximum Favorable Excursion in pips.
    average_trade : float
        Average PnL per trade in pips.
    average_holding_time : float
        Average holding time in ticks.
    journal : TradeJournal
        Complete trade journal.
    equity_curve : EquityCurve
        Equity curve.
    """

    trades: int
    win_rate: float
    profit_factor: float
    expectancy: float
    total_pnl: float
    max_drawdown: float
    average_mae: float
    average_mfe: float
    average_trade: float
    average_holding_time: float
    journal: TradeJournal
    equity_curve: EquityCurve
