"""Paper trade journal for tracking trades and metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PaperTrade:
    """A completed paper trade.

    Parameters
    ----------
    direction : str
        'BUY' or 'SELL'.
    entry_price : float
        Entry price.
    exit_price : float
        Exit price.
    lots : float
        Position size.
    pnl : float
        Realized PnL.
    entry_time : pd.Timestamp
        Entry timestamp.
    exit_time : pd.Timestamp
        Exit timestamp.
    exit_reason : str
        Reason for exit.
    """

    direction: str
    entry_price: float
    exit_price: float
    lots: float
    pnl: float
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    exit_reason: str


@dataclass
class PaperTradeJournal:
    """Journal tracking all paper trades with metrics.

    Parameters
    ----------
    trades : list[PaperTrade]
        All completed trades.
    """

    trades: list[PaperTrade] = field(default_factory=list)

    @property
    def count(self) -> int:
        """Total trade count."""
        return len(self.trades)

    @property
    def total_pnl(self) -> float:
        """Total realized PnL."""
        return sum(t.pnl for t in self.trades)

    @property
    def win_rate(self) -> float:
        """Win rate fraction."""
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.pnl > 0)
        return wins / len(self.trades)

    @property
    def profit_factor(self) -> float:
        """Gross profit / gross loss."""
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown from equity curve."""
        if not self.trades:
            return 0.0
        equity = np.cumsum([t.pnl for t in self.trades])
        running_max = np.maximum.accumulate(equity)
        drawdown = running_max - equity
        return float(np.max(drawdown))

    @property
    def expectancy(self) -> float:
        """Average PnL per trade."""
        if not self.trades:
            return 0.0
        return self.total_pnl / len(self.trades)

    def add_trade(self, trade: PaperTrade) -> None:
        """Add a completed trade.

        Parameters
        ----------
        trade : PaperTrade
            Completed trade.
        """
        self.trades.append(trade)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert journal to DataFrame.

        Returns
        -------
        pd.DataFrame
            Trade journal as DataFrame.
        """
        if not self.trades:
            return pd.DataFrame()
        rows = [
            {
                "direction": t.direction,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "lots": t.lots,
                "pnl": t.pnl,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "exit_reason": t.exit_reason,
            }
            for t in self.trades
        ]
        return pd.DataFrame(rows)
