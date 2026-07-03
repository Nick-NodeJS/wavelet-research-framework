"""Paper Trader: orchestrates market replay with EA for paper trading."""

from __future__ import annotations

import logging

import pandas as pd

from wavelet_research.engine.models import Tick
from wavelet_research.mt5.expert_advisor import EAConfig, MT5ExpertAdvisor
from wavelet_research.paper_trading.journal import PaperTrade, PaperTradeJournal
from wavelet_research.paper_trading.replay import MarketReplay

logger = logging.getLogger(__name__)


class PaperTrader:
    """Paper trading simulator using market replay and EA.

    Runs a full paper trading session by replaying market data
    through the Expert Advisor and managing virtual positions.

    Parameters
    ----------
    ea_config : EAConfig
        Expert Advisor configuration.
    initial_balance : float
        Starting account balance.
    """

    def __init__(
        self, ea_config: EAConfig, initial_balance: float = 10000.0
    ) -> None:
        self._ea = MT5ExpertAdvisor(ea_config, initial_balance)
        self._journal = PaperTradeJournal()
        self._initial_balance = initial_balance
        self._pending_entry: tuple[str, float, float, pd.Timestamp] | None = None

    @property
    def journal(self) -> PaperTradeJournal:
        """Paper trade journal."""
        return self._journal

    @property
    def balance(self) -> float:
        """Current account balance."""
        return self._ea.balance

    @property
    def total_trades(self) -> int:
        """Total completed trades."""
        return self._journal.count

    def run(self, data: pd.DataFrame) -> PaperTradeJournal:
        """Run paper trading session on dataset.

        Parameters
        ----------
        data : pd.DataFrame
            Normalized tick dataset.

        Returns
        -------
        PaperTradeJournal
            Complete trade journal.

        Raises
        ------
        ValueError
            If data is empty.
        """
        replay = MarketReplay(data)
        pip_size = self._ea.config.stop_loss_pips  # used for SL/TP calc

        for tick in replay:
            self._process_tick(tick)

        # Close any pending position at end
        if self._pending_entry is not None:
            self._close_pending(
                tick.mid if not data.empty else 0.0,
                tick.time if not data.empty else pd.Timestamp.now(),
                "end_of_data",
            )

        logger.info(
            "Paper trading complete: %d trades, PnL=%.2f",
            self._journal.count, self._journal.total_pnl,
        )
        return self._journal

    def _process_tick(self, tick: Tick) -> None:
        """Process a single tick during paper trading."""
        # Check if pending position should be closed
        if self._pending_entry is not None:
            direction, entry_price, lots, entry_time = self._pending_entry
            ticks_held = (tick.time - entry_time).total_seconds()

            # Simple SL/TP exit logic
            if direction == "BUY":
                pnl_pips = (tick.bid - entry_price) / 0.00001
            else:
                pnl_pips = (entry_price - tick.ask) / 0.00001

            sl = self._ea.config.stop_loss_pips
            tp = self._ea.config.take_profit_pips

            if pnl_pips <= -sl:
                self._close_pending(
                    tick.bid if direction == "BUY" else tick.ask,
                    tick.time, "stop_loss",
                )
            elif pnl_pips >= tp:
                self._close_pending(
                    tick.bid if direction == "BUY" else tick.ask,
                    tick.time, "take_profit",
                )
            return

        # Try to generate a new order
        order = self._ea.on_tick(tick)
        if order is not None:
            self._pending_entry = (
                order.direction, order.price,
                order.lots, tick.time,
            )

    def _close_pending(
        self, exit_price: float, exit_time: pd.Timestamp, reason: str
    ) -> None:
        """Close the pending position."""
        if self._pending_entry is None:
            return

        direction, entry_price, lots, entry_time = self._pending_entry

        if direction == "BUY":
            pnl = (exit_price - entry_price) * lots * 100000
        else:
            pnl = (entry_price - exit_price) * lots * 100000

        trade = PaperTrade(
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            lots=lots,
            pnl=pnl,
            entry_time=entry_time,
            exit_time=exit_time,
            exit_reason=reason,
        )
        self._journal.add_trade(trade)
        self._ea.close_position(pnl)
        self._pending_entry = None
