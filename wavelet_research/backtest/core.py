"""Core Backtesting Engine for sequential tick replay.

Replays historical ticks through the WaveletEngine and SignalEngine,
executing virtual trades and collecting results.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from wavelet_research.backtest.config import BacktestConfig
from wavelet_research.backtest.metrics import compute_report
from wavelet_research.backtest.models import BacktestReport, Trade
from wavelet_research.backtest.position import (
    OpenPosition,
    check_exit,
    close_position,
    compute_entry_price,
)
from wavelet_research.engine.core import WaveletEngine
from wavelet_research.engine.models import Tick, WaveletPoint
from wavelet_research.signal.core import SignalEngine
from wavelet_research.signal.models import Signal, SignalDecision

logger = logging.getLogger(__name__)


def _dataframe_to_ticks(data: pd.DataFrame) -> list[Tick]:
    """Convert a DataFrame to a list of Tick objects.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with time, bid, ask, mid, spread columns.

    Returns
    -------
    list[Tick]
        List of Tick objects.
    """
    ticks: list[Tick] = []
    for row in data.itertuples(index=False):
        ticks.append(
            Tick(
                time=row.time,  # type: ignore[attr-defined]
                bid=float(row.bid),  # type: ignore[attr-defined]
                ask=float(row.ask),  # type: ignore[attr-defined]
                mid=float(row.mid),  # type: ignore[attr-defined]
                spread=float(row.spread),  # type: ignore[attr-defined]
            )
        )
    return ticks


class BacktestEngine:
    """Deterministic backtesting engine.

    Replays historical ticks sequentially through the WaveletEngine
    and SignalEngine, managing virtual positions and collecting trades.

    Parameters
    ----------
    config : BacktestConfig
        Backtesting configuration.
    """

    def __init__(self, config: BacktestConfig) -> None:
        self._config = config

    @property
    def config(self) -> BacktestConfig:
        """Backtesting configuration."""
        return self._config

    def run(
        self,
        data: pd.DataFrame,
        wavelet_engine: WaveletEngine,
        signal_engine: SignalEngine,
    ) -> BacktestReport:
        """Execute a backtest on historical data.

        Processes ticks sequentially. No future data is used.

        Parameters
        ----------
        data : pd.DataFrame
            Normalized tick data with columns: time, bid, ask, mid, spread.
        wavelet_engine : WaveletEngine
            Causal wavelet engine for feature extraction.
        signal_engine : SignalEngine
            Signal engine for trade decisions.

        Returns
        -------
        BacktestReport
            Complete backtest report.

        Raises
        ------
        ValueError
            If data is empty.
        """
        if data.empty:
            raise ValueError("Dataset is empty.")

        ticks = _dataframe_to_ticks(data)
        trades: list[Trade] = []
        position: Optional[OpenPosition] = None

        for index, tick in enumerate(ticks):
            wavelet_point = wavelet_engine.update(tick)

            if wavelet_point is None:
                continue

            decision = signal_engine.decide(wavelet_point)

            # Manage open position
            if position is not None:
                position.update(tick, self._config.pip_size)

                exit_reason = check_exit(
                    position, tick, self._config,
                    wavelet_point, decision,
                )

                if exit_reason is not None:
                    trade = close_position(
                        position, tick, index, exit_reason, self._config
                    )
                    trades.append(trade)
                    position = None
                    logger.debug(
                        "Closed %s at index %d: %.2f pips (%s)",
                        trade.side, index, trade.pnl_pips, exit_reason,
                    )

            # Open new position if no position and signal is actionable
            elif decision.signal in (Signal.BUY, Signal.SELL):
                entry_price = compute_entry_price(
                    tick, decision.signal,
                    self._config.slippage_pips, self._config.pip_size,
                )
                position = OpenPosition(
                    side=decision.signal,
                    entry_index=index,
                    entry_price=entry_price,
                    entry_time=tick.time,
                    entry_trend=wavelet_point.trend,
                )
                logger.debug(
                    "Opened %s at index %d, price %.5f",
                    decision.signal.value, index, entry_price,
                )

        # Close any remaining position at last tick
        if position is not None and len(ticks) > 0:
            last_tick = ticks[-1]
            trade = close_position(
                position, last_tick, len(ticks) - 1,
                "end_of_data", self._config,
            )
            trades.append(trade)

        report = compute_report(trades)
        logger.info(
            "Backtest complete: %d trades, PF=%.2f, total=%.2f pips",
            report.trades, report.profit_factor, report.total_pnl,
        )
        return report
