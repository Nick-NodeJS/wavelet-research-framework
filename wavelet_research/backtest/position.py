"""Position lifecycle management for the Backtesting Engine.

Tracks open positions, computes running PnL, MAE, MFE,
and determines exit conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from wavelet_research.backtest.config import BacktestConfig, ExitStrategy
from wavelet_research.backtest.models import Trade
from wavelet_research.engine.models import Tick, WaveletPoint
from wavelet_research.signal.models import Signal, SignalDecision


@dataclass
class OpenPosition:
    """Mutable state for an open position.

    Parameters
    ----------
    side : Signal
        Trade direction (BUY or SELL).
    entry_index : int
        Tick index at entry.
    entry_price : float
        Execution price at entry (includes slippage).
    entry_time : pd.Timestamp
        Entry timestamp.
    entry_trend : float
        Wavelet trend at entry time.
    """

    side: Signal
    entry_index: int
    entry_price: float
    entry_time: pd.Timestamp
    entry_trend: float
    _max_favorable: float = 0.0
    _max_adverse: float = 0.0
    _ticks_held: int = 0

    @property
    def ticks_held(self) -> int:
        """Number of ticks position has been held."""
        return self._ticks_held

    def update(self, tick: Tick, pip_size: float) -> None:
        """Update position with a new tick.

        Parameters
        ----------
        tick : Tick
            Current tick.
        pip_size : float
            Pip size for the instrument.
        """
        self._ticks_held += 1
        running_pnl = self._compute_running_pnl(tick, pip_size)
        self._max_favorable = max(self._max_favorable, running_pnl)
        self._max_adverse = min(self._max_adverse, running_pnl)

    def compute_exit_pnl(self, tick: Tick, config: BacktestConfig) -> float:
        """Compute final PnL at exit including costs.

        Parameters
        ----------
        tick : Tick
            Exit tick.
        config : BacktestConfig
            Backtest configuration for costs.

        Returns
        -------
        float
            Net PnL in pips.
        """
        gross = self._compute_running_pnl(tick, config.pip_size)
        costs = config.commission_pips + 2 * config.slippage_pips
        return gross - costs

    def compute_gross_pnl(self, tick: Tick, pip_size: float) -> float:
        """Compute gross PnL at exit (no costs).

        Parameters
        ----------
        tick : Tick
            Exit tick.
        pip_size : float
            Pip size.

        Returns
        -------
        float
            Gross PnL in pips.
        """
        return self._compute_running_pnl(tick, pip_size)

    def get_exit_price(self, tick: Tick) -> float:
        """Get the execution price for closing the position.

        BUY positions close at bid. SELL positions close at ask.

        Parameters
        ----------
        tick : Tick
            Current tick.

        Returns
        -------
        float
            Exit execution price.
        """
        if self.side == Signal.BUY:
            return tick.bid
        return tick.ask

    @property
    def mae_pips(self) -> float:
        """Maximum Adverse Excursion in pips (positive value)."""
        return abs(self._max_adverse)

    @property
    def mfe_pips(self) -> float:
        """Maximum Favorable Excursion in pips."""
        return self._max_favorable

    def _compute_running_pnl(self, tick: Tick, pip_size: float) -> float:
        """Compute running PnL in pips.

        Parameters
        ----------
        tick : Tick
            Current tick.
        pip_size : float
            Pip size.

        Returns
        -------
        float
            Running PnL in pips.
        """
        exit_price = self.get_exit_price(tick)
        if self.side == Signal.BUY:
            return (exit_price - self.entry_price) / pip_size
        return (self.entry_price - exit_price) / pip_size


def compute_entry_price(tick: Tick, side: Signal, slippage_pips: float, pip_size: float) -> float:
    """Compute entry execution price including slippage.

    BUY at ask + slippage. SELL at bid - slippage.

    Parameters
    ----------
    tick : Tick
        Current tick.
    side : Signal
        Trade direction.
    slippage_pips : float
        Slippage in pips.
    pip_size : float
        Pip size.

    Returns
    -------
    float
        Entry execution price.
    """
    slippage = slippage_pips * pip_size
    if side == Signal.BUY:
        return tick.ask + slippage
    return tick.bid - slippage


def check_exit(
    position: OpenPosition,
    tick: Tick,
    config: BacktestConfig,
    wavelet_point: WaveletPoint | None,
    signal_decision: SignalDecision | None,
) -> str | None:
    """Check if exit conditions are met.

    Parameters
    ----------
    position : OpenPosition
        Current open position.
    tick : Tick
        Current tick.
    config : BacktestConfig
        Backtest configuration.
    wavelet_point : WaveletPoint or None
        Current wavelet features (for return-to-trend exit).
    signal_decision : SignalDecision or None
        Current signal decision (for opposite-signal exit).

    Returns
    -------
    str or None
        Exit reason if exit is triggered, None otherwise.
    """
    # Max hold always enforced as safety limit
    if position.ticks_held >= config.max_hold_ticks:
        return "max_hold"

    if config.exit_strategy == ExitStrategy.FIXED_TP_SL:
        running_pnl = position.compute_gross_pnl(tick, config.pip_size)
        if running_pnl >= config.take_profit_pips:
            return "take_profit"
        if running_pnl <= -config.stop_loss_pips:
            return "stop_loss"

    elif config.exit_strategy == ExitStrategy.OPPOSITE_SIGNAL:
        if signal_decision is not None:
            if (
                position.side == Signal.BUY
                and signal_decision.signal == Signal.SELL
            ):
                return "opposite_signal"
            if (
                position.side == Signal.SELL
                and signal_decision.signal == Signal.BUY
            ):
                return "opposite_signal"

    elif config.exit_strategy == ExitStrategy.RETURN_TO_TREND:
        if wavelet_point is not None:
            if position.side == Signal.BUY and tick.mid >= wavelet_point.trend:
                return "return_to_trend"
            if position.side == Signal.SELL and tick.mid <= wavelet_point.trend:
                return "return_to_trend"

    elif config.exit_strategy == ExitStrategy.DEVIATION_NORMALIZED:
        if wavelet_point is not None:
            dist = abs(tick.mid - wavelet_point.trend)
            norm = dist / max(abs(wavelet_point.z_score), 1e-9)
            if norm <= config.exit_deviation_threshold:
                return "deviation_normalized"

    elif config.exit_strategy == ExitStrategy.MAX_ADVERSE_MOVE:
        if wavelet_point is not None:
            if abs(wavelet_point.z_score) >= config.max_adverse_normalized_deviation:
                return "max_adverse_move"

    elif config.exit_strategy == ExitStrategy.TREND_INVALIDATION:
        if wavelet_point is not None:
            if position.side == Signal.BUY and wavelet_point.slope < 0:
                return "trend_invalidation"
            if position.side == Signal.SELL and wavelet_point.slope > 0:
                return "trend_invalidation"

    elif config.exit_strategy == ExitStrategy.COMBINED:
        if wavelet_point is not None:
            # Return to trend
            if config.exit_on_trend_touch:
                if position.side == Signal.BUY and tick.mid >= wavelet_point.trend:
                    return "return_to_trend"
                if position.side == Signal.SELL and tick.mid <= wavelet_point.trend:
                    return "return_to_trend"
            # Adverse move stop
            if abs(wavelet_point.z_score) >= config.max_adverse_normalized_deviation:
                return "max_adverse_move"

    return None


def close_position(
    position: OpenPosition,
    tick: Tick,
    exit_index: int,
    exit_reason: str,
    config: BacktestConfig,
) -> Trade:
    """Close a position and produce a Trade record.

    Parameters
    ----------
    position : OpenPosition
        Position to close.
    tick : Tick
        Exit tick.
    exit_index : int
        Tick index at exit.
    exit_reason : str
        Reason for exit.
    config : BacktestConfig
        Configuration for cost computation.

    Returns
    -------
    Trade
        Completed trade record.
    """
    exit_price = position.get_exit_price(tick)
    gross_pnl = position.compute_gross_pnl(tick, config.pip_size)
    net_pnl = position.compute_exit_pnl(tick, config)

    return Trade(
        entry_index=position.entry_index,
        exit_index=exit_index,
        side=position.side.value,
        entry_price=position.entry_price,
        exit_price=exit_price,
        pnl_pips=net_pnl,
        gross_pnl_pips=gross_pnl,
        mae_pips=position.mae_pips,
        mfe_pips=position.mfe_pips,
        holding_ticks=position.ticks_held,
        exit_reason=exit_reason,
        entry_time=position.entry_time,
        exit_time=tick.time,
    )
