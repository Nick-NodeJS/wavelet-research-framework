"""Configuration for the Backtesting Engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ExitStrategy(str, Enum):
    """Supported exit strategies."""

    FIXED_TP_SL = "fixed_tp_sl"
    MAX_HOLD = "max_hold"
    OPPOSITE_SIGNAL = "opposite_signal"
    RETURN_TO_TREND = "return_to_trend"
    # Story 23: trend-relative exits
    DEVIATION_NORMALIZED = "deviation_normalized"
    TREND_INVALIDATION = "trend_invalidation"
    MAX_ADVERSE_MOVE = "max_adverse_move"
    COMBINED = "combined"


@dataclass(frozen=True)
class BacktestConfig:
    """Immutable backtesting configuration.

    Parameters
    ----------
    pip_size : float
        Size of one pip for the instrument.
    exit_strategy : ExitStrategy
        Primary exit strategy.
    max_hold_ticks : int
        Maximum holding period in ticks. Always enforced as a safety limit.
    take_profit_pips : float
        Take profit level in pips (used with FIXED_TP_SL).
    stop_loss_pips : float
        Stop loss level in pips (used with FIXED_TP_SL).
    commission_pips : float
        Round-trip commission in pips per trade.
    slippage_pips : float
        Slippage in pips per side (applied on entry and exit).

    Raises
    ------
    ValueError
        If any parameter is invalid.
    """

    pip_size: float = 0.00001
    exit_strategy: ExitStrategy = ExitStrategy.MAX_HOLD
    max_hold_ticks: int = 100
    take_profit_pips: float = 10.0
    stop_loss_pips: float = 10.0
    commission_pips: float = 0.0
    slippage_pips: float = 0.0
    # Story 23: trend-relative exit parameters
    exit_on_trend_touch: bool = True
    exit_deviation_threshold: float = 0.2
    max_adverse_normalized_deviation: float = 2.5

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.pip_size <= 0:
            raise ValueError(f"pip_size must be > 0, got {self.pip_size}")
        if self.max_hold_ticks < 1:
            raise ValueError(
                f"max_hold_ticks must be >= 1, got {self.max_hold_ticks}"
            )
        if self.take_profit_pips <= 0:
            raise ValueError(
                f"take_profit_pips must be > 0, got {self.take_profit_pips}"
            )
        if self.stop_loss_pips <= 0:
            raise ValueError(
                f"stop_loss_pips must be > 0, got {self.stop_loss_pips}"
            )
        if self.commission_pips < 0:
            raise ValueError(
                f"commission_pips must be >= 0, got {self.commission_pips}"
            )
        if self.slippage_pips < 0:
            raise ValueError(
                f"slippage_pips must be >= 0, got {self.slippage_pips}"
            )
