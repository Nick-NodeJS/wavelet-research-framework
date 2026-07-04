"""Production safety controls for the EA risk contract (Story 28).

Python-side risk validation. The MT5 EA must request approval before
executing any order. This module provides the pure validation logic;
the service exposes it via the /risk-check endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class RiskBlockReason(str, Enum):
    """Reason an order was blocked by risk controls."""

    SERVICE_OFFLINE = "SERVICE_OFFLINE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    FILTER_BLOCKED = "FILTER_BLOCKED"
    MAX_RISK_EXCEEDED = "MAX_RISK_EXCEEDED"
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    MAX_POSITIONS = "MAX_POSITIONS"
    SYMBOL_NOT_WHITELISTED = "SYMBOL_NOT_WHITELISTED"
    SPREAD_TOO_WIDE = "SPREAD_TOO_WIDE"
    SESSION_FILTER = "SESSION_FILTER"
    COOLDOWN_AFTER_LOSS = "COOLDOWN_AFTER_LOSS"
    COOLDOWN_AFTER_ERROR = "COOLDOWN_AFTER_ERROR"
    DUPLICATE_POSITION = "DUPLICATE_POSITION"
    LIVE_MODE_DISABLED = "LIVE_MODE_DISABLED"
    MANUAL_KILL_SWITCH = "MANUAL_KILL_SWITCH"


@dataclass(frozen=True)
class RiskConfig:
    """Immutable EA risk configuration.

    Parameters
    ----------
    live_mode : bool
        Must be explicitly True to allow real execution.
    kill_switch : bool
        If True, all orders are blocked immediately.
    max_risk_per_trade_pct : float
        Maximum risk as % of balance per trade.
    max_daily_loss : float
        Maximum allowed daily loss (account currency).
    max_open_positions : int
        Maximum simultaneous open positions.
    symbol_whitelist : tuple[str, ...]
        Allowed symbols. Empty means all.
    max_spread : float
        Maximum allowed spread (price units).
    min_signal_confidence : float
        Minimum signal confidence to execute.
    cooldown_bars_after_loss : int
        Bars to wait after a losing trade.
    cooldown_bars_after_error : int
        Bars to wait after a service error.
    """

    live_mode: bool = False
    kill_switch: bool = False
    max_risk_per_trade_pct: float = 1.0
    max_daily_loss: float = 200.0
    max_open_positions: int = 1
    symbol_whitelist: tuple[str, ...] = ()
    max_spread: float = 0.0003
    min_signal_confidence: float = 0.6
    cooldown_bars_after_loss: int = 5
    cooldown_bars_after_error: int = 3

    def __post_init__(self) -> None:
        if not (0.0 < self.max_risk_per_trade_pct <= 10.0):
            raise ValueError(f"max_risk_per_trade_pct must be in (0, 10], got {self.max_risk_per_trade_pct}")
        if self.max_daily_loss <= 0:
            raise ValueError(f"max_daily_loss must be > 0, got {self.max_daily_loss}")
        if self.max_open_positions < 1:
            raise ValueError(f"max_open_positions must be >= 1, got {self.max_open_positions}")
        if not (0.0 <= self.min_signal_confidence <= 1.0):
            raise ValueError(f"min_signal_confidence must be in [0, 1], got {self.min_signal_confidence}")


@dataclass(frozen=True)
class RiskCheckRequest:
    """Input for the risk check (from MT5 EA).

    Parameters
    ----------
    symbol : str
        Symbol to trade.
    signal_confidence : float
        Confidence from signal engine.
    filter_can_trade : bool
        Filter engine decision.
    current_spread : float
        Current market spread.
    open_positions : int
        Number of currently open positions.
    daily_loss_so_far : float
        Realized loss today.
    bars_since_last_loss : int
        Bars since last losing trade.
    bars_since_last_error : int
        Bars since last service error.
    has_duplicate_position : bool
        Whether a position in this direction already exists.
    """

    symbol: str
    signal_confidence: float
    filter_can_trade: bool
    current_spread: float
    open_positions: int
    daily_loss_so_far: float
    bars_since_last_loss: int
    bars_since_last_error: int
    has_duplicate_position: bool


@dataclass(frozen=True)
class RiskCheckResult:
    """Result of a risk check evaluation.

    Parameters
    ----------
    can_execute : bool
        True when all safety controls pass.
    reasons : tuple[RiskBlockReason, ...]
        All triggered block reasons.
    """

    can_execute: bool
    reasons: tuple[RiskBlockReason, ...]

    def to_dict(self) -> dict[str, object]:
        """Serialise to JSON-compatible dict."""
        return {
            "can_execute": self.can_execute,
            "reasons": [r.value for r in self.reasons],
        }


def evaluate_risk(
    request: RiskCheckRequest,
    config: RiskConfig,
) -> RiskCheckResult:
    """Evaluate all safety controls for a pending order.

    Parameters
    ----------
    request : RiskCheckRequest
        Current order context from the EA.
    config : RiskConfig
        Risk configuration.

    Returns
    -------
    RiskCheckResult
        Approval or block with all triggered reasons.
    """
    reasons: list[RiskBlockReason] = []

    if config.kill_switch:
        reasons.append(RiskBlockReason.MANUAL_KILL_SWITCH)

    if not config.live_mode:
        reasons.append(RiskBlockReason.LIVE_MODE_DISABLED)

    if not request.filter_can_trade:
        reasons.append(RiskBlockReason.FILTER_BLOCKED)

    if request.signal_confidence < config.min_signal_confidence:
        reasons.append(RiskBlockReason.LOW_CONFIDENCE)

    if (
        config.symbol_whitelist
        and request.symbol not in config.symbol_whitelist
    ):
        reasons.append(RiskBlockReason.SYMBOL_NOT_WHITELISTED)

    if request.current_spread > config.max_spread:
        reasons.append(RiskBlockReason.SPREAD_TOO_WIDE)

    if request.open_positions >= config.max_open_positions:
        reasons.append(RiskBlockReason.MAX_POSITIONS)

    if request.daily_loss_so_far >= config.max_daily_loss:
        reasons.append(RiskBlockReason.DAILY_LOSS_LIMIT)

    if request.bars_since_last_loss < config.cooldown_bars_after_loss:
        reasons.append(RiskBlockReason.COOLDOWN_AFTER_LOSS)

    if request.bars_since_last_error < config.cooldown_bars_after_error:
        reasons.append(RiskBlockReason.COOLDOWN_AFTER_ERROR)

    if request.has_duplicate_position:
        reasons.append(RiskBlockReason.DUPLICATE_POSITION)

    return RiskCheckResult(
        can_execute=len(reasons) == 0,
        reasons=tuple(reasons),
    )
