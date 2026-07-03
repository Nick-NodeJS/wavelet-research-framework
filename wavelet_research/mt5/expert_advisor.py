"""MT5 Expert Advisor consuming the Signal Engine.

Provides execution logic for paper/live trading with configurable
risk management. Delegates signal generation to existing engines.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

from wavelet_research.engine.config import WaveletEngineConfig
from wavelet_research.engine.core import WaveletEngine
from wavelet_research.engine.models import Tick
from wavelet_research.mt5.risk import RiskConfig, compute_position_size
from wavelet_research.signal.config import SignalConfig
from wavelet_research.signal.core import SignalEngine
from wavelet_research.signal.models import Signal

logger = logging.getLogger(__name__)


class ExecutionMode(str, Enum):
    """EA execution mode."""

    PAPER = "paper"
    LIVE = "live"


@dataclass(frozen=True)
class EAConfig:
    """Expert Advisor configuration.

    Parameters
    ----------
    wavelet_config : WaveletEngineConfig
        Wavelet engine config.
    signal_config : SignalConfig
        Signal engine config.
    risk_config : RiskConfig
        Risk management config.
    mode : ExecutionMode
        Execution mode.
    stop_loss_pips : float
        Stop loss in pips.
    take_profit_pips : float
        Take profit in pips.
    pip_value : float
        Value per pip per lot.

    Raises
    ------
    ValueError
        If stop_loss or take_profit invalid.
    """

    wavelet_config: WaveletEngineConfig
    signal_config: SignalConfig
    risk_config: RiskConfig = field(default_factory=RiskConfig)
    mode: ExecutionMode = ExecutionMode.PAPER
    stop_loss_pips: float = 20.0
    take_profit_pips: float = 30.0
    pip_value: float = 10.0

    def __post_init__(self) -> None:
        if self.stop_loss_pips <= 0:
            raise ValueError(
                f"stop_loss_pips must be > 0, got {self.stop_loss_pips}"
            )
        if self.take_profit_pips <= 0:
            raise ValueError(
                f"take_profit_pips must be > 0, got {self.take_profit_pips}"
            )


@dataclass
class OrderRequest:
    """Order request to be sent to broker.

    Parameters
    ----------
    direction : str
        'BUY' or 'SELL'.
    lots : float
        Position size.
    stop_loss_pips : float
        Stop loss distance.
    take_profit_pips : float
        Take profit distance.
    price : float
        Entry price.
    confidence : float
        Signal confidence.
    """

    direction: str
    lots: float
    stop_loss_pips: float
    take_profit_pips: float
    price: float
    confidence: float


class MT5ExpertAdvisor:
    """Expert Advisor consuming WaveletEngine + SignalEngine.

    Processes ticks, generates signals, applies risk management,
    and produces order requests.

    Parameters
    ----------
    config : EAConfig
        EA configuration.
    initial_balance : float
        Starting account balance.
    """

    def __init__(self, config: EAConfig, initial_balance: float = 10000.0) -> None:
        self._config = config
        self._wavelet_engine = WaveletEngine(config.wavelet_config)
        self._signal_engine = SignalEngine(config.signal_config)
        self._balance = initial_balance
        self._open_positions: int = 0
        self._daily_loss: float = 0.0
        self._orders: list[OrderRequest] = []

    @property
    def config(self) -> EAConfig:
        """EA configuration."""
        return self._config

    @property
    def balance(self) -> float:
        """Current balance."""
        return self._balance

    @property
    def open_positions(self) -> int:
        """Number of open positions."""
        return self._open_positions

    @property
    def orders(self) -> list[OrderRequest]:
        """All generated order requests."""
        return list(self._orders)

    @property
    def daily_loss(self) -> float:
        """Current daily loss."""
        return self._daily_loss

    def on_tick(self, tick: Tick) -> OrderRequest | None:
        """Process a tick and potentially generate an order.

        Parameters
        ----------
        tick : Tick
            Incoming tick.

        Returns
        -------
        OrderRequest | None
            Order request if signal warrants, else None.
        """
        wavelet_point = self._wavelet_engine.update(tick)
        if wavelet_point is None:
            return None

        decision = self._signal_engine.decide(wavelet_point)

        if decision.signal == Signal.HOLD:
            return None

        if not self._check_risk_limits():
            logger.debug("Risk limits exceeded, skipping signal.")
            return None

        direction = "BUY" if decision.signal == Signal.BUY else "SELL"
        lots = self._compute_lots()
        price = tick.ask if direction == "BUY" else tick.bid

        order = OrderRequest(
            direction=direction,
            lots=lots,
            stop_loss_pips=self._config.stop_loss_pips,
            take_profit_pips=self._config.take_profit_pips,
            price=price,
            confidence=decision.confidence,
        )

        self._orders.append(order)
        self._open_positions += 1
        logger.info(
            "Order: %s %.2f lots @ %.5f (confidence=%.2f)",
            direction, lots, price, decision.confidence,
        )
        return order

    def close_position(self, pnl: float) -> None:
        """Record a position closure.

        Parameters
        ----------
        pnl : float
            Realized PnL for the position.
        """
        self._open_positions = max(0, self._open_positions - 1)
        self._balance += pnl
        if pnl < 0:
            self._daily_loss += abs(pnl)

    def reset_daily(self) -> None:
        """Reset daily loss counter (new trading day)."""
        self._daily_loss = 0.0

    def _check_risk_limits(self) -> bool:
        """Check if current risk limits allow a new trade."""
        if self._open_positions >= self._config.risk_config.max_open_positions:
            return False
        if self._daily_loss >= self._config.risk_config.max_daily_loss:
            return False
        return True

    def _compute_lots(self) -> float:
        """Compute position size."""
        if self._config.risk_config.use_fixed_lot:
            return self._config.risk_config.lot_size
        return compute_position_size(
            self._balance,
            self._config.risk_config.max_risk_per_trade,
            self._config.stop_loss_pips,
            self._config.pip_value,
        )
