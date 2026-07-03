"""Risk management for MT5 Expert Advisor."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskConfig:
    """Risk management configuration.

    Parameters
    ----------
    max_risk_per_trade : float
        Maximum risk per trade as fraction of balance (0.0, 1.0).
    max_open_positions : int
        Maximum concurrent open positions.
    max_daily_loss : float
        Maximum daily loss in account currency.
    lot_size : float
        Fixed lot size (used if position sizing is fixed).
    use_fixed_lot : bool
        Use fixed lot size instead of risk-based sizing.

    Raises
    ------
    ValueError
        If configuration is invalid.
    """

    max_risk_per_trade: float = 0.02
    max_open_positions: int = 1
    max_daily_loss: float = 100.0
    lot_size: float = 0.01
    use_fixed_lot: bool = True

    def __post_init__(self) -> None:
        if not (0.0 < self.max_risk_per_trade <= 1.0):
            raise ValueError(
                f"max_risk_per_trade must be in (0.0, 1.0], "
                f"got {self.max_risk_per_trade}"
            )
        if self.max_open_positions < 1:
            raise ValueError(
                f"max_open_positions must be >= 1, "
                f"got {self.max_open_positions}"
            )
        if self.max_daily_loss <= 0:
            raise ValueError(
                f"max_daily_loss must be > 0, got {self.max_daily_loss}"
            )
        if self.lot_size <= 0:
            raise ValueError(f"lot_size must be > 0, got {self.lot_size}")


def compute_position_size(
    balance: float,
    risk_per_trade: float,
    stop_loss_pips: float,
    pip_value: float,
) -> float:
    """Compute position size based on risk.

    Parameters
    ----------
    balance : float
        Account balance.
    risk_per_trade : float
        Risk fraction.
    stop_loss_pips : float
        Stop loss in pips.
    pip_value : float
        Value per pip per lot.

    Returns
    -------
    float
        Position size in lots.
    """
    if stop_loss_pips <= 0 or pip_value <= 0:
        return 0.0
    risk_amount = balance * risk_per_trade
    lots = risk_amount / (stop_loss_pips * pip_value)
    return max(0.01, round(lots, 2))
