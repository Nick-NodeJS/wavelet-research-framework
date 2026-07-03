"""Configuration for the Signal Engine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SignalConfig:
    """Immutable configuration for signal generation.

    Parameters
    ----------
    buy_z_threshold : float
        Absolute z-score threshold for BUY signals. Must be > 0.
    sell_z_threshold : float
        Absolute z-score threshold for SELL signals. Must be > 0.
    min_confidence : float
        Minimum confidence to emit a non-HOLD signal. In [0.0, 1.0].
    slope_filter_enabled : bool
        If True, require slope alignment for signals.
    energy_filter_enabled : bool
        If True, require minimum energy for signals.
    noise_filter_enabled : bool
        If True, reject signals when noise exceeds max_noise.
    max_noise : float
        Maximum noise level to allow a signal. Must be > 0.
    min_energy : float
        Minimum energy level to allow a signal. Must be >= 0.
    allow_buy : bool
        If False, BUY signals are suppressed.
    allow_sell : bool
        If False, SELL signals are suppressed.

    Raises
    ------
    ValueError
        If any parameter is invalid.
    """

    buy_z_threshold: float = 2.0
    sell_z_threshold: float = 2.0
    min_confidence: float = 0.0
    slope_filter_enabled: bool = True
    energy_filter_enabled: bool = False
    noise_filter_enabled: bool = False
    max_noise: float = 1.0
    min_energy: float = 0.0
    allow_buy: bool = True
    allow_sell: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.buy_z_threshold <= 0:
            raise ValueError(
                f"buy_z_threshold must be > 0, got {self.buy_z_threshold}"
            )
        if self.sell_z_threshold <= 0:
            raise ValueError(
                f"sell_z_threshold must be > 0, got {self.sell_z_threshold}"
            )
        if not (0.0 <= self.min_confidence <= 1.0):
            raise ValueError(
                f"min_confidence must be in [0.0, 1.0], got {self.min_confidence}"
            )
        if self.max_noise <= 0:
            raise ValueError(
                f"max_noise must be > 0, got {self.max_noise}"
            )
        if self.min_energy < 0:
            raise ValueError(
                f"min_energy must be >= 0, got {self.min_energy}"
            )
