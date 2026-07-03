"""Experiment configuration for the research framework.

Configurations are immutable dataclasses representing a full set of
parameters for a single research experiment.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable experiment configuration.

    Parameters
    ----------
    wavelet : str
        Wavelet family name (e.g., 'haar', 'db4').
    level : int
        Decomposition level.
    window : int
        Lookback window size in ticks.
    volatility_window : int
        Window size for volatility estimation.
    normalization : str
        Normalization method (e.g., 'zscore', 'minmax').
    signal_threshold : float
        Threshold for signal generation.
    max_hold : int
        Maximum holding period in ticks.
    pip_size : float
        Size of one pip for the instrument.
    """

    wavelet: str
    level: int
    window: int
    volatility_window: int
    normalization: str
    signal_threshold: float
    max_hold: int = 100
    pip_size: float = 0.00001

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.level < 1:
            raise ValueError(f"level must be >= 1, got {self.level}")
        if self.window < 2:
            raise ValueError(f"window must be >= 2, got {self.window}")
        if self.volatility_window < 1:
            raise ValueError(
                f"volatility_window must be >= 1, got {self.volatility_window}"
            )
        if self.signal_threshold <= 0:
            raise ValueError(
                f"signal_threshold must be > 0, got {self.signal_threshold}"
            )
        if self.max_hold < 1:
            raise ValueError(f"max_hold must be >= 1, got {self.max_hold}")
        if self.pip_size <= 0:
            raise ValueError(f"pip_size must be > 0, got {self.pip_size}")

    @property
    def identifier(self) -> str:
        """Unique string identifier for this configuration."""
        return (
            f"{self.wavelet}_L{self.level}_W{self.window}"
            f"_V{self.volatility_window}_N{self.normalization}"
            f"_T{self.signal_threshold}_H{self.max_hold}"
        )
