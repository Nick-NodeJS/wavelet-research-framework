"""Configuration for the Causal Wavelet Engine."""

from __future__ import annotations

from dataclasses import dataclass

SUPPORTED_WAVELETS: frozenset[str] = frozenset({
    "haar", "db4", "db6", "sym4", "coif3",
})

SUPPORTED_WINDOWS: frozenset[int] = frozenset({
    256, 512, 1024, 2048, 4096,
})


@dataclass(frozen=True)
class WaveletEngineConfig:
    """Immutable configuration for the Causal Wavelet Engine.

    Parameters
    ----------
    wavelet : str
        Wavelet family name. Must be one of the supported wavelets.
    window : int
        Rolling window size in ticks. Must be one of the supported windows.
    level : int
        Decomposition level. Must be >= 1.
    volatility_window : int
        Window size for rolling volatility estimation. Must be >= 1.

    Raises
    ------
    ValueError
        If any parameter is invalid or unsupported.
    """

    wavelet: str
    window: int
    level: int
    volatility_window: int = 256

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.wavelet not in SUPPORTED_WAVELETS:
            raise ValueError(
                f"Unsupported wavelet: {self.wavelet!r}. "
                f"Supported: {sorted(SUPPORTED_WAVELETS)}"
            )
        if self.window not in SUPPORTED_WINDOWS:
            raise ValueError(
                f"Unsupported window: {self.window}. "
                f"Supported: {sorted(SUPPORTED_WINDOWS)}"
            )
        if self.level < 1:
            raise ValueError(f"level must be >= 1, got {self.level}")
        if self.volatility_window < 1:
            raise ValueError(
                f"volatility_window must be >= 1, got {self.volatility_window}"
            )
