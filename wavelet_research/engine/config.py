"""Configuration for the Causal Wavelet Engine."""

from __future__ import annotations

from dataclasses import dataclass

from wavelet_research.engine.decomposition import (
    DEFAULT_TREND_MODE,
    SUPPORTED_TREND_MODES,
    TrendMode,
)

SUPPORTED_WAVELETS: frozenset[str] = frozenset({
    "haar",
    "db2", "db4", "db6",
    "sym4", "sym6",
    "coif1", "coif3",
})

SUPPORTED_WINDOWS: frozenset[int] = frozenset({
    128, 256, 512, 1024, 2048, 4096,
})

SUPPORTED_LEVELS: frozenset[int] = frozenset({1, 2, 3, 4})


@dataclass(frozen=True)
class WaveletEngineConfig:
    """Immutable configuration for the Causal Wavelet Engine.

    Parameters
    ----------
    wavelet : str
        Wavelet family name. Must be one of SUPPORTED_WAVELETS.
    window : int
        Rolling window size in ticks. Must be one of SUPPORTED_WINDOWS.
    level : int
        Decomposition level. Must be in SUPPORTED_LEVELS (1–4).
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
    trend_mode: TrendMode = DEFAULT_TREND_MODE

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
        if self.trend_mode not in TrendMode.__members__.values():
            raise ValueError(
                f"Unsupported trend_mode: {self.trend_mode!r}. "
                f"Supported: {sorted(SUPPORTED_TREND_MODES)}"
            )
