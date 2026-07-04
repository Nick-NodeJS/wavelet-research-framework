"""Service configuration.

Loaded from environment variables or explicit constructor args.
No hardcoded constants.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from wavelet_research.engine.config import SUPPORTED_WAVELETS, WaveletEngineConfig

_VERSION = "0.2.0"
_DEFAULT_WAVELET = "db4"


@dataclass(frozen=True)
class ServiceConfig:
    """Configuration for the Wavelet Service.

    Parameters
    ----------
    wavelet : str
        Wavelet family. Default 'db4'. Must be one of SUPPORTED_WAVELETS.
    window : int
        Rolling window size in ticks.
    level : int
        Decomposition level.
    volatility_window : int
        Window for rolling volatility.
    host : str
        Service bind host.
    port : int
        Service bind port.
    debug : bool
        Debug mode.
    version : str
        Service version string.

    Raises
    ------
    ValueError
        If wavelet is not in SUPPORTED_WAVELETS or port is out of range.
    """

    wavelet: str = _DEFAULT_WAVELET
    window: int = 512
    level: int = 2
    volatility_window: int = 256
    host: str = "127.0.0.1"
    port: int = 5000
    debug: bool = False
    version: str = _VERSION

    def __post_init__(self) -> None:
        if self.wavelet not in SUPPORTED_WAVELETS:
            raise ValueError(
                f"Unsupported wavelet: {self.wavelet!r}. "
                f"Supported: {sorted(SUPPORTED_WAVELETS)}"
            )
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"port must be 1-65535, got {self.port}")

    def to_engine_config(self) -> WaveletEngineConfig:
        """Build the WaveletEngineConfig for this service.

        Returns
        -------
        WaveletEngineConfig
            Engine configuration derived from service config.
        """
        return WaveletEngineConfig(
            wavelet=self.wavelet,
            window=self.window,
            level=self.level,
            volatility_window=self.volatility_window,
        )

    @classmethod
    def from_env(cls) -> "ServiceConfig":
        """Load configuration from environment variables.

        Environment variables (all optional):
          WAVELET_FAMILY          str, default 'db4'
          WAVELET_WINDOW          int, default 512
          WAVELET_LEVEL           int, default 2
          WAVELET_VOL_WINDOW      int, default 256
          SERVICE_HOST            str, default 127.0.0.1
          SERVICE_PORT            int, default 5000
          SERVICE_DEBUG           bool (1/0), default 0

        Returns
        -------
        ServiceConfig
            Populated service configuration.
        """
        return cls(
            wavelet=os.environ.get("WAVELET_FAMILY", _DEFAULT_WAVELET),
            window=int(os.environ.get("WAVELET_WINDOW", "512")),
            level=int(os.environ.get("WAVELET_LEVEL", "2")),
            volatility_window=int(os.environ.get("WAVELET_VOL_WINDOW", "256")),
            host=os.environ.get("SERVICE_HOST", "127.0.0.1"),
            port=int(os.environ.get("SERVICE_PORT", "5000")),
            debug=os.environ.get("SERVICE_DEBUG", "0") == "1",
        )
