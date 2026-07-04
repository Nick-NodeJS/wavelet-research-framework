"""Configuration for the MT5 Thin Indicator client layer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IndicatorConfig:
    """Configuration mirroring MT5 indicator input parameters.

    All values are intentionally public to match MT5's ``input`` variables.

    Parameters
    ----------
    server_url : str
        Base URL of the Python Wavelet Service (e.g. ``http://127.0.0.1:5000``).
    tick_window : int
        Number of latest ticks to send per request. Default: 2048.
    request_timeout_seconds : float
        HTTP request timeout in seconds. Default: 0.5.
    auto_refresh : bool
        Whether to request new data on every MT5 recalculation. Default: True.
    draw_trend : bool
        Draw trend buffer on the main chart. Default: True.
    draw_relative_deviation : bool
        Draw relative deviation in the separate window. Default: True.
    draw_z_score : bool
        Draw z-score in the separate window. Default: True.
    draw_energy : bool
        Draw energy in the separate window. Default: True.

    Raises
    ------
    ValueError
        If any parameter value is out of range.
    """

    server_url: str = "http://127.0.0.1:5000"
    tick_window: int = 2048
    request_timeout_seconds: float = 0.5
    auto_refresh: bool = True
    draw_trend: bool = True
    draw_relative_deviation: bool = True
    draw_z_score: bool = True
    draw_energy: bool = True

    def __post_init__(self) -> None:
        if self.tick_window < 1:
            raise ValueError(f"tick_window must be >= 1, got {self.tick_window}")
        if self.request_timeout_seconds <= 0:
            raise ValueError(
                f"request_timeout_seconds must be > 0, got {self.request_timeout_seconds}"
            )
        if not self.server_url.startswith("http"):
            raise ValueError(
                f"server_url must start with 'http', got {self.server_url!r}"
            )

    @property
    def wavelet_endpoint(self) -> str:
        """Full URL for the /wavelet endpoint.

        Returns
        -------
        str
            Endpoint URL.
        """
        return self.server_url.rstrip("/") + "/wavelet"

    @property
    def health_endpoint(self) -> str:
        """Full URL for the /health endpoint.

        Returns
        -------
        str
            Endpoint URL.
        """
        return self.server_url.rstrip("/") + "/health"
