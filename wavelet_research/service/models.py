"""Request and response models for the Wavelet Service."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TickRequest:
    """A single tick in a service request.

    Parameters
    ----------
    time : str
        ISO timestamp string.
    bid : float
        Bid price.
    ask : float
        Ask price.
    mid : float
        Mid price. If omitted from JSON, computed as (bid+ask)/2.
    """

    time: str
    bid: float
    ask: float
    mid: float


@dataclass(frozen=True)
class WaveletRequest:
    """Incoming request body for the /wavelet endpoint.

    Parameters
    ----------
    ticks : tuple[TickRequest, ...]
        Sequence of normalized ticks.
    trend_mode : str
        Wavelet approximation level for trend reconstruction.
        One of "A1", "A2", "A3", "A4". Defaults to "A2".
    window : int | None
        Rolling window size override. If None, service default is used.
    wavelet : str | None
        Wavelet family override. If None, service default is used.
    level : int | None
        Decomposition level override. If None, service default is used.
    """

    ticks: tuple[TickRequest, ...]
    trend_mode: str = "A2"
    window: int | None = None
    wavelet: str | None = None
    level: int | None = None


@dataclass(frozen=True)
class WaveletResponse:
    """Response body for the /wavelet endpoint.

    All arrays have the same length as the number of ticks in the request
    (positions without enough history carry 0.0).

    Parameters
    ----------
    trend : tuple[float, ...]
        Reconstructed trend values.
    relative_deviation : tuple[float, ...]
        (mid - trend) / local_volatility.
    z_score : tuple[float, ...]
        Deviation normalized by rolling volatility.
    energy : tuple[float, ...]
        Normalized detail coefficient energy.
    noise : tuple[float, ...]
        Estimated noise level.
    """

    trend: tuple[float, ...]
    relative_deviation: tuple[float, ...]
    z_score: tuple[float, ...]
    energy: tuple[float, ...]
    noise: tuple[float, ...]

    def to_dict(self) -> dict[str, list[float]]:
        """Serialise to a JSON-compatible dictionary.

        Returns
        -------
        dict[str, list[float]]
            All arrays as plain lists.
        """
        return {
            "trend": list(self.trend),
            "relative_deviation": list(self.relative_deviation),
            "z_score": list(self.z_score),
            "energy": list(self.energy),
            "noise": list(self.noise),
        }


@dataclass(frozen=True)
class HealthResponse:
    """Response body for the /health endpoint.

    Parameters
    ----------
    status : str
        Always "ok" when healthy.
    wavelet : str
        Wavelet family in use.
    version : str
        Service version string.
    """

    status: str
    wavelet: str
    version: str

    def to_dict(self) -> dict[str, str]:
        """Serialise to a JSON-compatible dictionary.

        Returns
        -------
        dict[str, str]
            Plain key-value pairs.
        """
        return {
            "status": self.status,
            "wavelet": self.wavelet,
            "version": self.version,
        }
