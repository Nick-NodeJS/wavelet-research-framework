"""Domain models for the MT5 Thin Indicator client layer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ConnectionStatus(Enum):
    """Connection state displayed on the MT5 indicator.

    Maps directly to the status string shown in MT5's chart comment.
    """

    CONNECTING = "Connecting..."
    CONNECTED = "Connected"
    TIMEOUT = "Timeout"
    INVALID_RESPONSE = "Invalid Response"
    SERVICE_OFFLINE = "Service Offline"


@dataclass(frozen=True)
class TickPayload:
    """A single tick formatted for the JSON request body.

    Parameters
    ----------
    time : str
        ISO 8601 timestamp string.
    bid : float
        Bid price.
    ask : float
        Ask price.
    mid : float
        Mid price: (bid + ask) / 2.
    """

    time: str
    bid: float
    ask: float
    mid: float

    def to_dict(self) -> dict[str, object]:
        """Serialise to a JSON-compatible dictionary.

        Returns
        -------
        dict[str, object]
            Plain key-value pairs.
        """
        return {
            "time": self.time,
            "bid": self.bid,
            "ask": self.ask,
            "mid": self.mid,
        }


@dataclass(frozen=True)
class ParsedWaveletResponse:
    """Parsed and validated response from the /wavelet endpoint.

    All arrays have the same length.

    Parameters
    ----------
    trend : tuple[float, ...]
        Reconstructed wavelet trend values.
    relative_deviation : tuple[float, ...]
        Relative deviation (mid - trend) / volatility.
    z_score : tuple[float, ...]
        Z-score values.
    energy : tuple[float, ...]
        Wavelet energy values.
    noise : tuple[float, ...]
        Estimated noise level.
    """

    trend: tuple[float, ...]
    relative_deviation: tuple[float, ...]
    z_score: tuple[float, ...]
    energy: tuple[float, ...]
    noise: tuple[float, ...]

    def __post_init__(self) -> None:
        lengths = {
            len(self.trend),
            len(self.relative_deviation),
            len(self.z_score),
            len(self.energy),
            len(self.noise),
        }
        if len(lengths) != 1:
            raise ValueError(
                f"All response arrays must have equal length, got: {lengths}"
            )

    @property
    def length(self) -> int:
        """Number of data points in the response.

        Returns
        -------
        int
            Length of all arrays.
        """
        return len(self.trend)


@dataclass(frozen=True)
class IndicatorResponse:
    """Result of a single indicator update cycle.

    Combines the parsed data with the current connection status.

    Parameters
    ----------
    status : ConnectionStatus
        Connection status after the request.
    data : ParsedWaveletResponse | None
        Parsed response data, or None if the request failed.
    elapsed_ms : float
        Time taken for the HTTP round-trip in milliseconds.
    """

    status: ConnectionStatus
    data: ParsedWaveletResponse | None
    elapsed_ms: float

    @property
    def is_successful(self) -> bool:
        """Whether the update produced valid data.

        Returns
        -------
        bool
            True if data is available.
        """
        return self.data is not None and self.status == ConnectionStatus.CONNECTED
