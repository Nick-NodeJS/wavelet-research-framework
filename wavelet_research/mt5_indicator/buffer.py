"""Indicator buffer — holds the latest response data for MT5 rendering.

A thin in-memory store. Never calculates. Only stores and retrieves.
"""

from __future__ import annotations

from wavelet_research.mt5_indicator.models import (
    ConnectionStatus,
    ParsedWaveletResponse,
)


class IndicatorBuffer:
    """Stores the latest wavelet response arrays for MT5 indicator rendering.

    MT5 indicator buffers are indexed arrays written on every recalculation.
    This class mirrors that pattern in Python for testing and for the
    Python-side logic that populates the buffer before handing values
    to MQL5 via the service bridge.

    Parameters
    ----------
    size : int
        Number of bars the buffer covers. Must be >= 1.

    Raises
    ------
    ValueError
        If size < 1.
    """

    def __init__(self, size: int) -> None:
        if size < 1:
            raise ValueError(f"size must be >= 1, got {size}")
        self._size = size
        self._trend: list[float] = [0.0] * size
        self._relative_deviation: list[float] = [0.0] * size
        self._z_score: list[float] = [0.0] * size
        self._energy: list[float] = [0.0] * size
        self._noise: list[float] = [0.0] * size
        self._status: ConnectionStatus = ConnectionStatus.CONNECTING

    @property
    def size(self) -> int:
        """Buffer capacity.

        Returns
        -------
        int
            Number of slots.
        """
        return self._size

    @property
    def status(self) -> ConnectionStatus:
        """Current connection status.

        Returns
        -------
        ConnectionStatus
            Latest status.
        """
        return self._status

    @property
    def trend(self) -> list[float]:
        """Current trend buffer (copy).

        Returns
        -------
        list[float]
            Trend values.
        """
        return list(self._trend)

    @property
    def relative_deviation(self) -> list[float]:
        """Current relative deviation buffer (copy).

        Returns
        -------
        list[float]
            Relative deviation values.
        """
        return list(self._relative_deviation)

    @property
    def z_score(self) -> list[float]:
        """Current z-score buffer (copy).

        Returns
        -------
        list[float]
            Z-score values.
        """
        return list(self._z_score)

    @property
    def energy(self) -> list[float]:
        """Current energy buffer (copy).

        Returns
        -------
        list[float]
            Energy values.
        """
        return list(self._energy)

    @property
    def noise(self) -> list[float]:
        """Current noise buffer (copy).

        Returns
        -------
        list[float]
            Noise values.
        """
        return list(self._noise)

    def update(self, response: ParsedWaveletResponse) -> None:
        """Overwrite buffer contents with the latest service response.

        The response is aligned to the buffer tail (most recent values
        at the highest index, matching MT5 bar indexing where index 0
        is the current bar).

        Parameters
        ----------
        response : ParsedWaveletResponse
            Validated response from the Wavelet Service.
        """
        n = min(response.length, self._size)
        offset = response.length - n

        self._trend = list(response.trend[offset:offset + n])
        self._relative_deviation = list(response.relative_deviation[offset:offset + n])
        self._z_score = list(response.z_score[offset:offset + n])
        self._energy = list(response.energy[offset:offset + n])
        self._noise = list(response.noise[offset:offset + n])

        if n < self._size:
            pad = self._size - n
            self._trend = [0.0] * pad + self._trend
            self._relative_deviation = [0.0] * pad + self._relative_deviation
            self._z_score = [0.0] * pad + self._z_score
            self._energy = [0.0] * pad + self._energy
            self._noise = [0.0] * pad + self._noise

        self._status = ConnectionStatus.CONNECTED

    def set_status(self, status: ConnectionStatus) -> None:
        """Set the connection status without changing buffer data.

        Parameters
        ----------
        status : ConnectionStatus
            New connection status.
        """
        self._status = status

    def clear(self) -> None:
        """Reset all buffers to zero and status to CONNECTING."""
        self._trend = [0.0] * self._size
        self._relative_deviation = [0.0] * self._size
        self._z_score = [0.0] * self._size
        self._energy = [0.0] * self._size
        self._noise = [0.0] * self._size
        self._status = ConnectionStatus.CONNECTING
