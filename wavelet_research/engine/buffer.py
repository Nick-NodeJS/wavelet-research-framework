"""Efficient rolling buffer for streaming tick processing."""

from __future__ import annotations

import numpy as np


class RollingBuffer:
    """Fixed-size circular buffer for streaming float values.

    Provides O(1) append and O(1) numpy view construction.

    Parameters
    ----------
    capacity : int
        Maximum number of values to store.

    Raises
    ------
    ValueError
        If capacity < 1.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self._capacity = capacity
        self._data = np.empty(capacity, dtype=np.float64)
        self._count = 0
        self._head = 0

    @property
    def capacity(self) -> int:
        """Buffer capacity."""
        return self._capacity

    @property
    def count(self) -> int:
        """Number of values currently stored."""
        return self._count

    @property
    def is_full(self) -> bool:
        """Whether the buffer has reached capacity."""
        return self._count >= self._capacity

    def append(self, value: float) -> None:
        """Append a value to the buffer.

        If the buffer is full, the oldest value is overwritten.

        Parameters
        ----------
        value : float
            Value to append.
        """
        self._data[self._head] = value
        self._head = (self._head + 1) % self._capacity
        if self._count < self._capacity:
            self._count += 1

    def as_array(self) -> np.ndarray:
        """Return buffer contents as a contiguous numpy array in insertion order.

        Returns
        -------
        np.ndarray
            Array of stored values, oldest first.

        Raises
        ------
        ValueError
            If buffer is empty.
        """
        if self._count == 0:
            raise ValueError("Buffer is empty.")

        if self._count < self._capacity:
            return self._data[: self._count].copy()

        # Circular: head points to the oldest element
        return np.concatenate([
            self._data[self._head :],
            self._data[: self._head],
        ])
