"""Market replay for paper trading simulation."""

from __future__ import annotations

import logging
from typing import Iterator

import pandas as pd

from wavelet_research.engine.models import Tick

logger = logging.getLogger(__name__)


class MarketReplay:
    """Replays historical tick data as a stream.

    Provides an iterator interface for sequential tick replay,
    simulating real-time market data.

    Parameters
    ----------
    data : pd.DataFrame
        Normalized tick dataset.
    speed_multiplier : float
        Replay speed multiplier (for display, not used for logic).

    Raises
    ------
    ValueError
        If data is empty.
    """

    def __init__(
        self, data: pd.DataFrame, speed_multiplier: float = 1.0
    ) -> None:
        if data.empty:
            raise ValueError("Data is empty.")
        self._data = data
        self._speed_multiplier = speed_multiplier
        self._index: int = 0

    @property
    def total_ticks(self) -> int:
        """Total number of ticks in replay."""
        return len(self._data)

    @property
    def current_index(self) -> int:
        """Current replay position."""
        return self._index

    @property
    def is_complete(self) -> bool:
        """Whether replay is complete."""
        return self._index >= len(self._data)

    @property
    def progress(self) -> float:
        """Replay progress as fraction [0.0, 1.0]."""
        if len(self._data) == 0:
            return 1.0
        return self._index / len(self._data)

    def next_tick(self) -> Tick | None:
        """Get the next tick in the replay.

        Returns
        -------
        Tick | None
            Next tick, or None if replay is complete.
        """
        if self._index >= len(self._data):
            return None

        row = self._data.iloc[self._index]
        tick = Tick(
            time=pd.Timestamp(row["time"]) if "time" in row.index else pd.Timestamp.now(),
            bid=float(row["bid"]),
            ask=float(row["ask"]),
            mid=float(row["mid"]),
            spread=float(row["spread"]),
        )
        self._index += 1
        return tick

    def __iter__(self) -> Iterator[Tick]:
        """Iterate over all ticks."""
        while not self.is_complete:
            tick = self.next_tick()
            if tick is not None:
                yield tick

    def reset(self) -> None:
        """Reset replay to beginning."""
        self._index = 0
