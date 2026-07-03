"""Domain models for tick data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TickFrame:
    """Normalized tick data container.

    Provides access to time, bid, ask, mid, spread, and flags columns.

    Parameters
    ----------
    frame : pd.DataFrame
        Normalized dataframe with columns: time, bid, ask, mid, spread, flags.
    """

    frame: pd.DataFrame

    @property
    def time(self) -> pd.Series:
        """Tick timestamps with millisecond precision."""
        return self.frame["time"]

    @property
    def bid(self) -> pd.Series:
        """Bid prices."""
        return self.frame["bid"]

    @property
    def ask(self) -> pd.Series:
        """Ask prices."""
        return self.frame["ask"]

    @property
    def mid(self) -> pd.Series:
        """Mid prices: (bid + ask) / 2."""
        return self.frame["mid"]

    @property
    def spread(self) -> pd.Series:
        """Spread: ask - bid."""
        return self.frame["spread"]

    @property
    def flags(self) -> pd.Series:
        """MT5 tick flags."""
        return self.frame["flags"]

    def __len__(self) -> int:
        return len(self.frame)

    def __repr__(self) -> str:
        return f"TickFrame(ticks={len(self)}, start={self.time.iloc[0]}, end={self.time.iloc[-1]})"
