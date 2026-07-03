"""Domain models for the Causal Wavelet Engine."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Tick:
    """A single normalized tick.

    Parameters
    ----------
    time : pd.Timestamp
        Tick timestamp.
    bid : float
        Bid price.
    ask : float
        Ask price.
    mid : float
        Mid price: (bid + ask) / 2.
    spread : float
        Spread: ask - bid.
    """

    time: pd.Timestamp
    bid: float
    ask: float
    mid: float
    spread: float


@dataclass(frozen=True)
class WaveletPoint:
    """Output of the Causal Wavelet Engine for a single tick.

    All values are computed using only historical data up to and including
    the current tick. No future information is used.

    Parameters
    ----------
    trend : float
        Reconstructed trend value from approximation coefficients.
    deviation : float
        Difference between mid price and trend: mid - trend.
    z_score : float
        Deviation normalized by rolling volatility.
    slope : float
        Trend slope: current trend minus previous trend value.
    energy : float
        Normalized detail coefficient energy.
    noise : float
        Estimated noise level from the finest detail coefficients.
    """

    trend: float
    deviation: float
    z_score: float
    slope: float
    energy: float
    noise: float
