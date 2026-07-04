"""Domain models for historical deviation statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DeviationEvent:
    """A single historical snapshot for building deviation statistics.

    Parameters
    ----------
    timestamp : str
        ISO timestamp of the snapshot.
    symbol : str
        Trading symbol.
    window : int
        Engine window size.
    trend_value : float
        Trend at snapshot time.
    price : float
        Mid price at snapshot time.
    normalized_deviation : float
        z-score at snapshot time.
    trend_slope : float
        Trend slope at snapshot time.
    volatility_bucket : str
        "low" | "normal" | "high" — bucketed for lookup.
    future_return_1 : float
        Price return 1 bar forward (offline label).
    future_return_3 : float
        Price return 3 bars forward.
    future_return_5 : float
        Price return 5 bars forward.
    future_return_10 : float
        Price return 10 bars forward.
    future_return_20 : float
        Price return 20 bars forward.
    returned_to_trend : bool
        Whether price returned to within near-threshold.
    bars_to_return : int
        How many bars until return (0 if never within window).
    max_favorable_excursion : float
        Maximum favorable price move (in z-score units) after entry.
    max_adverse_excursion : float
        Maximum adverse price move (in z-score units) after entry.
    """

    timestamp: str
    symbol: str
    window: int
    trend_value: float
    price: float
    normalized_deviation: float
    trend_slope: float
    volatility_bucket: str
    future_return_1: float
    future_return_3: float
    future_return_5: float
    future_return_10: float
    future_return_20: float
    returned_to_trend: bool
    bars_to_return: int
    max_favorable_excursion: float
    max_adverse_excursion: float


@dataclass(frozen=True)
class DeviationQueryResult:
    """Result of a historical stats query.

    Parameters
    ----------
    sample_size : int
        Number of matching historical events.
    return_to_trend_probability : float
        Fraction of matching events where price returned to trend.
    median_bars_to_return : float
        Median bars until return.
    expected_return : float
        Mean of future_return_5 across matching events.
    expected_adverse_excursion : float
        Mean max_adverse_excursion across matching events.
    confidence_level : str
        "high" | "medium" | "low" | "insufficient".
    """

    sample_size: int
    return_to_trend_probability: float
    median_bars_to_return: float
    expected_return: float
    expected_adverse_excursion: float
    confidence_level: str
