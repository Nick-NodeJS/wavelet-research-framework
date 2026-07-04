"""Normalization helpers for deviation calculations. Pure functions."""

from __future__ import annotations

_NEAR_THRESHOLD = 0.3


def relative_distance(raw_distance: float, trend: float) -> float:
    """Compute relative distance: raw / trend.

    Parameters
    ----------
    raw_distance : float
        price - trend.
    trend : float
        Current trend value.

    Returns
    -------
    float
        Relative distance, or 0.0 if trend is zero.
    """
    if trend == 0.0:
        return 0.0
    return raw_distance / trend


def volatility_normalized_distance(raw_distance: float, volatility: float) -> float:
    """Normalize by rolling volatility (same as z-score).

    Parameters
    ----------
    raw_distance : float
        price - trend.
    volatility : float
        Rolling volatility estimate.

    Returns
    -------
    float
        Normalized distance, or 0.0 if volatility is zero.
    """
    if volatility <= 0.0:
        return 0.0
    return raw_distance / volatility


def classify_side(z_score: float, near_threshold: float = _NEAR_THRESHOLD) -> str:
    """Classify price side relative to trend.

    Parameters
    ----------
    z_score : float
        Normalized deviation.
    near_threshold : float
        Absolute z-score below which side is NEAR.

    Returns
    -------
    str
        "above", "below", or "near".
    """
    if abs(z_score) < near_threshold:
        return "near"
    return "above" if z_score > 0 else "below"
