"""Feature extraction from wavelet decomposition results.

Computes trend, deviation, z-score, slope, energy, and noise
from wavelet coefficients and the raw signal.
"""

from __future__ import annotations

import numpy as np

from wavelet_research.engine.models import WaveletPoint


def compute_deviation(price: float, trend: float) -> float:
    """Compute deviation between price and trend.

    Parameters
    ----------
    price : float
        Current mid price.
    trend : float
        Current trend value.

    Returns
    -------
    float
        Deviation: price - trend.
    """
    return price - trend


def compute_volatility(values: np.ndarray, volatility_window: int) -> float:
    """Compute rolling volatility from price returns.

    Parameters
    ----------
    values : np.ndarray
        Price values in the rolling window.
    volatility_window : int
        Number of returns to use for volatility estimation.

    Returns
    -------
    float
        Standard deviation of returns. Returns 0.0 if insufficient data.
    """
    returns = np.diff(values)
    if len(returns) == 0:
        return 0.0
    if len(returns) >= volatility_window:
        sample = returns[-volatility_window:]
    else:
        sample = returns
    return float(np.std(sample))


def compute_z_score(deviation: float, volatility: float) -> float:
    """Compute z-score: deviation normalized by volatility.

    Parameters
    ----------
    deviation : float
        Price deviation from trend.
    volatility : float
        Rolling volatility.

    Returns
    -------
    float
        Z-score, or 0.0 if volatility is zero.
    """
    if volatility <= 0:
        return 0.0
    return deviation / volatility


def compute_slope(trend_series: np.ndarray) -> float:
    """Compute trend slope from the last two trend values.

    Parameters
    ----------
    trend_series : np.ndarray
        Reconstructed trend series.

    Returns
    -------
    float
        Slope: trend[-1] - trend[-2], or 0.0 if too short.
    """
    if len(trend_series) < 2:
        return 0.0
    return float(trend_series[-1] - trend_series[-2])


def compute_energy(detail_coefficients: list[np.ndarray], signal_length: int) -> float:
    """Compute normalized detail coefficient energy.

    Parameters
    ----------
    detail_coefficients : list[np.ndarray]
        Detail coefficients from decomposition (excluding approximation).
    signal_length : int
        Length of the original signal for normalization.

    Returns
    -------
    float
        Normalized energy.
    """
    total_energy = 0.0
    for coeff in detail_coefficients:
        total_energy += float(np.sum(np.square(coeff)))
    return total_energy / max(1, signal_length)


def compute_noise(finest_detail_coefficients: np.ndarray) -> float:
    """Estimate noise level from the finest detail coefficients.

    Uses the median absolute deviation (MAD) estimator for robust
    noise estimation, following Donoho & Johnstone (1994).

    Parameters
    ----------
    finest_detail_coefficients : np.ndarray
        Level-1 detail coefficients (the finest scale).

    Returns
    -------
    float
        Estimated noise standard deviation.
    """
    if len(finest_detail_coefficients) == 0:
        return 0.0
    mad = float(np.median(np.abs(finest_detail_coefficients)))
    return mad / 0.6745


def extract_features(
    values: np.ndarray,
    trend_series: np.ndarray,
    coefficients: list[np.ndarray],
    volatility_window: int,
) -> WaveletPoint:
    """Extract all wavelet features for the current tick.

    Parameters
    ----------
    values : np.ndarray
        Price values in the rolling window.
    trend_series : np.ndarray
        Reconstructed trend series.
    coefficients : list[np.ndarray]
        Full wavelet coefficients [cA, cD_n, ..., cD_1].
    volatility_window : int
        Window size for volatility estimation.

    Returns
    -------
    WaveletPoint
        Computed features for the latest tick.
    """
    price = float(values[-1])
    trend = float(trend_series[-1])

    deviation = compute_deviation(price, trend)
    volatility = compute_volatility(values, volatility_window)
    z_score = compute_z_score(deviation, volatility)
    slope = compute_slope(trend_series)

    detail_coefficients = coefficients[1:]
    energy = compute_energy(detail_coefficients, len(values))

    # Finest detail coefficients are the last in the list
    finest_detail = coefficients[-1] if len(coefficients) > 1 else np.array([])
    noise = compute_noise(finest_detail)

    return WaveletPoint(
        trend=trend,
        deviation=deviation,
        z_score=z_score,
        slope=slope,
        energy=energy,
        noise=noise,
    )
