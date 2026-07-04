"""Pure metric computations for trend quality audit."""

from __future__ import annotations

import numpy as np


def compute_repaint(
    first_seen: list[float],
    recheck: list[float],
) -> tuple[float, float]:
    """Compute repaint metrics by comparing first-seen vs later trend values.

    Parameters
    ----------
    first_seen : list[float]
        Trend values as first emitted at each index.
    recheck : list[float]
        Trend values at the same indices computed from a longer window.

    Returns
    -------
    tuple[float, float]
        (repaint_max, repaint_mean) — max and mean absolute differences.
    """
    if not first_seen or len(first_seen) != len(recheck):
        return 0.0, 0.0
    diffs = [abs(a - b) for a, b in zip(first_seen, recheck)]
    return float(max(diffs)), float(sum(diffs) / len(diffs))


def compute_smoothness(price_series: np.ndarray, trend_series: np.ndarray) -> float:
    """Compute smoothness as ratio of trend std-dev to price std-dev.

    Parameters
    ----------
    price_series : np.ndarray
        Raw price values.
    trend_series : np.ndarray
        Reconstructed trend values.

    Returns
    -------
    float
        Smoothness ratio. Lower is smoother. 0.0 if price std is zero.
    """
    price_std = float(np.std(price_series))
    if price_std == 0.0:
        return 0.0
    return float(np.std(trend_series)) / price_std


def compute_direction_stability(trend_series: np.ndarray) -> float:
    """Fraction of consecutive bars with consistent slope sign.

    Parameters
    ----------
    trend_series : np.ndarray
        Trend values.

    Returns
    -------
    float
        Stability ratio in [0.0, 1.0].
    """
    if len(trend_series) < 3:
        return 1.0
    slopes = np.diff(trend_series)
    signs = np.sign(slopes)
    consistent = int(np.sum(signs[1:] == signs[:-1]))
    return consistent / max(1, len(signs) - 1)


def compute_cross_frequency(
    price_series: np.ndarray, trend_series: np.ndarray
) -> float:
    """Fraction of bars where price crosses the trend.

    Parameters
    ----------
    price_series : np.ndarray
        Raw mid prices.
    trend_series : np.ndarray
        Trend values (same length).

    Returns
    -------
    float
        Cross frequency in [0.0, 1.0].
    """
    if len(price_series) < 2:
        return 0.0
    above = price_series > trend_series
    crosses = int(np.sum(above[1:] != above[:-1]))
    return crosses / max(1, len(price_series) - 1)


def compute_lag_estimate(
    price_series: np.ndarray, trend_series: np.ndarray, max_lag: int = 50
) -> int:
    """Estimate trend lag by finding the lag that maximises cross-correlation.

    Parameters
    ----------
    price_series : np.ndarray
        Raw prices.
    trend_series : np.ndarray
        Trend values.
    max_lag : int
        Maximum lag to test.

    Returns
    -------
    int
        Estimated lag in bars.
    """
    n = min(len(price_series), len(trend_series))
    if n < max_lag + 2:
        return 0
    p = price_series[:n] - np.mean(price_series[:n])
    t = trend_series[:n] - np.mean(trend_series[:n])
    best_lag, best_corr = 0, -1.0
    for lag in range(0, max_lag + 1):
        corr = float(np.corrcoef(p[lag:], t[: n - lag])[0, 1]) if n - lag > 1 else 0.0
        if corr > best_corr:
            best_corr, best_lag = corr, lag
    return best_lag


def compute_quality_score(
    repaint_max: float,
    smoothness: float,
    direction_stability: float,
    cross_frequency: float,
    repaint_threshold: float = 0.001,
    smoothness_threshold: float = 0.5,
) -> float:
    """Compute a composite quality score in [0.0, 1.0].

    Parameters
    ----------
    repaint_max : float
        Maximum repaint delta.
    smoothness : float
        Trend std / price std ratio.
    direction_stability : float
        Fraction of consistent slope bars.
    cross_frequency : float
        Fraction of price-cross bars.
    repaint_threshold : float
        Max acceptable repaint.
    smoothness_threshold : float
        Max acceptable smoothness ratio.

    Returns
    -------
    float
        Score in [0.0, 1.0].
    """
    repaint_score = max(0.0, 1.0 - repaint_max / max(repaint_threshold, 1e-9))
    smoothness_score = max(0.0, 1.0 - smoothness / max(smoothness_threshold, 1e-9))
    stability_score = direction_stability
    cross_score = 1.0 - min(1.0, cross_frequency * 2)
    return float(
        0.3 * repaint_score
        + 0.3 * smoothness_score
        + 0.2 * stability_score
        + 0.2 * cross_score
    )
