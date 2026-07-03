"""Threshold-based signal rules and confidence computation.

Pure functions implementing the core decision logic.
"""

from __future__ import annotations

from wavelet_research.engine.models import WaveletPoint
from wavelet_research.signal.config import SignalConfig
from wavelet_research.signal.models import Signal


def check_buy_threshold(point: WaveletPoint, config: SignalConfig) -> bool:
    """Check if z-score crosses the BUY threshold.

    Parameters
    ----------
    point : WaveletPoint
        Current wavelet features.
    config : SignalConfig
        Signal configuration.

    Returns
    -------
    bool
        True if z-score <= -buy_z_threshold.
    """
    return point.z_score <= -config.buy_z_threshold


def check_sell_threshold(point: WaveletPoint, config: SignalConfig) -> bool:
    """Check if z-score crosses the SELL threshold.

    Parameters
    ----------
    point : WaveletPoint
        Current wavelet features.
    config : SignalConfig
        Signal configuration.

    Returns
    -------
    bool
        True if z-score >= sell_z_threshold.
    """
    return point.z_score >= config.sell_z_threshold


def check_slope_filter(
    point: WaveletPoint, direction: Signal
) -> bool:
    """Check if slope aligns with the signal direction.

    BUY: slope should be non-negative (trend recovering or flat).
    SELL: slope should be non-positive (trend weakening or flat).

    Parameters
    ----------
    point : WaveletPoint
        Current wavelet features.
    direction : Signal
        Proposed signal direction.

    Returns
    -------
    bool
        True if slope confirms the direction.
    """
    if direction == Signal.BUY:
        return point.slope >= 0
    if direction == Signal.SELL:
        return point.slope <= 0
    return True


def check_energy_filter(point: WaveletPoint, config: SignalConfig) -> bool:
    """Check if energy exceeds the minimum threshold.

    Parameters
    ----------
    point : WaveletPoint
        Current wavelet features.
    config : SignalConfig
        Signal configuration.

    Returns
    -------
    bool
        True if energy >= min_energy.
    """
    return point.energy >= config.min_energy


def check_noise_filter(point: WaveletPoint, config: SignalConfig) -> bool:
    """Check if noise is within acceptable bounds.

    Parameters
    ----------
    point : WaveletPoint
        Current wavelet features.
    config : SignalConfig
        Signal configuration.

    Returns
    -------
    bool
        True if noise <= max_noise (signal is acceptable).
    """
    return point.noise <= config.max_noise


def compute_confidence(
    point: WaveletPoint,
    config: SignalConfig,
    direction: Signal,
) -> float:
    """Compute a normalized confidence score in [0.0, 1.0].

    Based on:
    - z-score strength (how far past the threshold)
    - slope alignment bonus
    - energy confirmation bonus
    - noise penalty

    Parameters
    ----------
    point : WaveletPoint
        Current wavelet features.
    config : SignalConfig
        Signal configuration.
    direction : Signal
        The proposed signal direction.

    Returns
    -------
    float
        Confidence score in [0.0, 1.0].
    """
    abs_z = abs(point.z_score)
    threshold = (
        config.buy_z_threshold if direction == Signal.BUY
        else config.sell_z_threshold
    )

    # Z-score component: how far past threshold, capped at 2x threshold
    z_strength = min((abs_z - threshold) / max(threshold, 1e-10), 1.0)
    z_strength = max(z_strength, 0.0)

    # Slope alignment: bonus if slope matches direction
    slope_aligned = check_slope_filter(point, direction)
    slope_bonus = 0.15 if slope_aligned else 0.0

    # Energy component: normalized, capped contribution
    energy_bonus = min(point.energy * 100, 0.15)

    # Noise penalty: higher noise reduces confidence
    noise_penalty = min(point.noise * 10, 0.3) if point.noise > 0 else 0.0

    raw = z_strength * 0.7 + slope_bonus + energy_bonus - noise_penalty
    return max(0.0, min(1.0, raw))
