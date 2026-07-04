"""Wavelet decomposition and trend reconstruction.

Uses pywt for wavelet decomposition with periodization boundary mode.
Reconstructs the trend from approximation coefficients only.

Supported trend modes A1–A4 select which approximation level is used
for reconstruction.  A higher number = smoother / slower trend.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import pywt


class TrendMode(str, Enum):
    """Selectable wavelet approximation level for trend reconstruction.

    Each mode reconstructs the trend using only the approximation
    coefficients at the indicated level.  Detail coefficients are
    zeroed out.  A higher number produces a smoother, slower-moving
    trend.

    Values
    ------
    A1 : finest approximation — fastest trend
    A2 : default — matches original implementation
    A3 : medium smoothing
    A4 : coarsest approximation — slowest trend
    """

    A1 = "A1"
    A2 = "A2"
    A3 = "A3"
    A4 = "A4"


SUPPORTED_TREND_MODES: frozenset[str] = frozenset(m.value for m in TrendMode)

DEFAULT_TREND_MODE: TrendMode = TrendMode.A2


def safe_decomposition_level(
    wavelet_name: str, window_size: int, requested_level: int
) -> int:
    """Compute a safe decomposition level that does not exceed the maximum.

    Parameters
    ----------
    wavelet_name : str
        Name of the wavelet family.
    window_size : int
        Size of the input signal.
    requested_level : int
        Desired decomposition level.

    Returns
    -------
    int
        Clamped decomposition level (at least 1).
    """
    wavelet = pywt.Wavelet(wavelet_name)
    max_level = pywt.dwt_max_level(window_size, wavelet.dec_len)
    return max(1, min(requested_level, max_level))


def decompose(
    values: np.ndarray, wavelet_name: str, level: int
) -> list[np.ndarray]:
    """Perform wavelet decomposition.

    Parameters
    ----------
    values : np.ndarray
        Input signal.
    wavelet_name : str
        Wavelet family name.
    level : int
        Decomposition level.

    Returns
    -------
    list[np.ndarray]
        Wavelet coefficients [cA, cD_n, ..., cD_1].
    """
    return pywt.wavedec(values, wavelet_name, mode="periodization", level=level)


def reconstruct_trend(
    coefficients: list[np.ndarray],
    wavelet_name: str,
    signal_length: int,
    approximation_level: int | None = None,
) -> np.ndarray:
    """Reconstruct the trend from approximation coefficients only.

    Zeroes out all detail coefficients at and below ``approximation_level``
    and reconstructs the signal.  If ``approximation_level`` is None or
    equals the decomposition depth, the full approximation (all details
    zeroed) is used — identical to the original behaviour.

    When ``approximation_level`` is less than the decomposition depth,
    the signal is re-decomposed to that shallower level so that the
    returned trend reflects only that approximation.

    Parameters
    ----------
    coefficients : list[np.ndarray]
        Wavelet coefficients from decompose() — [cA_N, cD_N, ..., cD_1].
    wavelet_name : str
        Wavelet family name.
    signal_length : int
        Length of the original signal (to trim reconstruction).
    approximation_level : int | None
        Target approximation level (1 = finest, N = coarsest).  Must be
        between 1 and len(coefficients)-1.  Defaults to the full depth.

    Returns
    -------
    np.ndarray
        Reconstructed trend signal of length ``signal_length``.
    """
    full_depth = len(coefficients) - 1  # number of detail levels in this decomposition

    if approximation_level is None or approximation_level >= full_depth:
        # Default / full-depth behaviour: keep cA_N, zero all details.
        # Identical to the original implementation.
        trend_coeffs = [coefficients[0]] + [
            np.zeros_like(c) for c in coefficients[1:]
        ]
        trend = pywt.waverec(trend_coeffs, wavelet_name, mode="periodization")
        return trend[:signal_length]

    # Shallower approximation: coefficients already contain cA_N and all detail
    # levels.  For Ax where x < N we need to reconstruct from cA_N upwards
    # through the detail levels that are above level x, keeping those details
    # as-is, and zeroing the details at levels x down to 1.
    #
    # pywt.wavedec returns [cA_N, cD_N, cD_{N-1}, ..., cD_1]
    # so index 0 = cA_N, index 1 = cD_N, ..., index k = cD_{N-k+1}
    # detail at level L is at index (full_depth - L + 1)
    #
    # We want to keep cA_N + details for levels > approximation_level,
    # then zero details for levels <= approximation_level.
    clamped = max(1, min(approximation_level, full_depth))
    target_depth = full_depth - clamped  # number of detail levels to keep

    # Build coefficient list: keep cA and details above the target, zero rest
    reconstructed_coeffs: list[np.ndarray] = [coefficients[0]]
    for idx, detail in enumerate(coefficients[1:], start=1):
        if idx <= target_depth:
            reconstructed_coeffs.append(detail)  # keep coarser details
        else:
            reconstructed_coeffs.append(np.zeros_like(detail))  # zero finer

    trend = pywt.waverec(reconstructed_coeffs, wavelet_name, mode="periodization")
    return trend[:signal_length]
