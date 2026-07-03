"""Wavelet decomposition and trend reconstruction.

Uses pywt for wavelet decomposition with periodization boundary mode.
Reconstructs the trend from approximation coefficients only.
"""

from __future__ import annotations

import numpy as np
import pywt


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
) -> np.ndarray:
    """Reconstruct the trend from approximation coefficients only.

    Zeroes out all detail coefficients and reconstructs.

    Parameters
    ----------
    coefficients : list[np.ndarray]
        Wavelet coefficients from decompose().
    wavelet_name : str
        Wavelet family name.
    signal_length : int
        Length of the original signal (to trim reconstruction).

    Returns
    -------
    np.ndarray
        Reconstructed trend signal.
    """
    trend_coeffs = [coefficients[0]] + [
        np.zeros_like(c) for c in coefficients[1:]
    ]
    trend = pywt.waverec(trend_coeffs, wavelet_name, mode="periodization")
    return trend[:signal_length]
