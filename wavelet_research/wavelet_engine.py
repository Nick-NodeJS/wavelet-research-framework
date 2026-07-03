from dataclasses import dataclass
import numpy as np
import pywt


@dataclass(frozen=True)
class WaveletConfig:
    wavelet: str
    window: int
    level: int
    threshold: float
    vol_window: int = 256


@dataclass(frozen=True)
class WaveletPoint:
    trend: float
    deviation: float
    z_score: float
    slope: float
    energy: float


def _safe_level(wavelet_name: str, window: int, requested_level: int) -> int:
    wavelet = pywt.Wavelet(wavelet_name)
    max_level = pywt.dwt_max_level(window, wavelet.dec_len)
    return max(1, min(requested_level, max_level))


def causal_wavelet_point(values: np.ndarray, cfg: WaveletConfig) -> WaveletPoint:
    """
    Calculates trend/deviation for the latest point only.

    Causal decision rule:
    - values must contain only historical data up to current tick.
    - output is only for values[-1].
    """
    if len(values) < cfg.window:
        raise ValueError("Not enough values for configured window")

    x = np.asarray(values[-cfg.window:], dtype=float)

    level = _safe_level(cfg.wavelet, cfg.window, cfg.level)
    coeffs = pywt.wavedec(x, cfg.wavelet, mode="periodization", level=level)

    # Trend = reconstruction from approximation coefficients only.
    trend_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    trend_series = pywt.waverec(trend_coeffs, cfg.wavelet, mode="periodization")[: len(x)]

    trend = float(trend_series[-1])
    prev_trend = float(trend_series[-2]) if len(trend_series) > 1 else trend
    price = float(x[-1])

    deviation = price - trend

    returns = np.diff(x)
    if len(returns) >= cfg.vol_window:
        vol_sample = returns[-cfg.vol_window:]
    else:
        vol_sample = returns

    vol = float(np.std(vol_sample))
    z_score = float(deviation / vol) if vol > 0 else 0.0

    detail_energy = 0.0
    for c in coeffs[1:]:
        detail_energy += float(np.sum(np.square(c)))
    energy = detail_energy / max(1, len(x))

    return WaveletPoint(
        trend=trend,
        deviation=float(deviation),
        z_score=z_score,
        slope=float(trend - prev_trend),
        energy=float(energy),
    )
