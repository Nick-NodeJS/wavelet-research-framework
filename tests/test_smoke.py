import numpy as np
from wavelet_research.wavelet_engine import WaveletConfig, causal_wavelet_point


def test_wavelet_point_smoke():
    x = np.sin(np.linspace(0, 20, 512)) + np.random.default_rng(1).normal(0, 0.01, 512)
    cfg = WaveletConfig(wavelet="haar", window=256, level=2, threshold=2.5)
    p = causal_wavelet_point(x, cfg)
    assert isinstance(p.trend, float)
    assert isinstance(p.z_score, float)
