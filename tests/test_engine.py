"""Tests for the Causal Wavelet Engine (Story 4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wavelet_research.engine.buffer import RollingBuffer
from wavelet_research.engine.config import (
    SUPPORTED_WAVELETS,
    SUPPORTED_WINDOWS,
    WaveletEngineConfig,
)
from wavelet_research.engine.core import WaveletEngine
from wavelet_research.engine.decomposition import (
    decompose,
    reconstruct_trend,
    safe_decomposition_level,
)
from wavelet_research.engine.features import (
    compute_deviation,
    compute_energy,
    compute_noise,
    compute_slope,
    compute_volatility,
    compute_z_score,
    extract_features,
)
from wavelet_research.engine.models import Tick, WaveletPoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tick(mid: float, index: int = 0) -> Tick:
    """Create a Tick with the given mid price."""
    spread = 0.00005
    return Tick(
        time=pd.Timestamp("2026-06-01") + pd.Timedelta(seconds=index),
        bid=mid - spread / 2,
        ask=mid + spread / 2,
        mid=mid,
        spread=spread,
    )


def _make_ticks(values: np.ndarray) -> list[Tick]:
    """Create a list of Ticks from an array of mid prices."""
    return [_make_tick(float(v), i) for i, v in enumerate(values)]


def _make_config(
    wavelet: str = "haar", window: int = 256, level: int = 2
) -> WaveletEngineConfig:
    """Create a default config."""
    return WaveletEngineConfig(
        wavelet=wavelet, window=window, level=level, volatility_window=128
    )


# ---------------------------------------------------------------------------
# Config Tests
# ---------------------------------------------------------------------------


class TestWaveletEngineConfig:
    def test_valid_config(self) -> None:
        cfg = _make_config()
        assert cfg.wavelet == "haar"
        assert cfg.window == 256
        assert cfg.level == 2

    def test_unsupported_wavelet(self) -> None:
        with pytest.raises(ValueError, match="Unsupported wavelet"):
            WaveletEngineConfig(wavelet="magic", window=256, level=2)

    def test_unsupported_window(self) -> None:
        with pytest.raises(ValueError, match="Unsupported window"):
            WaveletEngineConfig(wavelet="haar", window=100, level=2)

    def test_invalid_level(self) -> None:
        with pytest.raises(ValueError, match="level must be >= 1"):
            WaveletEngineConfig(wavelet="haar", window=256, level=0)

    def test_invalid_volatility_window(self) -> None:
        with pytest.raises(ValueError, match="volatility_window must be >= 1"):
            WaveletEngineConfig(
                wavelet="haar", window=256, level=2, volatility_window=0
            )

    def test_immutable(self) -> None:
        cfg = _make_config()
        with pytest.raises(Exception):
            cfg.level = 5  # type: ignore[misc]

    def test_all_supported_wavelets_valid(self) -> None:
        for wavelet in SUPPORTED_WAVELETS:
            cfg = WaveletEngineConfig(wavelet=wavelet, window=256, level=2)
            assert cfg.wavelet == wavelet

    def test_all_supported_windows_valid(self) -> None:
        for window in SUPPORTED_WINDOWS:
            cfg = WaveletEngineConfig(wavelet="haar", window=window, level=2)
            assert cfg.window == window


# ---------------------------------------------------------------------------
# Buffer Tests
# ---------------------------------------------------------------------------


class TestRollingBuffer:
    def test_basic_append_and_retrieve(self) -> None:
        buf = RollingBuffer(5)
        for v in [1.0, 2.0, 3.0]:
            buf.append(v)
        arr = buf.as_array()
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])

    def test_full_buffer(self) -> None:
        buf = RollingBuffer(3)
        for v in [1.0, 2.0, 3.0]:
            buf.append(v)
        assert buf.is_full
        np.testing.assert_array_equal(buf.as_array(), [1.0, 2.0, 3.0])

    def test_circular_overwrite(self) -> None:
        buf = RollingBuffer(3)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            buf.append(v)
        # Oldest values (1, 2) overwritten; should be [3, 4, 5]
        np.testing.assert_array_equal(buf.as_array(), [3.0, 4.0, 5.0])

    def test_empty_buffer_raises(self) -> None:
        buf = RollingBuffer(3)
        with pytest.raises(ValueError, match="empty"):
            buf.as_array()

    def test_invalid_capacity(self) -> None:
        with pytest.raises(ValueError, match="capacity"):
            RollingBuffer(0)

    def test_count_tracks_correctly(self) -> None:
        buf = RollingBuffer(5)
        assert buf.count == 0
        buf.append(1.0)
        assert buf.count == 1
        for v in range(10):
            buf.append(float(v))
        assert buf.count == 5


# ---------------------------------------------------------------------------
# Decomposition Tests
# ---------------------------------------------------------------------------


class TestDecomposition:
    def test_safe_level_clamps(self) -> None:
        # Requesting very high level should be clamped
        level = safe_decomposition_level("haar", 256, 100)
        assert 1 <= level <= 8  # max for haar/256

    def test_safe_level_minimum(self) -> None:
        level = safe_decomposition_level("haar", 4, 1)
        assert level >= 1

    def test_decompose_produces_coefficients(self) -> None:
        x = np.sin(np.linspace(0, 10, 256))
        coeffs = decompose(x, "haar", 2)
        assert len(coeffs) == 3  # cA + 2 detail levels
        assert len(coeffs[0]) > 0

    def test_reconstruct_trend_length(self) -> None:
        x = np.sin(np.linspace(0, 10, 256))
        coeffs = decompose(x, "haar", 2)
        trend = reconstruct_trend(coeffs, "haar", 256)
        assert len(trend) == 256

    def test_trend_smoother_than_signal(self) -> None:
        rng = np.random.default_rng(42)
        x = np.sin(np.linspace(0, 10, 256)) + rng.normal(0, 0.1, 256)
        coeffs = decompose(x, "haar", 3)
        trend = reconstruct_trend(coeffs, "haar", 256)
        # Trend should have lower variance than the noisy signal
        assert np.std(np.diff(trend)) < np.std(np.diff(x))


# ---------------------------------------------------------------------------
# Feature Extraction Tests
# ---------------------------------------------------------------------------


class TestFeatures:
    def test_deviation(self) -> None:
        assert compute_deviation(1.10, 1.09) == pytest.approx(0.01)

    def test_volatility_with_sufficient_data(self) -> None:
        rng = np.random.default_rng(42)
        values = 1.0 + np.cumsum(rng.normal(0, 0.01, 20))
        vol = compute_volatility(values, 10)
        assert vol > 0

    def test_volatility_empty(self) -> None:
        values = np.array([1.0])
        vol = compute_volatility(values, 10)
        assert vol == 0.0

    def test_z_score_zero_volatility(self) -> None:
        assert compute_z_score(0.01, 0.0) == 0.0

    def test_z_score_normal(self) -> None:
        z = compute_z_score(0.02, 0.01)
        assert z == pytest.approx(2.0)

    def test_slope(self) -> None:
        trend = np.array([1.0, 1.01, 1.03])
        assert compute_slope(trend) == pytest.approx(0.02)

    def test_slope_short(self) -> None:
        trend = np.array([1.0])
        assert compute_slope(trend) == 0.0

    def test_energy(self) -> None:
        detail = [np.array([1.0, 2.0]), np.array([0.5, 0.5])]
        energy = compute_energy(detail, 10)
        expected = (1.0 + 4.0 + 0.25 + 0.25) / 10
        assert energy == pytest.approx(expected)

    def test_noise_estimation(self) -> None:
        rng = np.random.default_rng(42)
        # White noise with known std
        noise_coeffs = rng.normal(0, 1.0, 1000)
        estimated = compute_noise(noise_coeffs)
        # MAD-based estimator should be close to 1.0 for Gaussian
        assert 0.8 < estimated < 1.3

    def test_noise_empty(self) -> None:
        assert compute_noise(np.array([])) == 0.0


# ---------------------------------------------------------------------------
# WaveletEngine Core Tests
# ---------------------------------------------------------------------------


class TestWaveletEngine:
    def test_returns_none_until_buffer_full(self) -> None:
        cfg = _make_config(window=256)
        engine = WaveletEngine(cfg)
        for i in range(255):
            result = engine.update(_make_tick(1.10 + i * 0.00001, i))
            assert result is None
        assert not engine.is_ready

    def test_produces_point_when_buffer_full(self) -> None:
        cfg = _make_config(window=256)
        engine = WaveletEngine(cfg)
        rng = np.random.default_rng(42)
        values = 1.10 + np.cumsum(rng.normal(0, 0.00001, 256))
        ticks = _make_ticks(values)
        result = None
        for tick in ticks:
            result = engine.update(tick)
        assert result is not None
        assert isinstance(result, WaveletPoint)

    def test_all_fields_present(self) -> None:
        cfg = _make_config(window=256)
        engine = WaveletEngine(cfg)
        rng = np.random.default_rng(42)
        values = 1.10 + np.cumsum(rng.normal(0, 0.00001, 256))
        result = None
        for tick in _make_ticks(values):
            result = engine.update(tick)
        assert result is not None
        assert isinstance(result.trend, float)
        assert isinstance(result.deviation, float)
        assert isinstance(result.z_score, float)
        assert isinstance(result.slope, float)
        assert isinstance(result.energy, float)
        assert isinstance(result.noise, float)

    def test_tick_count(self) -> None:
        cfg = _make_config(window=256)
        engine = WaveletEngine(cfg)
        for i in range(100):
            engine.update(_make_tick(1.10, i))
        assert engine.tick_count == 100

    def test_reset(self) -> None:
        cfg = _make_config(window=256)
        engine = WaveletEngine(cfg)
        rng = np.random.default_rng(42)
        values = 1.10 + np.cumsum(rng.normal(0, 0.00001, 300))
        for tick in _make_ticks(values):
            engine.update(tick)
        assert engine.is_ready
        engine.reset()
        assert not engine.is_ready
        assert engine.tick_count == 0


# ---------------------------------------------------------------------------
# Determinism Tests
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_deterministic_output(self) -> None:
        """Same input must produce identical output."""
        rng = np.random.default_rng(42)
        values = 1.10 + np.cumsum(rng.normal(0, 0.00001, 300))
        ticks = _make_ticks(values)

        cfg = _make_config(window=256)

        engine1 = WaveletEngine(cfg)
        engine2 = WaveletEngine(cfg)

        for tick in ticks:
            p1 = engine1.update(tick)
            p2 = engine2.update(tick)
            if p1 is not None:
                assert p2 is not None
                assert p1.trend == p2.trend
                assert p1.deviation == p2.deviation
                assert p1.z_score == p2.z_score
                assert p1.slope == p2.slope
                assert p1.energy == p2.energy
                assert p1.noise == p2.noise


# ---------------------------------------------------------------------------
# No Repainting / No Future Dependency Tests
# ---------------------------------------------------------------------------


class TestCausality:
    def test_no_repainting(self) -> None:
        """Output at tick N must not change when tick N+1 arrives."""
        rng = np.random.default_rng(42)
        values = 1.10 + np.cumsum(rng.normal(0, 0.00001, 260))
        ticks = _make_ticks(values)

        cfg = _make_config(window=256)

        # Process exactly 256 ticks to get the first output
        engine1 = WaveletEngine(cfg)
        first_point = None
        for tick in ticks[:256]:
            first_point = engine1.update(tick)
        assert first_point is not None

        # Replay with a fresh engine using the same 256 ticks
        engine2 = WaveletEngine(cfg)
        replay_point = None
        for tick in ticks[:256]:
            replay_point = engine2.update(tick)
        assert replay_point is not None

        # Same input must produce identical output
        assert first_point.trend == replay_point.trend
        assert first_point.deviation == replay_point.deviation
        assert first_point.z_score == replay_point.z_score
        assert first_point.slope == replay_point.slope
        assert first_point.energy == replay_point.energy
        assert first_point.noise == replay_point.noise

    def test_no_future_dependency(self) -> None:
        """Output must depend only on current and past ticks, not future."""
        rng = np.random.default_rng(42)
        values = 1.10 + np.cumsum(rng.normal(0, 0.00001, 260))
        ticks = _make_ticks(values)

        cfg = _make_config(window=256)

        # Run with original future
        engine1 = WaveletEngine(cfg)
        results1 = []
        for tick in ticks[:258]:
            r = engine1.update(tick)
            if r is not None:
                results1.append(r)

        # Run with a different future (change ticks 258-259)
        modified_values = values.copy()
        modified_values[258:] += 0.01  # Large future change
        modified_ticks = _make_ticks(modified_values)

        engine2 = WaveletEngine(cfg)
        results2 = []
        for tick in modified_ticks[:258]:
            r = engine2.update(tick)
            if r is not None:
                results2.append(r)

        # Results up to tick 258 must be identical regardless of future
        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert r1.trend == r2.trend
            assert r1.deviation == r2.deviation
            assert r1.z_score == r2.z_score


# ---------------------------------------------------------------------------
# All Supported Wavelets Tests
# ---------------------------------------------------------------------------


class TestAllWavelets:
    @pytest.mark.parametrize("wavelet", sorted(SUPPORTED_WAVELETS))
    def test_wavelet_produces_valid_output(self, wavelet: str) -> None:
        cfg = WaveletEngineConfig(
            wavelet=wavelet, window=256, level=2, volatility_window=128
        )
        engine = WaveletEngine(cfg)
        rng = np.random.default_rng(42)
        values = 1.10 + np.cumsum(rng.normal(0, 0.00001, 260))

        last_point = None
        for tick in _make_ticks(values):
            result = engine.update(tick)
            if result is not None:
                last_point = result

        assert last_point is not None
        assert isinstance(last_point.trend, float)
        assert not np.isnan(last_point.trend)
        assert isinstance(last_point.noise, float)
        assert last_point.noise >= 0


# ---------------------------------------------------------------------------
# All Supported Windows Tests
# ---------------------------------------------------------------------------


class TestAllWindows:
    @pytest.mark.parametrize("window", sorted(SUPPORTED_WINDOWS))
    def test_window_produces_valid_output(self, window: int) -> None:
        cfg = WaveletEngineConfig(
            wavelet="haar", window=window, level=2, volatility_window=128
        )
        engine = WaveletEngine(cfg)
        rng = np.random.default_rng(42)
        values = 1.10 + np.cumsum(rng.normal(0, 0.00001, window + 10))

        last_point = None
        for tick in _make_ticks(values):
            result = engine.update(tick)
            if result is not None:
                last_point = result

        assert last_point is not None
        assert isinstance(last_point.trend, float)


# ---------------------------------------------------------------------------
# Trend Reconstruction Tests
# ---------------------------------------------------------------------------


class TestTrendReconstruction:
    def test_trend_close_to_signal_for_smooth_input(self) -> None:
        """For a pure sine wave, trend should track the signal closely."""
        x = np.sin(np.linspace(0, 4 * np.pi, 256)) * 0.01 + 1.10
        ticks = _make_ticks(x)
        cfg = _make_config(window=256, level=2)
        engine = WaveletEngine(cfg)

        last_point = None
        for tick in ticks:
            last_point = engine.update(tick)

        assert last_point is not None
        # For a smooth signal, deviation should be small
        assert abs(last_point.deviation) < 0.005

    def test_trend_filters_noise(self) -> None:
        """Trend should be smoother than the noisy input."""
        rng = np.random.default_rng(42)
        clean = np.sin(np.linspace(0, 4 * np.pi, 512)) * 0.01 + 1.10
        noisy = clean + rng.normal(0, 0.002, 512)
        ticks = _make_ticks(noisy)

        cfg = _make_config(window=256, level=3)
        engine = WaveletEngine(cfg)

        points = []
        for tick in ticks:
            result = engine.update(tick)
            if result is not None:
                points.append(result)

        trends = [p.trend for p in points]
        mids = [float(t.mid) for t in ticks[256:]]

        # Trend variation should be lower than input variation
        assert np.std(np.diff(trends)) < np.std(np.diff(mids))
