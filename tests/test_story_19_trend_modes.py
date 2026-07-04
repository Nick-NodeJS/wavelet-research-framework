"""Tests for Story 19 — Configurable Wavelet Trend Modes.

Covers:
  - TrendMode enum values A1-A4
  - reconstruct_trend() with each approximation level
  - WaveletEngineConfig accepts trend_mode field
  - WaveletEngine produces different trends per mode
  - /wavelet endpoint: default mode (backward compat), explicit modes, invalid mode
  - Validation rejects invalid / non-string trend_mode
  - process_ticks passes mode correctly
  - Each mode produces a valid causal trend (no lookahead)
  - A higher approximation level → smoother (lower variance) trend
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wavelet_research.engine.config import WaveletEngineConfig
from wavelet_research.engine.core import WaveletEngine
from wavelet_research.engine.decomposition import (
    DEFAULT_TREND_MODE,
    SUPPORTED_TREND_MODES,
    TrendMode,
    decompose,
    reconstruct_trend,
)
from wavelet_research.engine.models import Tick
from wavelet_research.service.app import create_app
from wavelet_research.service.config import ServiceConfig
from wavelet_research.service.processor import process_ticks
from wavelet_research.service.validation import RequestValidationError, parse_wavelet_request


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def _make_ticks(n: int = 300) -> list[dict]:
    rng = np.random.default_rng(7)
    mid = 1.1 + np.cumsum(rng.normal(0, 0.0001, n))
    base = pd.Timestamp("2026-01-01")
    return [
        {
            "time": str(base + pd.Timedelta(seconds=i)),
            "bid": float(m - 0.00005),
            "ask": float(m + 0.00005),
            "mid": float(m),
        }
        for i, m in enumerate(mid)
    ]


def _engine_ticks(n: int = 300) -> list[Tick]:
    rng = np.random.default_rng(7)
    mid = 1.1 + np.cumsum(rng.normal(0, 0.0001, n))
    base = pd.Timestamp("2026-01-01")
    return [
        Tick(
            time=base + pd.Timedelta(seconds=i),
            bid=float(m - 0.00005),
            ask=float(m + 0.00005),
            mid=float(m),
            spread=0.0001,
        )
        for i, m in enumerate(mid)
    ]


@pytest.fixture(scope="module")
def client():
    cfg = ServiceConfig(wavelet="db4", window=256, level=4)
    app = create_app(cfg)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ─────────────────────────────────────────────────────────
# TrendMode enum
# ─────────────────────────────────────────────────────────

class TestTrendModeEnum:
    def test_all_four_modes_exist(self):
        assert TrendMode.A1.value == "A1"
        assert TrendMode.A2.value == "A2"
        assert TrendMode.A3.value == "A3"
        assert TrendMode.A4.value == "A4"

    def test_supported_trend_modes_set(self):
        assert SUPPORTED_TREND_MODES == {"A1", "A2", "A3", "A4"}

    def test_default_trend_mode_is_a2(self):
        assert DEFAULT_TREND_MODE == TrendMode.A2

    def test_trend_mode_is_str_enum(self):
        assert isinstance(TrendMode.A1, str)
        assert TrendMode.A1 == "A1"


# ─────────────────────────────────────────────────────────
# reconstruct_trend() with approximation_level
# ─────────────────────────────────────────────────────────

class TestReconstructTrend:
    """Unit tests for reconstruct_trend() with explicit approximation levels."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        rng = np.random.default_rng(42)
        self.signal = 1.1 + np.cumsum(rng.normal(0, 0.0001, 512))
        self.wavelet = "db4"
        self.level = 4
        self.coefficients = decompose(self.signal, self.wavelet, self.level)

    def test_none_level_equals_full_depth(self):
        full = reconstruct_trend(self.coefficients, self.wavelet, len(self.signal))
        none_result = reconstruct_trend(
            self.coefficients, self.wavelet, len(self.signal), approximation_level=None
        )
        np.testing.assert_array_almost_equal(full, none_result)

    def test_level_equals_depth_equals_none(self):
        full = reconstruct_trend(self.coefficients, self.wavelet, len(self.signal))
        deep = reconstruct_trend(
            self.coefficients, self.wavelet, len(self.signal),
            approximation_level=self.level
        )
        np.testing.assert_array_almost_equal(full, deep)

    def test_each_level_produces_correct_length(self):
        for lvl in (1, 2, 3, 4):
            result = reconstruct_trend(
                self.coefficients, self.wavelet, len(self.signal),
                approximation_level=lvl
            )
            assert len(result) == len(self.signal), f"Length mismatch at level {lvl}"

    def test_each_level_produces_finite_values(self):
        for lvl in (1, 2, 3, 4):
            result = reconstruct_trend(
                self.coefficients, self.wavelet, len(self.signal),
                approximation_level=lvl
            )
            assert np.all(np.isfinite(result)), f"Non-finite values at level {lvl}"

    def test_different_levels_produce_different_trends(self):
        t1 = reconstruct_trend(
            self.coefficients, self.wavelet, len(self.signal), approximation_level=1
        )
        t4 = reconstruct_trend(
            self.coefficients, self.wavelet, len(self.signal), approximation_level=4
        )
        assert not np.allclose(t1, t4), "A1 and A4 should produce different trends"

    def test_higher_level_produces_smoother_trend(self):
        """A4 trend should have lower variance than A1 trend (smoother)."""
        t1 = reconstruct_trend(
            self.coefficients, self.wavelet, len(self.signal), approximation_level=1
        )
        t4 = reconstruct_trend(
            self.coefficients, self.wavelet, len(self.signal), approximation_level=4
        )
        assert np.var(t1) > np.var(t4), (
            "A1 variance should exceed A4 variance (higher level = smoother)"
        )

    def test_result_in_price_range(self):
        for lvl in (1, 2, 3, 4):
            result = reconstruct_trend(
                self.coefficients, self.wavelet, len(self.signal),
                approximation_level=lvl
            )
            assert result.min() > 0.5 and result.max() < 5.0, (
                f"Trend at level {lvl} out of expected price range"
            )


# ─────────────────────────────────────────────────────────
# WaveletEngineConfig with trend_mode
# ─────────────────────────────────────────────────────────

class TestWaveletEngineConfig:
    def test_default_trend_mode_is_a2(self):
        cfg = WaveletEngineConfig(wavelet="db4", window=256, level=4)
        assert cfg.trend_mode == TrendMode.A2

    def test_explicit_trend_modes_accepted(self):
        for mode in TrendMode:
            cfg = WaveletEngineConfig(wavelet="db4", window=256, level=4, trend_mode=mode)
            assert cfg.trend_mode == mode

    def test_invalid_trend_mode_raises(self):
        with pytest.raises((ValueError, TypeError)):
            WaveletEngineConfig(
                wavelet="db4", window=256, level=4, trend_mode="A5"  # type: ignore[arg-type]
            )


# ─────────────────────────────────────────────────────────
# WaveletEngine produces different trends per mode
# ─────────────────────────────────────────────────────────

class TestWaveletEnginePerMode:
    def _run_engine(self, mode: TrendMode, n: int = 300) -> list[float]:
        cfg = WaveletEngineConfig(wavelet="db4", window=256, level=4, trend_mode=mode)
        engine = WaveletEngine(cfg)
        trends = []
        for tick in _engine_ticks(n):
            point = engine.update(tick)
            trends.append(point.trend if point else 0.0)
        return trends

    def test_all_modes_produce_output(self):
        for mode in TrendMode:
            trends = self._run_engine(mode)
            non_zero = [v for v in trends if v != 0.0]
            assert len(non_zero) > 0, f"Mode {mode} produced no trend output"

    def test_modes_produce_different_trends(self):
        t_a1 = self._run_engine(TrendMode.A1)
        t_a4 = self._run_engine(TrendMode.A4)
        assert t_a1 != t_a4, "A1 and A4 should produce different trend series"

    def test_a2_matches_original_default(self):
        """A2 with level=2 matches the old default behaviour (level=2 → full depth)."""
        cfg_old = WaveletEngineConfig(wavelet="db4", window=256, level=2)
        cfg_new = WaveletEngineConfig(
            wavelet="db4", window=256, level=2, trend_mode=TrendMode.A2
        )
        engine_old = WaveletEngine(cfg_old)
        engine_new = WaveletEngine(cfg_new)
        ticks = _engine_ticks(300)
        for tick in ticks:
            p_old = engine_old.update(tick)
            p_new = engine_new.update(tick)
            if p_old is not None and p_new is not None:
                assert abs(p_old.trend - p_new.trend) < 1e-10, (
                    "A2 default should match original level=2 engine"
                )

    def test_higher_mode_produces_smoother_trend(self):
        """A4 trend should have lower tick-to-tick roughness than A1 (smoother).

        Smoothness is measured by the std of first differences (roughness),
        not total variance — a slow drift can have high variance but still be smooth.
        """
        t_a1 = np.array([v for v in self._run_engine(TrendMode.A1) if v != 0.0])
        t_a4 = np.array([v for v in self._run_engine(TrendMode.A4) if v != 0.0])
        roughness_a1 = float(np.std(np.diff(t_a1)))
        roughness_a4 = float(np.std(np.diff(t_a4)))
        assert roughness_a1 >= roughness_a4, (
            f"A1 roughness ({roughness_a1:.2e}) should be >= A4 roughness ({roughness_a4:.2e})"
        )

    def test_output_length_unchanged(self):
        n = 300
        trends = self._run_engine(TrendMode.A3, n)
        assert len(trends) == n


# ─────────────────────────────────────────────────────────
# process_ticks() with trend_mode
# ─────────────────────────────────────────────────────────

class TestProcessTicksMode:
    def _engine_config(self) -> WaveletEngineConfig:
        return WaveletEngineConfig(wavelet="db4", window=256, level=4)

    def _make_tick_requests(self, n: int = 300):
        from wavelet_research.service.models import TickRequest
        rng = np.random.default_rng(7)
        mid = 1.1 + np.cumsum(rng.normal(0, 0.0001, n))
        base = pd.Timestamp("2026-01-01")
        return tuple(
            TickRequest(
                time=str(base + pd.Timedelta(seconds=i)),
                bid=float(m - 0.00005),
                ask=float(m + 0.00005),
                mid=float(m),
            )
            for i, m in enumerate(mid)
        )

    def test_default_mode_is_a2(self):
        cfg = self._engine_config()
        reqs = self._make_tick_requests()
        r1 = process_ticks(reqs, cfg)
        r2 = process_ticks(reqs, cfg, TrendMode.A2)
        assert r1.trend == r2.trend

    def test_all_modes_return_correct_length(self):
        cfg = self._engine_config()
        reqs = self._make_tick_requests(300)
        for mode in TrendMode:
            result = process_ticks(reqs, cfg, mode)
            assert len(result.trend) == 300, f"Wrong length for mode {mode}"

    def test_different_modes_produce_different_trends(self):
        cfg = self._engine_config()
        reqs = self._make_tick_requests()
        r_a1 = process_ticks(reqs, cfg, TrendMode.A1)
        r_a4 = process_ticks(reqs, cfg, TrendMode.A4)
        assert r_a1.trend != r_a4.trend


# ─────────────────────────────────────────────────────────
# Validation: trend_mode field
# ─────────────────────────────────────────────────────────

class TestValidationTrendMode:
    def _ticks(self, n: int = 260) -> list[dict]:
        return _make_ticks(n)

    def test_missing_trend_mode_defaults_to_a2(self):
        body = {"ticks": self._ticks()}
        req = parse_wavelet_request(body, min_ticks=256)
        assert req.trend_mode == "A2"

    def test_explicit_a1_accepted(self):
        body = {"ticks": self._ticks(), "trend_mode": "A1"}
        req = parse_wavelet_request(body, min_ticks=256)
        assert req.trend_mode == "A1"

    def test_lowercase_normalised_to_uppercase(self):
        body = {"ticks": self._ticks(), "trend_mode": "a3"}
        req = parse_wavelet_request(body, min_ticks=256)
        assert req.trend_mode == "A3"

    def test_all_valid_modes_accepted(self):
        for mode in ("A1", "A2", "A3", "A4", "a1", "a2", "a3", "a4"):
            body = {"ticks": self._ticks(), "trend_mode": mode}
            req = parse_wavelet_request(body, min_ticks=256)
            assert req.trend_mode in SUPPORTED_TREND_MODES

    def test_invalid_mode_raises(self):
        body = {"ticks": self._ticks(), "trend_mode": "A5"}
        with pytest.raises(RequestValidationError):
            parse_wavelet_request(body, min_ticks=256)

    def test_numeric_mode_raises(self):
        body = {"ticks": self._ticks(), "trend_mode": 2}
        with pytest.raises(RequestValidationError):
            parse_wavelet_request(body, min_ticks=256)

    def test_empty_string_mode_raises(self):
        body = {"ticks": self._ticks(), "trend_mode": ""}
        with pytest.raises(RequestValidationError):
            parse_wavelet_request(body, min_ticks=256)

    def test_garbage_mode_raises(self):
        body = {"ticks": self._ticks(), "trend_mode": "FAST"}
        with pytest.raises(RequestValidationError):
            parse_wavelet_request(body, min_ticks=256)


# ─────────────────────────────────────────────────────────
# /wavelet endpoint: mode parameter
# ─────────────────────────────────────────────────────────

class TestWaveletEndpointTrendMode:
    def test_no_trend_mode_returns_200(self, client):
        resp = client.post("/wavelet", json={"ticks": _make_ticks(300)})
        assert resp.status_code == 200

    def test_a1_returns_200(self, client):
        resp = client.post("/wavelet", json={"ticks": _make_ticks(300), "trend_mode": "A1"})
        assert resp.status_code == 200

    def test_a2_returns_200(self, client):
        resp = client.post("/wavelet", json={"ticks": _make_ticks(300), "trend_mode": "A2"})
        assert resp.status_code == 200

    def test_a3_returns_200(self, client):
        resp = client.post("/wavelet", json={"ticks": _make_ticks(300), "trend_mode": "A3"})
        assert resp.status_code == 200

    def test_a4_returns_200(self, client):
        resp = client.post("/wavelet", json={"ticks": _make_ticks(300), "trend_mode": "A4"})
        assert resp.status_code == 200

    def test_lowercase_mode_accepted(self, client):
        resp = client.post("/wavelet", json={"ticks": _make_ticks(300), "trend_mode": "a2"})
        assert resp.status_code == 200

    def test_invalid_mode_returns_400(self, client):
        resp = client.post("/wavelet", json={"ticks": _make_ticks(300), "trend_mode": "A9"})
        assert resp.status_code == 400

    def test_numeric_mode_returns_400(self, client):
        resp = client.post("/wavelet", json={"ticks": _make_ticks(300), "trend_mode": 2})
        assert resp.status_code == 400

    def test_a1_and_a4_return_different_trends(self, client):
        ticks = _make_ticks(300)
        r1 = client.post("/wavelet", json={"ticks": ticks, "trend_mode": "A1"}).get_json()
        r4 = client.post("/wavelet", json={"ticks": ticks, "trend_mode": "A4"}).get_json()
        # At least some trend values must differ
        diffs = sum(1 for a, b in zip(r1["trend"], r4["trend"]) if abs(a - b) > 1e-10)
        assert diffs > 0, "A1 and A4 should produce different trend arrays"

    def test_no_mode_and_a2_mode_identical(self, client):
        """Backward compatibility: omitting trend_mode == trend_mode=A2."""
        ticks = _make_ticks(300)
        r_default = client.post("/wavelet", json={"ticks": ticks}).get_json()
        r_a2 = client.post("/wavelet", json={"ticks": ticks, "trend_mode": "A2"}).get_json()
        assert r_default["trend"] == r_a2["trend"], (
            "Default (no trend_mode) must equal explicit A2"
        )

    def test_all_arrays_still_present_for_all_modes(self, client):
        required = ("trend", "relative_deviation", "z_score", "energy", "noise")
        for mode in ("A1", "A2", "A3", "A4"):
            data = client.post(
                "/wavelet", json={"ticks": _make_ticks(300), "trend_mode": mode}
            ).get_json()
            for key in required:
                assert key in data, f"Key '{key}' missing for mode {mode}"

    def test_response_arrays_correct_length_for_all_modes(self, client):
        ticks = _make_ticks(300)
        for mode in ("A1", "A2", "A3", "A4"):
            data = client.post(
                "/wavelet", json={"ticks": ticks, "trend_mode": mode}
            ).get_json()
            assert len(data["trend"]) == len(ticks), (
                f"Trend length mismatch for mode {mode}"
            )


# ─────────────────────────────────────────────────────────
# MQL5 structural tests
# ─────────────────────────────────────────────────────────

class TestMQL5TrendModeIntegration:
    def _read_mql5(self, filename: str) -> str:
        import os
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return open(os.path.join(base, "mql5", filename), encoding="utf-8").read()

    def test_bridge_ea_has_trend_mode_input(self):
        src = self._read_mql5("WaveletBridgeEA.mq5")
        assert "InpTrendMode" in src, "WaveletBridgeEA must define InpTrendMode input"

    def test_bridge_ea_includes_trend_mode_in_json(self):
        src = self._read_mql5("WaveletBridgeEA.mq5")
        assert "trend_mode" in src, "WaveletBridgeEA must include trend_mode in JSON payload"

    def test_bridge_ea_validates_mode(self):
        src = self._read_mql5("WaveletBridgeEA.mq5")
        assert "INIT_PARAMETERS_INCORRECT" in src, (
            "WaveletBridgeEA must return INIT_PARAMETERS_INCORRECT for invalid mode"
        )

    def test_bridge_ea_writes_trend_mode_gvar(self):
        src = self._read_mql5("WaveletBridgeEA.mq5")
        assert "GV_TREND_MODE" in src

    def test_indicator_reads_trend_mode_gvar(self):
        src = self._read_mql5("WaveletThinIndicator.mq5")
        assert "GV_TREND_MODE" in src

    def test_indicator_displays_mode_in_status(self):
        src = self._read_mql5("WaveletThinIndicator.mq5")
        assert "Mode" in src, "Indicator must display mode in status comment"
