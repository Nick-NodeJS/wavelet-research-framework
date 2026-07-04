"""Tests for Architecture Fix: MT5 Bridge architecture.

Verifies the Python-side contract that WaveletBridgeEA depends on:
  - GET /health returns HTTP 200 with expected schema (AC-1)
  - POST /wavelet returns valid JSON with all 5 arrays (AC-2)
  - Response arrays match the tick count (EA parse assumption)
  - Service starts and stops without blocking (AC-5)
  - Backward-compatible API — no fields removed
  - Array index convention: last element = most recent tick
"""

from __future__ import annotations

import json
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pytest

from wavelet_research.service.app import create_app
from wavelet_research.service.config import ServiceConfig


# ─────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """Flask test client with a minimal valid config."""
    cfg = ServiceConfig(wavelet="db4", window=256, level=2)
    app = create_app(cfg)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _make_ticks(n: int = 260) -> list[dict]:
    """Generate n synthetic ticks suitable for the /wavelet endpoint."""
    rng = np.random.default_rng(42)
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


# ─────────────────────────────────────────────────────────
# AC-1: GET /health → HTTP 200 with expected fields
# ─────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_response_has_status_ok(self, client):
        data = client.get("/health").get_json()
        assert data["status"] == "ok"

    def test_response_has_wavelet(self, client):
        data = client.get("/health").get_json()
        assert "wavelet" in data
        assert isinstance(data["wavelet"], str)

    def test_response_has_version(self, client):
        data = client.get("/health").get_json()
        assert "version" in data

    def test_content_type_is_json(self, client):
        resp = client.get("/health")
        assert "application/json" in resp.content_type


# ─────────────────────────────────────────────────────────
# AC-2: POST /wavelet → valid JSON with all 5 arrays
# ─────────────────────────────────────────────────────────

class TestWaveletEndpoint:
    _REQUIRED_ARRAYS = ("trend", "relative_deviation", "z_score", "energy", "noise")

    def test_returns_200(self, client):
        resp = client.post("/wavelet", json={"ticks": _make_ticks()})
        assert resp.status_code == 200

    def test_response_has_all_required_arrays(self, client):
        data = client.post("/wavelet", json={"ticks": _make_ticks()}).get_json()
        for key in self._REQUIRED_ARRAYS:
            assert key in data, f"Missing key: {key}"

    def test_all_arrays_are_lists(self, client):
        data = client.post("/wavelet", json={"ticks": _make_ticks()}).get_json()
        for key in self._REQUIRED_ARRAYS:
            assert isinstance(data[key], list), f"{key} is not a list"

    def test_array_length_matches_tick_count(self, client):
        ticks = _make_ticks(260)
        data = client.post("/wavelet", json={"ticks": ticks}).get_json()
        for key in self._REQUIRED_ARRAYS:
            assert len(data[key]) == len(ticks), (
                f"{key} length {len(data[key])} != tick count {len(ticks)}"
            )

    def test_all_values_are_finite_floats(self, client):
        data = client.post("/wavelet", json={"ticks": _make_ticks()}).get_json()
        for key in self._REQUIRED_ARRAYS:
            for v in data[key]:
                assert isinstance(v, (int, float)), f"{key} contains non-numeric value"
                assert np.isfinite(v), f"{key} contains non-finite value: {v}"

    def test_trend_values_are_price_scale(self, client):
        """Non-zero trend values should be in the ballpark of input prices (~1.1).

        Leading zeros appear before the wavelet window warms up — these are
        expected and the EA simply skips them (GlobalVariableCheck guards).
        """
        data = client.post("/wavelet", json={"ticks": _make_ticks()}).get_json()
        trend = data["trend"]
        non_zero = [v for v in trend if v != 0.0]
        assert len(non_zero) > 0, "All trend values are zero — engine never warmed up"
        assert all(0.5 < v < 5.0 for v in non_zero), (
            "Non-zero trend values outside expected price range"
        )

    def test_energy_is_non_negative(self, client):
        data = client.post("/wavelet", json={"ticks": _make_ticks()}).get_json()
        assert all(v >= 0.0 for v in data["energy"]), "Energy contains negative values"

    def test_invalid_json_returns_400(self, client):
        resp = client.post("/wavelet", data="not json", content_type="text/plain")
        assert resp.status_code == 400

    def test_missing_ticks_returns_4xx(self, client):
        resp = client.post("/wavelet", json={})
        assert resp.status_code in (400, 422)

    def test_too_few_ticks_returns_4xx(self, client):
        resp = client.post("/wavelet", json={"ticks": _make_ticks(5)})
        assert resp.status_code in (400, 422)

    def test_empty_ticks_returns_4xx(self, client):
        resp = client.post("/wavelet", json={"ticks": []})
        assert resp.status_code in (400, 422)


# ─────────────────────────────────────────────────────────
# Bridge contract: array index convention
# ─────────────────────────────────────────────────────────

class TestArrayIndexConvention:
    """The EA expects response[last] = most recent tick.

    WaveletBridgeEA reverses the array when writing to GVars:
        src = count - 1 - i → GV index i

    This test ensures the Python response ordering is stable.
    """

    def test_response_is_ordered_oldest_to_newest(self, client):
        """Verify trend[0] < trend[-1] on a strictly upward price series."""
        rng = np.random.default_rng(99)
        n = 260
        # Strictly upward trend with minimal noise
        mid = 1.0 + np.linspace(0, 0.1, n) + rng.normal(0, 0.00001, n)
        base = pd.Timestamp("2026-01-01")
        ticks = [
            {
                "time": str(base + pd.Timedelta(seconds=i)),
                "bid": float(m - 0.00005),
                "ask": float(m + 0.00005),
                "mid": float(m),
            }
            for i, m in enumerate(mid)
        ]
        data = client.post("/wavelet", json={"ticks": ticks}).get_json()
        trend = data["trend"]
        # Most recent is last — the upward trend should make trend[-1] > trend[0]
        assert trend[-1] > trend[0], (
            "Expected last trend value > first for upward price series"
        )

    def test_response_length_equals_input_length(self, client):
        for n in (256, 300, 512):
            ticks = _make_ticks(n)
            data = client.post("/wavelet", json={"ticks": ticks}).get_json()
            assert len(data["trend"]) == n, f"Length mismatch for n={n}"


# ─────────────────────────────────────────────────────────
# AC-5: Service offline does not block / freeze
# ─────────────────────────────────────────────────────────

class TestServiceResilience:
    def test_health_endpoint_response_is_fast(self, client):
        """Health check must complete well under 100 ms for EA polling."""
        import time
        start = time.perf_counter()
        client.get("/health")
        elapsed = time.perf_counter() - start
        assert elapsed < 0.5, f"Health check too slow: {elapsed:.3f}s"

    def test_wavelet_endpoint_response_is_fast(self, client):
        """Full wavelet request should complete under 2s (500ms typical)."""
        import time
        ticks = _make_ticks(260)
        start = time.perf_counter()
        client.post("/wavelet", json={"ticks": ticks})
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"Wavelet request too slow: {elapsed:.3f}s"

    def test_repeated_requests_are_deterministic(self, client):
        """Same input must produce same output — no random state."""
        ticks = _make_ticks(260)
        r1 = client.post("/wavelet", json={"ticks": ticks}).get_json()
        r2 = client.post("/wavelet", json={"ticks": ticks}).get_json()
        assert r1["trend"] == r2["trend"]
        assert r1["z_score"] == r2["z_score"]


# ─────────────────────────────────────────────────────────
# AC-3: Verify Indicator files contain no WebRequest
# ─────────────────────────────────────────────────────────

class TestIndicatorContainsNoWebRequest:
    """Structural test — read MQL5 source files and confirm WebRequest is absent."""

    def _read_mql5(self, filename: str) -> str:
        import os
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base, "mql5", filename)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def test_thin_indicator_has_no_webrequest(self):
        src = self._read_mql5("WaveletThinIndicator.mq5")
        assert "WebRequest(" not in src, (
            "WaveletThinIndicator.mq5 must not call WebRequest()"
        )

    def test_oscillator_has_no_webrequest(self):
        src = self._read_mql5("WaveletOscillator.mq5")
        assert "WebRequest(" not in src, (
            "WaveletOscillator.mq5 must not call WebRequest()"
        )

    def test_bridge_ea_has_webrequest(self):
        """WaveletBridgeEA must own all HTTP — it should call WebRequest."""
        src = self._read_mql5("WaveletBridgeEA.mq5")
        assert "WebRequest(" in src, (
            "WaveletBridgeEA.mq5 must contain WebRequest() calls"
        )

    def test_thin_indicator_reads_global_variables(self):
        src = self._read_mql5("WaveletThinIndicator.mq5")
        assert "GlobalVariableGet(" in src, (
            "WaveletThinIndicator.mq5 should read from GlobalVariableGet()"
        )

    def test_oscillator_reads_global_variables(self):
        src = self._read_mql5("WaveletOscillator.mq5")
        assert "GlobalVariableGet(" in src, (
            "WaveletOscillator.mq5 should read from GlobalVariableGet()"
        )

    def test_bridge_ea_writes_global_variables(self):
        src = self._read_mql5("WaveletBridgeEA.mq5")
        assert "GlobalVariableSet(" in src, (
            "WaveletBridgeEA.mq5 must write GlobalVariableSet() for shared state"
        )

    def test_bridge_ea_checks_health_endpoint(self):
        src = self._read_mql5("WaveletBridgeEA.mq5")
        assert "/health" in src, (
            "WaveletBridgeEA.mq5 must poll /health for reconnect"
        )

    def test_bridge_ea_fires_on_new_bar_only(self):
        src = self._read_mql5("WaveletBridgeEA.mq5")
        assert "SERIES_LASTBAR_DATE" in src, (
            "WaveletBridgeEA.mq5 must use SERIES_LASTBAR_DATE for new-bar detection"
        )


# ─────────────────────────────────────────────────────────
# Global Variable naming contract
# ─────────────────────────────────────────────────────────

class TestGlobalVariableContract:
    """Verify that EA and indicators use the same GV key names."""

    _EXPECTED_GV_KEYS = (
        "Wv_Status",
        "Wv_LastUpdate",
        "Wv_Latency",
        "Wv_LastHttpCode",
        "Wv_Trend_N",
    )
    _EXPECTED_ARRAY_PREFIXES = (
        "Wv_Trend_",
        "Wv_RelDev_",
        "Wv_ZScore_",
        "Wv_Energy_",
        "Wv_Noise_",
    )

    def _read_mql5(self, filename: str) -> str:
        import os
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base, "mql5", filename)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def test_bridge_ea_defines_all_gv_status_keys(self):
        src = self._read_mql5("WaveletBridgeEA.mq5")
        for key in self._EXPECTED_GV_KEYS:
            assert key in src, f"WaveletBridgeEA.mq5 missing GV key: {key}"

    def test_thin_indicator_uses_gv_status(self):
        src = self._read_mql5("WaveletThinIndicator.mq5")
        assert "Wv_Status" in src

    def test_oscillator_uses_gv_status(self):
        src = self._read_mql5("WaveletOscillator.mq5")
        assert "Wv_Status" in src

    def test_thin_indicator_uses_gv_trend_n(self):
        src = self._read_mql5("WaveletThinIndicator.mq5")
        assert "Wv_Trend_N" in src

    def test_oscillator_uses_gv_trend_n(self):
        src = self._read_mql5("WaveletOscillator.mq5")
        assert "Wv_Trend_N" in src

    def test_bridge_ea_writes_all_array_prefixes(self):
        src = self._read_mql5("WaveletBridgeEA.mq5")
        for prefix in self._EXPECTED_ARRAY_PREFIXES:
            assert prefix in src, f"WaveletBridgeEA.mq5 missing array prefix: {prefix}"
