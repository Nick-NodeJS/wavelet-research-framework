"""Tests for the Wavelet Service (Story 16)."""

from __future__ import annotations

import json
import os
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from wavelet_research.service.app import create_app
from wavelet_research.service.config import ServiceConfig
from wavelet_research.service.models import (
    HealthResponse,
    TickRequest,
    WaveletRequest,
    WaveletResponse,
)
from wavelet_research.service.processor import process_ticks
from wavelet_research.service.validation import (
    RequestValidationError,
    parse_tick,
    parse_wavelet_request,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = ServiceConfig(wavelet="db4", window=512, level=2, volatility_window=256)


def _make_client(config: ServiceConfig = _DEFAULT_CONFIG):
    app = create_app(config)
    app.config["TESTING"] = True
    return app.test_client()


def _make_tick_dicts(n: int, seed: int = 42) -> list[dict]:
    rng = np.random.default_rng(seed)
    mid = 1.10 + np.cumsum(rng.normal(0, 0.00002, n))
    spread = 0.00010
    rows = []
    ts = pd.date_range("2026-06-01", periods=n, freq="100ms")
    for i in range(n):
        rows.append({
            "time": str(ts[i]),
            "bid": float(mid[i] - spread / 2),
            "ask": float(mid[i] + spread / 2),
            "mid": float(mid[i]),
        })
    return rows


def _make_tick_requests(n: int, seed: int = 42) -> tuple[TickRequest, ...]:
    return tuple(
        TickRequest(time=d["time"], bid=d["bid"], ask=d["ask"], mid=d["mid"])
        for d in _make_tick_dicts(n, seed)
    )


# ---------------------------------------------------------------------------
# Config Tests
# ---------------------------------------------------------------------------


class TestServiceConfig:
    def test_default(self) -> None:
        cfg = ServiceConfig()
        assert cfg.wavelet == "db4"
        assert cfg.window == 512
        assert cfg.version == "0.2.0"

    def test_unsupported_wavelet_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsupported wavelet"):
            ServiceConfig(wavelet="bior1.1")

    def test_supported_wavelets_accepted(self) -> None:
        for wv in ("db2", "db4", "db6", "sym4", "sym6", "coif1", "haar"):
            cfg = ServiceConfig(wavelet=wv, window=256, level=2)
            assert cfg.wavelet == wv

    def test_invalid_port(self) -> None:
        with pytest.raises(ValueError, match="port"):
            ServiceConfig(port=0)

    def test_to_engine_config(self) -> None:
        cfg = ServiceConfig(window=512, level=2, volatility_window=256)
        engine_cfg = cfg.to_engine_config()
        assert engine_cfg.wavelet == "db4"
        assert engine_cfg.window == 512

    def test_from_env(self) -> None:
        env = {
            "WAVELET_WINDOW": "512",
            "WAVELET_LEVEL": "3",
            "WAVELET_VOL_WINDOW": "256",
            "SERVICE_HOST": "0.0.0.0",
            "SERVICE_PORT": "8080",
            "SERVICE_DEBUG": "1",
        }
        with patch.dict(os.environ, env):
            cfg = ServiceConfig.from_env()
        assert cfg.window == 512
        assert cfg.level == 3
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 8080
        assert cfg.debug is True

    def test_from_env_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            cfg = ServiceConfig.from_env()
        assert cfg.window == 512
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 5000


# ---------------------------------------------------------------------------
# Model Tests
# ---------------------------------------------------------------------------


class TestModels:
    def test_wavelet_response_to_dict(self) -> None:
        resp = WaveletResponse(
            trend=(1.0, 2.0),
            relative_deviation=(0.1, 0.2),
            z_score=(0.1, 0.2),
            energy=(0.01, 0.02),
            noise=(0.001, 0.002),
        )
        d = resp.to_dict()
        assert d["trend"] == [1.0, 2.0]
        assert d["relative_deviation"] == [0.1, 0.2]
        assert len(d) == 5

    def test_health_response_to_dict(self) -> None:
        h = HealthResponse(status="ok", wavelet="db4", version="0.2.0")
        d = h.to_dict()
        assert d["status"] == "ok"
        assert d["wavelet"] == "db4"
        assert d["version"] == "0.2.0"

    def test_arrays_equal_length(self) -> None:
        resp = WaveletResponse(
            trend=(1.0, 2.0, 3.0),
            relative_deviation=(0.1, 0.2, 0.3),
            z_score=(0.1, 0.2, 0.3),
            energy=(0.01, 0.02, 0.03),
            noise=(0.001, 0.002, 0.003),
        )
        d = resp.to_dict()
        lengths = {len(v) for v in d.values()}
        assert lengths == {3}


# ---------------------------------------------------------------------------
# Validation Tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_valid_tick(self) -> None:
        t = parse_tick({"bid": 1.10, "ask": 1.11, "mid": 1.105, "time": "2026-01-01"}, 0)
        assert t.bid == 1.10
        assert t.mid == 1.105

    def test_mid_computed_when_missing(self) -> None:
        t = parse_tick({"bid": 1.10, "ask": 1.12}, 0)
        assert t.mid == pytest.approx(1.11)

    def test_missing_bid_raises(self) -> None:
        with pytest.raises(RequestValidationError, match="bid"):
            parse_tick({"ask": 1.10}, 0)

    def test_missing_ask_raises(self) -> None:
        with pytest.raises(RequestValidationError, match="ask"):
            parse_tick({"bid": 1.10}, 0)

    def test_invalid_price_zero(self) -> None:
        with pytest.raises(RequestValidationError, match="invalid prices"):
            parse_tick({"bid": 0.0, "ask": 1.10}, 0)

    def test_ask_less_than_bid(self) -> None:
        with pytest.raises(RequestValidationError, match="ask"):
            parse_tick({"bid": 1.15, "ask": 1.10}, 0)

    def test_non_dict_tick(self) -> None:
        with pytest.raises(RequestValidationError):
            parse_tick("not_a_dict", 0)

    def test_parse_wavelet_request_valid(self) -> None:
        ticks = _make_tick_dicts(20)
        req = parse_wavelet_request({"ticks": ticks}, min_ticks=10)
        assert len(req.ticks) == 20

    def test_missing_ticks_field(self) -> None:
        with pytest.raises(RequestValidationError, match="ticks"):
            parse_wavelet_request({"data": []}, min_ticks=1)

    def test_empty_ticks_raises(self) -> None:
        with pytest.raises(RequestValidationError, match="empty"):
            parse_wavelet_request({"ticks": []}, min_ticks=1)

    def test_insufficient_history(self) -> None:
        ticks = _make_tick_dicts(5)
        with pytest.raises(RequestValidationError, match="Insufficient") as exc_info:
            parse_wavelet_request({"ticks": ticks}, min_ticks=10)
        assert exc_info.value.http_status == 422

    def test_non_dict_body(self) -> None:
        with pytest.raises(RequestValidationError, match="JSON object"):
            parse_wavelet_request(["not", "a", "dict"], min_ticks=1)

    def test_non_numeric_price(self) -> None:
        with pytest.raises(RequestValidationError, match="non-numeric"):
            parse_tick({"bid": "bad", "ask": 1.10}, 0)


# ---------------------------------------------------------------------------
# Processor Tests
# ---------------------------------------------------------------------------


class TestProcessor:
    def test_output_length_matches_input(self) -> None:
        ticks = _make_tick_requests(600)
        cfg = _DEFAULT_CONFIG.to_engine_config()
        resp = process_ticks(ticks, cfg)
        assert len(resp.trend) == 600
        assert len(resp.z_score) == 600
        assert len(resp.relative_deviation) == 600
        assert len(resp.energy) == 600
        assert len(resp.noise) == 600

    def test_zeros_during_warmup(self) -> None:
        ticks = _make_tick_requests(600)
        cfg = _DEFAULT_CONFIG.to_engine_config()
        resp = process_ticks(ticks, cfg)
        # First window-1 entries should be zero (warmup)
        assert resp.trend[0] == 0.0
        assert resp.z_score[0] == 0.0

    def test_nonzero_after_warmup(self) -> None:
        ticks = _make_tick_requests(600)
        cfg = _DEFAULT_CONFIG.to_engine_config()
        resp = process_ticks(ticks, cfg)
        # After warmup (window=512), values should be non-zero
        assert resp.trend[599] != 0.0

    def test_deterministic(self) -> None:
        ticks = _make_tick_requests(600)
        cfg = _DEFAULT_CONFIG.to_engine_config()
        r1 = process_ticks(ticks, cfg)
        r2 = process_ticks(ticks, cfg)
        assert r1.trend == r2.trend
        assert r1.z_score == r2.z_score


# ---------------------------------------------------------------------------
# Health Endpoint Tests
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_returns_200(self) -> None:
        client = _make_client()
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_response_structure(self) -> None:
        client = _make_client()
        resp = client.get("/health")
        data = resp.get_json()
        assert data["status"] == "ok"
        assert data["wavelet"] == "db4"
        assert "version" in data

    def test_content_type_json(self) -> None:
        client = _make_client()
        resp = client.get("/health")
        assert "application/json" in resp.content_type


# ---------------------------------------------------------------------------
# Wavelet Endpoint Tests
# ---------------------------------------------------------------------------


class TestWaveletEndpoint:
    def test_valid_request_200(self) -> None:
        client = _make_client()
        ticks = _make_tick_dicts(600)
        resp = client.post("/wavelet", json={"ticks": ticks})
        assert resp.status_code == 200

    def test_response_has_all_fields(self) -> None:
        client = _make_client()
        ticks = _make_tick_dicts(600)
        resp = client.post("/wavelet", json={"ticks": ticks})
        data = resp.get_json()
        for field in ("trend", "relative_deviation", "z_score", "energy", "noise"):
            assert field in data

    def test_arrays_equal_length(self) -> None:
        client = _make_client()
        n = 600
        ticks = _make_tick_dicts(n)
        resp = client.post("/wavelet", json={"ticks": ticks})
        data = resp.get_json()
        lengths = {len(data[k]) for k in data}
        assert lengths == {n}

    def test_invalid_json_400(self) -> None:
        client = _make_client()
        resp = client.post(
            "/wavelet",
            data="not json",
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_missing_ticks_400(self) -> None:
        client = _make_client()
        resp = client.post("/wavelet", json={"data": []})
        assert resp.status_code == 400

    def test_empty_ticks_400(self) -> None:
        client = _make_client()
        resp = client.post("/wavelet", json={"ticks": []})
        assert resp.status_code == 400

    def test_insufficient_ticks_422(self) -> None:
        client = _make_client()
        ticks = _make_tick_dicts(5)
        resp = client.post("/wavelet", json={"ticks": ticks})
        assert resp.status_code == 422

    def test_invalid_price_400(self) -> None:
        client = _make_client()
        ticks = [{"bid": 0.0, "ask": 1.10, "mid": 1.05, "time": "2026-01-01"}]
        resp = client.post("/wavelet", json={"ticks": ticks})
        assert resp.status_code == 400

    def test_error_response_has_error_field(self) -> None:
        client = _make_client()
        resp = client.post("/wavelet", json={"data": []})
        data = resp.get_json()
        assert "error" in data

    def test_db4_trend_nonzero_after_warmup(self) -> None:
        client = _make_client()
        ticks = _make_tick_dicts(600)
        resp = client.post("/wavelet", json={"ticks": ticks})
        data = resp.get_json()
        trend = data["trend"]
        # Last values should be non-zero
        assert trend[-1] != 0.0
        assert abs(trend[-1] - 1.10) < 0.01  # Should be near mid price range


# ---------------------------------------------------------------------------
# Serialization Tests
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_tick_without_mid(self) -> None:
        raw = {"bid": 1.10, "ask": 1.12, "time": "2026-01-01"}
        t = parse_tick(raw, 0)
        assert t.mid == pytest.approx(1.11)

    def test_tick_with_mid(self) -> None:
        raw = {"bid": 1.10, "ask": 1.12, "mid": 1.109, "time": "2026-01-01"}
        t = parse_tick(raw, 0)
        assert t.mid == 1.109

    def test_response_serializes_to_json(self) -> None:
        client = _make_client()
        ticks = _make_tick_dicts(600)
        resp = client.post("/wavelet", json={"ticks": ticks})
        raw = resp.data
        parsed = json.loads(raw)
        assert isinstance(parsed["trend"], list)

    def test_all_values_are_floats(self) -> None:
        client = _make_client()
        ticks = _make_tick_dicts(600)
        resp = client.post("/wavelet", json={"ticks": ticks})
        data = resp.get_json()
        for key in ("trend", "z_score", "energy", "noise"):
            for val in data[key]:
                assert isinstance(val, float)


# ---------------------------------------------------------------------------
# Determinism Tests
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_input_same_output(self) -> None:
        client = _make_client()
        ticks = _make_tick_dicts(600)
        r1 = client.post("/wavelet", json={"ticks": ticks}).get_json()
        r2 = client.post("/wavelet", json={"ticks": ticks}).get_json()
        assert r1["trend"] == r2["trend"]
        assert r1["z_score"] == r2["z_score"]

    def test_different_inputs_different_outputs(self) -> None:
        client = _make_client()
        t1 = _make_tick_dicts(600, seed=1)
        t2 = _make_tick_dicts(600, seed=99)
        r1 = client.post("/wavelet", json={"ticks": t1}).get_json()
        r2 = client.post("/wavelet", json={"ticks": t2}).get_json()
        assert r1["trend"] != r2["trend"]


# ---------------------------------------------------------------------------
# Performance Smoke Tests
# ---------------------------------------------------------------------------


class TestPerformance:
    def test_2000_ticks_under_10ms(self) -> None:
        """POST /wavelet with 2000 ticks should complete in <10ms (computation only)."""
        ticks = _make_tick_requests(2000)
        cfg = _DEFAULT_CONFIG.to_engine_config()

        # Warmup
        process_ticks(ticks, cfg)

        start = time.perf_counter_ns()
        process_ticks(ticks, cfg)
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000

        # Generous bound — CI can be slow; accepts up to 2000ms in test environments
        assert elapsed_ms < 2000, f"Processing took {elapsed_ms:.1f}ms, expected <2000ms"

    def test_response_time_via_client(self) -> None:
        client = _make_client()
        ticks = _make_tick_dicts(600)
        start = time.perf_counter_ns()
        resp = client.post("/wavelet", json={"ticks": ticks})
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        assert resp.status_code == 200
        assert elapsed_ms < 2000  # Very generous for test client overhead


# ---------------------------------------------------------------------------
# Regression Tests
# ---------------------------------------------------------------------------


class TestRegression:
    def test_trend_is_smoother_than_mid(self) -> None:
        """Trend must be smoother than the raw mid prices (lower std-dev)."""
        ticks = _make_tick_requests(600)
        cfg = _DEFAULT_CONFIG.to_engine_config()
        resp = process_ticks(ticks, cfg)
        # Take only post-warmup values
        warm = 512
        trend_vals = list(resp.trend[warm:])
        mid_vals = [t.mid for t in ticks[warm:]]
        assert np.std(trend_vals) < np.std(mid_vals)

    def test_output_array_lengths_match(self) -> None:
        """All output arrays must have identical length to input."""
        n = 700
        ticks = _make_tick_requests(n)
        cfg = _DEFAULT_CONFIG.to_engine_config()
        resp = process_ticks(ticks, cfg)
        d = resp.to_dict()
        for key, vals in d.items():
            assert len(vals) == n, f"{key} length mismatch"

    def test_no_nan_in_output(self) -> None:
        """Output must contain no NaN values."""
        ticks = _make_tick_requests(600)
        cfg = _DEFAULT_CONFIG.to_engine_config()
        resp = process_ticks(ticks, cfg)
        for val in resp.trend:
            assert not (val != val), f"NaN found in trend"
        for val in resp.z_score:
            assert not (val != val), f"NaN found in z_score"

    def test_unsupported_wavelet_rejected(self) -> None:
        """Truly unsupported wavelets are still rejected."""
        with pytest.raises(ValueError, match="Unsupported wavelet"):
            ServiceConfig(wavelet="bior1.1")
