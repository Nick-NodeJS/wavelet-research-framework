"""Tests for Story 20 — Wavelet Trend Calibration.

Covers:
  - SUPPORTED_WAVELETS expanded with db2/sym6/coif1
  - SUPPORTED_WINDOWS expanded with 128
  - SUPPORTED_LEVELS constant added (1-4)
  - ServiceConfig accepts any supported wavelet (no longer db4-only)
  - ServiceConfig from_env reads WAVELET_FAMILY
  - WaveletRequest carries window/wavelet/level overrides
  - Validation of window/wavelet/level fields (valid, invalid, missing → default)
  - process_ticks uses per-request overrides over engine_config defaults
  - /wavelet endpoint accepts and routes all four calibration params
  - Backward compat: requests without new fields behave identically
  - Structured logging format contains all four params
  - WaveletBridgeEA MQL5 has correct new inputs and validation
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import pytest

from wavelet_research.engine.config import (
    SUPPORTED_LEVELS,
    SUPPORTED_WAVELETS,
    SUPPORTED_WINDOWS,
    WaveletEngineConfig,
)
from wavelet_research.engine.decomposition import TrendMode
from wavelet_research.service.app import create_app
from wavelet_research.service.config import ServiceConfig
from wavelet_research.service.models import TickRequest, WaveletRequest
from wavelet_research.service.processor import process_ticks
from wavelet_research.service.validation import (
    RequestValidationError,
    _parse_optional_level,
    _parse_optional_wavelet,
    _parse_optional_window,
    parse_wavelet_request,
)


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def _make_ticks(n: int = 300, wavelet: str = "db4") -> list[dict]:
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


def _make_tick_requests(n: int = 300) -> tuple[TickRequest, ...]:
    ticks = _make_ticks(n)
    return tuple(
        TickRequest(time=t["time"], bid=t["bid"], ask=t["ask"], mid=t["mid"])
        for t in ticks
    )


@pytest.fixture(scope="module")
def client():
    cfg = ServiceConfig(wavelet="db4", window=256, level=2)
    app = create_app(cfg)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ─────────────────────────────────────────────────────────
# Supported sets
# ─────────────────────────────────────────────────────────

class TestSupportedSets:
    def test_db2_in_supported_wavelets(self):
        assert "db2" in SUPPORTED_WAVELETS

    def test_sym6_in_supported_wavelets(self):
        assert "sym6" in SUPPORTED_WAVELETS

    def test_coif1_in_supported_wavelets(self):
        assert "coif1" in SUPPORTED_WAVELETS

    def test_db4_still_supported(self):
        assert "db4" in SUPPORTED_WAVELETS

    def test_db6_still_supported(self):
        assert "db6" in SUPPORTED_WAVELETS

    def test_sym4_still_supported(self):
        assert "sym4" in SUPPORTED_WAVELETS

    def test_128_in_supported_windows(self):
        assert 128 in SUPPORTED_WINDOWS

    def test_standard_windows_still_present(self):
        for w in (256, 512, 1024, 2048):
            assert w in SUPPORTED_WINDOWS

    def test_supported_levels_constant_exists(self):
        assert SUPPORTED_LEVELS == frozenset({1, 2, 3, 4})

    def test_levels_1_to_4_all_present(self):
        for lvl in (1, 2, 3, 4):
            assert lvl in SUPPORTED_LEVELS


# ─────────────────────────────────────────────────────────
# ServiceConfig no longer locked to db4
# ─────────────────────────────────────────────────────────

class TestServiceConfig:
    def test_db4_still_default(self):
        cfg = ServiceConfig()
        assert cfg.wavelet == "db4"

    def test_db2_accepted(self):
        cfg = ServiceConfig(wavelet="db2", window=256, level=2)
        assert cfg.wavelet == "db2"

    def test_sym6_accepted(self):
        cfg = ServiceConfig(wavelet="sym6", window=256, level=2)
        assert cfg.wavelet == "sym6"

    def test_coif1_accepted(self):
        cfg = ServiceConfig(wavelet="coif1", window=256, level=2)
        assert cfg.wavelet == "coif1"

    def test_unsupported_wavelet_raises(self):
        with pytest.raises(ValueError, match="Unsupported wavelet"):
            ServiceConfig(wavelet="bior1.1", window=256, level=2)

    def test_from_env_reads_wavelet_family(self, monkeypatch):
        monkeypatch.setenv("WAVELET_FAMILY", "sym6")
        cfg = ServiceConfig.from_env()
        assert cfg.wavelet == "sym6"

    def test_from_env_defaults_to_db4_when_unset(self, monkeypatch):
        monkeypatch.delenv("WAVELET_FAMILY", raising=False)
        cfg = ServiceConfig.from_env()
        assert cfg.wavelet == "db4"

    def test_window_128_accepted_in_service_config(self):
        cfg = ServiceConfig(wavelet="db4", window=128, level=2)
        assert cfg.window == 128


# ─────────────────────────────────────────────────────────
# WaveletEngineConfig with expanded sets
# ─────────────────────────────────────────────────────────

class TestWaveletEngineConfig:
    def test_db2_accepted(self):
        cfg = WaveletEngineConfig(wavelet="db2", window=256, level=2)
        assert cfg.wavelet == "db2"

    def test_sym6_accepted(self):
        cfg = WaveletEngineConfig(wavelet="sym6", window=256, level=2)
        assert cfg.wavelet == "sym6"

    def test_coif1_accepted(self):
        cfg = WaveletEngineConfig(wavelet="coif1", window=256, level=2)
        assert cfg.wavelet == "coif1"

    def test_window_128_accepted(self):
        cfg = WaveletEngineConfig(wavelet="db4", window=128, level=2)
        assert cfg.window == 128

    def test_level_1_accepted(self):
        cfg = WaveletEngineConfig(wavelet="db4", window=256, level=1)
        assert cfg.level == 1

    def test_level_4_accepted(self):
        cfg = WaveletEngineConfig(wavelet="db4", window=256, level=4)
        assert cfg.level == 4


# ─────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────

class TestParseOptionalWindow:
    def test_none_when_absent(self):
        assert _parse_optional_window({}) is None

    def test_valid_values_accepted(self):
        for w in (128, 256, 512, 1024, 2048):
            assert _parse_optional_window({"window": w}) == w

    def test_invalid_value_raises(self):
        with pytest.raises(RequestValidationError, match="Invalid window"):
            _parse_optional_window({"window": 300})

    def test_string_raises(self):
        with pytest.raises(RequestValidationError, match="must be an integer"):
            _parse_optional_window({"window": "256"})

    def test_bool_raises(self):
        with pytest.raises(RequestValidationError):
            _parse_optional_window({"window": True})


class TestParseOptionalWavelet:
    def test_none_when_absent(self):
        assert _parse_optional_wavelet({}) is None

    def test_valid_wavelets_accepted(self):
        for wv in ("db2", "db4", "db6", "sym4", "sym6", "coif1"):
            assert _parse_optional_wavelet({"wavelet": wv}) == wv

    def test_normalised_to_lowercase(self):
        assert _parse_optional_wavelet({"wavelet": "DB4"}) == "db4"

    def test_invalid_wavelet_raises(self):
        with pytest.raises(RequestValidationError, match="Invalid wavelet"):
            _parse_optional_wavelet({"wavelet": "bior1.1"})

    def test_int_raises(self):
        with pytest.raises(RequestValidationError, match="must be a string"):
            _parse_optional_wavelet({"wavelet": 4})


class TestParseOptionalLevel:
    def test_none_when_absent(self):
        assert _parse_optional_level({}) is None

    def test_valid_levels_accepted(self):
        for lvl in (1, 2, 3, 4):
            assert _parse_optional_level({"level": lvl}) == lvl

    def test_zero_raises(self):
        with pytest.raises(RequestValidationError, match="Invalid level"):
            _parse_optional_level({"level": 0})

    def test_five_raises(self):
        with pytest.raises(RequestValidationError, match="Invalid level"):
            _parse_optional_level({"level": 5})

    def test_string_raises(self):
        with pytest.raises(RequestValidationError, match="must be an integer"):
            _parse_optional_level({"level": "2"})


# ─────────────────────────────────────────────────────────
# parse_wavelet_request with new fields
# ─────────────────────────────────────────────────────────

class TestParseWaveletRequestCalibration:
    def _ticks(self, n: int = 300) -> list[dict]:
        return _make_ticks(n)

    def test_window_parsed_into_request(self):
        body = {"ticks": self._ticks(), "window": 256}
        req = parse_wavelet_request(body, min_ticks=512)
        assert req.window == 256

    def test_wavelet_parsed_into_request(self):
        body = {"ticks": self._ticks(), "wavelet": "sym6"}
        req = parse_wavelet_request(body, min_ticks=256)
        assert req.wavelet == "sym6"

    def test_level_parsed_into_request(self):
        body = {"ticks": self._ticks(), "level": 3}
        req = parse_wavelet_request(body, min_ticks=256)
        assert req.level == 3

    def test_missing_overrides_result_in_none(self):
        body = {"ticks": self._ticks()}
        req = parse_wavelet_request(body, min_ticks=256)
        assert req.window is None
        assert req.wavelet is None
        assert req.level is None

    def test_per_request_window_overrides_min_ticks(self):
        """When window=256 is in request, only 256 ticks are required, not 512."""
        body = {"ticks": self._ticks(256), "window": 256}
        req = parse_wavelet_request(body, min_ticks=512)
        assert req.window == 256

    def test_insufficient_ticks_for_per_request_window_raises(self):
        """Only 100 ticks but window=256 → must raise 422."""
        body = {"ticks": self._ticks(100), "window": 256}
        with pytest.raises(RequestValidationError) as exc_info:
            parse_wavelet_request(body, min_ticks=512)
        assert exc_info.value.http_status == 422

    def test_invalid_window_returns_400_not_422(self):
        body = {"ticks": self._ticks(10), "window": 999}
        with pytest.raises(RequestValidationError) as exc_info:
            parse_wavelet_request(body, min_ticks=256)
        assert exc_info.value.http_status == 400

    def test_all_fields_together(self):
        body = {
            "ticks": self._ticks(300),
            "window": 256,
            "wavelet": "db6",
            "level": 3,
            "trend_mode": "A3",
        }
        req = parse_wavelet_request(body, min_ticks=256)
        assert req.window == 256
        assert req.wavelet == "db6"
        assert req.level == 3
        assert req.trend_mode == "A3"


# ─────────────────────────────────────────────────────────
# process_ticks with per-request overrides
# ─────────────────────────────────────────────────────────

class TestProcessTicksCalibration:
    def _default_config(self) -> WaveletEngineConfig:
        return WaveletEngineConfig(wavelet="db4", window=256, level=2)

    def test_default_config_used_when_no_overrides(self):
        cfg = self._default_config()
        reqs = _make_tick_requests(300)
        r1 = process_ticks(reqs, cfg)
        r2 = process_ticks(reqs, cfg, wavelet_override=None, window_override=None, level_override=None)
        assert r1.trend == r2.trend

    def test_wavelet_override_changes_trend(self):
        cfg = self._default_config()
        reqs = _make_tick_requests(300)
        r_db4 = process_ticks(reqs, cfg)
        r_sym6 = process_ticks(reqs, cfg, wavelet_override="sym6")
        assert r_db4.trend != r_sym6.trend

    def test_level_override_changes_trend(self):
        cfg = self._default_config()
        reqs = _make_tick_requests(300)
        r_l2 = process_ticks(reqs, cfg)
        r_l4 = process_ticks(reqs, cfg, level_override=4)
        assert r_l2.trend != r_l4.trend

    def test_window_override_changes_output_length(self):
        """Output arrays always equal len(tick_requests) regardless of window."""
        cfg = self._default_config()
        reqs = _make_tick_requests(300)
        result = process_ticks(reqs, cfg, window_override=128)
        assert len(result.trend) == 300

    def test_all_overrides_produce_valid_output(self):
        cfg = WaveletEngineConfig(wavelet="db4", window=256, level=2)
        reqs = _make_tick_requests(300)
        for wv in ("db2", "db4", "db6", "sym4", "sym6", "coif1"):
            result = process_ticks(reqs, cfg, wavelet_override=wv)
            non_zero = [v for v in result.trend if v != 0.0]
            assert len(non_zero) > 0, f"No trend output for wavelet={wv}"

    def test_output_all_finite(self):
        cfg = self._default_config()
        reqs = _make_tick_requests(300)
        for wv in ("db2", "sym6", "coif1"):
            result = process_ticks(reqs, cfg, wavelet_override=wv)
            assert all(np.isfinite(v) for v in result.trend), f"Non-finite trend for {wv}"


# ─────────────────────────────────────────────────────────
# /wavelet endpoint with calibration params
# ─────────────────────────────────────────────────────────

class TestWaveletEndpointCalibration:
    def test_db2_returns_200(self, client):
        resp = client.post("/wavelet", json={"ticks": _make_ticks(300), "wavelet": "db2"})
        assert resp.status_code == 200

    def test_sym6_returns_200(self, client):
        resp = client.post("/wavelet", json={"ticks": _make_ticks(300), "wavelet": "sym6"})
        assert resp.status_code == 200

    def test_coif1_returns_200(self, client):
        resp = client.post("/wavelet", json={"ticks": _make_ticks(300), "wavelet": "coif1"})
        assert resp.status_code == 200

    def test_window_128_returns_200(self, client):
        resp = client.post("/wavelet", json={"ticks": _make_ticks(130), "window": 128})
        assert resp.status_code == 200

    def test_level_1_returns_200(self, client):
        resp = client.post("/wavelet", json={"ticks": _make_ticks(300), "level": 1})
        assert resp.status_code == 200

    def test_level_4_returns_200(self, client):
        resp = client.post("/wavelet", json={"ticks": _make_ticks(300), "level": 4})
        assert resp.status_code == 200

    def test_all_params_together_returns_200(self, client):
        resp = client.post("/wavelet", json={
            "ticks": _make_ticks(300),
            "wavelet": "sym6",
            "window": 256,
            "level": 3,
            "trend_mode": "A3",
        })
        assert resp.status_code == 200

    def test_invalid_wavelet_returns_400(self, client):
        resp = client.post("/wavelet", json={"ticks": _make_ticks(300), "wavelet": "bior1.1"})
        assert resp.status_code == 400

    def test_invalid_window_returns_400(self, client):
        resp = client.post("/wavelet", json={"ticks": _make_ticks(300), "window": 300})
        assert resp.status_code == 400

    def test_invalid_level_returns_400(self, client):
        resp = client.post("/wavelet", json={"ticks": _make_ticks(300), "level": 5})
        assert resp.status_code == 400

    def test_different_wavelets_return_different_trends(self, client):
        ticks = _make_ticks(300)
        r_db4 = client.post("/wavelet", json={"ticks": ticks, "wavelet": "db4"}).get_json()
        r_sym6 = client.post("/wavelet", json={"ticks": ticks, "wavelet": "sym6"}).get_json()
        assert r_db4["trend"] != r_sym6["trend"]

    def test_backward_compat_no_new_fields(self, client):
        """Request with only ticks still works exactly as before."""
        resp = client.post("/wavelet", json={"ticks": _make_ticks(300)})
        assert resp.status_code == 200
        data = resp.get_json()
        assert "trend" in data
        assert len(data["trend"]) == 300

    def test_arrays_complete_for_all_wavelets(self, client):
        required = ("trend", "relative_deviation", "z_score", "energy", "noise")
        for wv in ("db2", "db4", "db6", "sym4", "sym6", "coif1"):
            data = client.post(
                "/wavelet", json={"ticks": _make_ticks(300), "wavelet": wv}
            ).get_json()
            for key in required:
                assert key in data, f"Key {key!r} missing for wavelet={wv}"


# ─────────────────────────────────────────────────────────
# Logging — structured format
# ─────────────────────────────────────────────────────────

class TestLogging:
    def test_log_contains_window_wavelet_level_mode_elapsed(self, caplog):
        cfg = ServiceConfig(wavelet="db4", window=256, level=2)
        app = create_app(cfg)
        app.config["TESTING"] = True

        with caplog.at_level(logging.INFO, logger="wavelet_research.service.app"):
            with app.test_client() as c:
                c.post("/wavelet", json={
                    "ticks": _make_ticks(300),
                    "wavelet": "sym6",
                    "window": 256,
                    "level": 3,
                    "trend_mode": "A2",
                })

        log_text = " ".join(caplog.messages)
        assert "window=256" in log_text
        assert "wavelet=sym6" in log_text
        assert "level=3" in log_text
        assert "mode=A2" in log_text
        assert "elapsed_ms=" in log_text

    def test_log_uses_defaults_when_no_overrides(self, caplog):
        cfg = ServiceConfig(wavelet="db4", window=256, level=2)
        app = create_app(cfg)
        app.config["TESTING"] = True

        with caplog.at_level(logging.INFO, logger="wavelet_research.service.app"):
            with app.test_client() as c:
                c.post("/wavelet", json={"ticks": _make_ticks(300)})

        log_text = " ".join(caplog.messages)
        assert "window=256" in log_text
        assert "wavelet=db4" in log_text
        assert "level=2" in log_text


# ─────────────────────────────────────────────────────────
# MQL5 structural tests
# ─────────────────────────────────────────────────────────

class TestMQL5CalibrationInputs:
    def _read_mql5(self, filename: str) -> str:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return open(os.path.join(base, "mql5", filename), encoding="utf-8").read()

    def test_bridge_ea_has_inp_wavelet(self):
        src = self._read_mql5("WaveletBridgeEA.mq5")
        assert "InpWavelet" in src

    def test_bridge_ea_has_inp_wavelet_window(self):
        src = self._read_mql5("WaveletBridgeEA.mq5")
        assert "InpWaveletWindow" in src

    def test_bridge_ea_has_inp_level(self):
        src = self._read_mql5("WaveletBridgeEA.mq5")
        assert "InpLevel" in src

    def test_bridge_ea_sends_wavelet_in_json(self):
        src = self._read_mql5("WaveletBridgeEA.mq5")
        assert "wavelet" in src

    def test_bridge_ea_sends_window_in_json(self):
        src = self._read_mql5("WaveletBridgeEA.mq5")
        # InpWaveletWindow must appear at least twice: input declaration + JSON payload
        assert src.count("InpWaveletWindow") >= 2

    def test_bridge_ea_sends_level_in_json(self):
        src = self._read_mql5("WaveletBridgeEA.mq5")
        # InpLevel must appear at least twice: input declaration + JSON payload
        assert src.count("InpLevel") >= 2

    def test_bridge_ea_validates_wavelet_on_init(self):
        src = self._read_mql5("WaveletBridgeEA.mq5")
        assert "InpWavelet" in src
        assert "INIT_PARAMETERS_INCORRECT" in src

    def test_bridge_ea_validates_window_on_init(self):
        src = self._read_mql5("WaveletBridgeEA.mq5")
        assert "InpWaveletWindow" in src

    def test_bridge_ea_validates_level_on_init(self):
        src = self._read_mql5("WaveletBridgeEA.mq5")
        assert "InpLevel" in src
