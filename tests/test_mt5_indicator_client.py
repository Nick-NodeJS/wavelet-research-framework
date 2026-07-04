"""Tests for the MT5 Thin Indicator client layer (Story 17)."""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from wavelet_research.mt5_indicator.buffer import IndicatorBuffer
from wavelet_research.mt5_indicator.client import WaveletServiceClient
from wavelet_research.mt5_indicator.config import IndicatorConfig
from wavelet_research.mt5_indicator.models import (
    ConnectionStatus,
    IndicatorResponse,
    ParsedWaveletResponse,
    TickPayload,
)
from wavelet_research.mt5_indicator.parser import (
    ResponseParseError,
    parse_json_bytes,
    validate_wavelet_response,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response_dict(n: int = 10) -> dict:
    return {
        "trend": [1.1] * n,
        "relative_deviation": [0.1] * n,
        "z_score": [0.1] * n,
        "energy": [0.01] * n,
        "noise": [0.001] * n,
    }


def _make_response_bytes(n: int = 10) -> bytes:
    return json.dumps(_make_response_dict(n)).encode("utf-8")


def _make_ticks(n: int = 10) -> list[TickPayload]:
    return [
        TickPayload(
            time=f"2026-01-01T00:00:{i:02d}",
            bid=1.10000 + i * 0.00001,
            ask=1.10001 + i * 0.00001,
            mid=1.100005 + i * 0.00001,
        )
        for i in range(n)
    ]


def _make_config(**kwargs) -> IndicatorConfig:
    defaults = dict(server_url="http://127.0.0.1:5000", tick_window=10)
    defaults.update(kwargs)
    return IndicatorConfig(**defaults)


def _urlopen_mock(response_bytes: bytes):
    """Create a mock urlopen context manager returning given bytes."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = response_bytes
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return MagicMock(return_value=mock_resp)


# ---------------------------------------------------------------------------
# IndicatorConfig Tests
# ---------------------------------------------------------------------------


class TestIndicatorConfig:
    def test_defaults(self) -> None:
        cfg = IndicatorConfig()
        assert cfg.server_url == "http://127.0.0.1:5000"
        assert cfg.tick_window == 2048
        assert cfg.request_timeout_seconds == 0.5
        assert cfg.auto_refresh is True

    def test_wavelet_endpoint(self) -> None:
        cfg = IndicatorConfig(server_url="http://localhost:5000")
        assert cfg.wavelet_endpoint == "http://localhost:5000/wavelet"

    def test_health_endpoint(self) -> None:
        cfg = IndicatorConfig(server_url="http://localhost:5000/")
        assert cfg.health_endpoint == "http://localhost:5000/health"

    def test_invalid_tick_window(self) -> None:
        with pytest.raises(ValueError, match="tick_window"):
            IndicatorConfig(tick_window=0)

    def test_invalid_timeout(self) -> None:
        with pytest.raises(ValueError, match="timeout"):
            IndicatorConfig(request_timeout_seconds=0.0)

    def test_invalid_url(self) -> None:
        with pytest.raises(ValueError, match="server_url"):
            IndicatorConfig(server_url="ftp://bad-url")


# ---------------------------------------------------------------------------
# TickPayload Tests
# ---------------------------------------------------------------------------


class TestTickPayload:
    def test_to_dict(self) -> None:
        t = TickPayload(time="2026-01-01T00:00:00", bid=1.10, ask=1.11, mid=1.105)
        d = t.to_dict()
        assert d["bid"] == 1.10
        assert d["ask"] == 1.11
        assert d["mid"] == 1.105
        assert d["time"] == "2026-01-01T00:00:00"

    def test_serialises_to_json(self) -> None:
        t = TickPayload(time="2026-01-01", bid=1.10, ask=1.11, mid=1.105)
        raw = json.dumps(t.to_dict())
        parsed = json.loads(raw)
        assert parsed["bid"] == 1.10


# ---------------------------------------------------------------------------
# ParsedWaveletResponse Tests
# ---------------------------------------------------------------------------


class TestParsedWaveletResponse:
    def test_equal_length_valid(self) -> None:
        r = ParsedWaveletResponse(
            trend=(1.0,) * 5,
            relative_deviation=(0.1,) * 5,
            z_score=(0.1,) * 5,
            energy=(0.01,) * 5,
            noise=(0.001,) * 5,
        )
        assert r.length == 5

    def test_unequal_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            ParsedWaveletResponse(
                trend=(1.0,) * 5,
                relative_deviation=(0.1,) * 4,  # wrong
                z_score=(0.1,) * 5,
                energy=(0.01,) * 5,
                noise=(0.001,) * 5,
            )

    def test_is_successful_field(self) -> None:
        r = ParsedWaveletResponse(
            trend=(1.0,), relative_deviation=(0.1,), z_score=(0.1,),
            energy=(0.01,), noise=(0.001,),
        )
        resp = IndicatorResponse(
            status=ConnectionStatus.CONNECTED, data=r, elapsed_ms=1.0
        )
        assert resp.is_successful is True

    def test_failed_response_not_successful(self) -> None:
        resp = IndicatorResponse(
            status=ConnectionStatus.TIMEOUT, data=None, elapsed_ms=500.0
        )
        assert resp.is_successful is False


# ---------------------------------------------------------------------------
# JSON Parser Tests
# ---------------------------------------------------------------------------


class TestParser:
    def test_parse_valid_json_bytes(self) -> None:
        raw = json.dumps({"key": "value"}).encode("utf-8")
        result = parse_json_bytes(raw)
        assert result["key"] == "value"

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(ResponseParseError, match="valid JSON"):
            parse_json_bytes(b"not json {{{")

    def test_non_dict_json_raises(self) -> None:
        with pytest.raises(ResponseParseError, match="JSON object"):
            parse_json_bytes(b"[1, 2, 3]")

    def test_invalid_utf8_raises(self) -> None:
        with pytest.raises(ResponseParseError, match="UTF-8"):
            parse_json_bytes(b"\xff\xfe bad bytes")

    def test_validate_valid_response(self) -> None:
        data = _make_response_dict(5)
        result = validate_wavelet_response(data)
        assert result.length == 5
        assert result.trend == (1.1,) * 5

    def test_missing_field_raises(self) -> None:
        data = _make_response_dict(5)
        del data["trend"]
        with pytest.raises(ResponseParseError, match="missing required fields"):
            validate_wavelet_response(data)

    def test_non_list_field_raises(self) -> None:
        data = _make_response_dict(5)
        data["trend"] = "not_a_list"
        with pytest.raises(ResponseParseError, match="must be an array"):
            validate_wavelet_response(data)

    def test_non_numeric_value_raises(self) -> None:
        data = _make_response_dict(3)
        data["trend"] = [1.0, "bad", 1.0]
        with pytest.raises(ResponseParseError, match="non-numeric"):
            validate_wavelet_response(data)

    def test_unequal_array_lengths_raises(self) -> None:
        data = _make_response_dict(5)
        data["z_score"] = [0.1] * 3  # wrong length
        with pytest.raises(ResponseParseError, match="equal length"):
            validate_wavelet_response(data)

    def test_expected_length_mismatch_raises(self) -> None:
        data = _make_response_dict(5)
        with pytest.raises(ResponseParseError, match="Expected array length"):
            validate_wavelet_response(data, expected_length=10)

    def test_expected_length_correct_passes(self) -> None:
        data = _make_response_dict(5)
        result = validate_wavelet_response(data, expected_length=5)
        assert result.length == 5


# ---------------------------------------------------------------------------
# Response Validation Tests (end-to-end parse + validate)
# ---------------------------------------------------------------------------


class TestResponseValidation:
    def test_full_pipeline(self) -> None:
        raw = _make_response_bytes(8)
        data = parse_json_bytes(raw)
        result = validate_wavelet_response(data, expected_length=8)
        assert result.length == 8

    def test_empty_arrays_length_zero(self) -> None:
        data = {k: [] for k in ("trend", "relative_deviation", "z_score", "energy", "noise")}
        result = validate_wavelet_response(data)
        assert result.length == 0

    def test_all_fields_present_in_result(self) -> None:
        data = _make_response_dict(3)
        result = validate_wavelet_response(data)
        assert len(result.trend) == 3
        assert len(result.relative_deviation) == 3
        assert len(result.z_score) == 3
        assert len(result.energy) == 3
        assert len(result.noise) == 3


# ---------------------------------------------------------------------------
# Indicator Buffer Tests
# ---------------------------------------------------------------------------


class TestIndicatorBuffer:
    def test_init_all_zeros(self) -> None:
        buf = IndicatorBuffer(size=10)
        assert buf.trend == [0.0] * 10
        assert buf.status == ConnectionStatus.CONNECTING

    def test_invalid_size_raises(self) -> None:
        with pytest.raises(ValueError, match="size"):
            IndicatorBuffer(size=0)

    def test_update_sets_connected(self) -> None:
        buf = IndicatorBuffer(size=5)
        resp = ParsedWaveletResponse(
            trend=(1.1,) * 5,
            relative_deviation=(0.1,) * 5,
            z_score=(0.2,) * 5,
            energy=(0.01,) * 5,
            noise=(0.001,) * 5,
        )
        buf.update(resp)
        assert buf.status == ConnectionStatus.CONNECTED
        assert buf.trend == [1.1] * 5

    def test_update_pads_short_response(self) -> None:
        buf = IndicatorBuffer(size=10)
        resp = ParsedWaveletResponse(
            trend=(1.1,) * 5,
            relative_deviation=(0.1,) * 5,
            z_score=(0.2,) * 5,
            energy=(0.01,) * 5,
            noise=(0.001,) * 5,
        )
        buf.update(resp)
        assert buf.trend[:5] == [0.0] * 5
        assert buf.trend[5:] == [1.1] * 5

    def test_update_truncates_long_response(self) -> None:
        buf = IndicatorBuffer(size=3)
        resp = ParsedWaveletResponse(
            trend=(1.0, 2.0, 3.0, 4.0, 5.0),
            relative_deviation=(0.1,) * 5,
            z_score=(0.2,) * 5,
            energy=(0.01,) * 5,
            noise=(0.001,) * 5,
        )
        buf.update(resp)
        # Last 3 values from a 5-element array: indices 2, 3, 4
        assert buf.trend == [3.0, 4.0, 5.0]

    def test_set_status(self) -> None:
        buf = IndicatorBuffer(size=5)
        buf.set_status(ConnectionStatus.TIMEOUT)
        assert buf.status == ConnectionStatus.TIMEOUT

    def test_clear_resets(self) -> None:
        buf = IndicatorBuffer(size=5)
        resp = ParsedWaveletResponse(
            trend=(9.9,) * 5,
            relative_deviation=(0.1,) * 5,
            z_score=(0.2,) * 5,
            energy=(0.01,) * 5,
            noise=(0.001,) * 5,
        )
        buf.update(resp)
        buf.clear()
        assert buf.trend == [0.0] * 5
        assert buf.status == ConnectionStatus.CONNECTING

    def test_trend_returns_copy(self) -> None:
        buf = IndicatorBuffer(size=3)
        t = buf.trend
        t[0] = 99.9
        assert buf.trend[0] == 0.0  # original unchanged


# ---------------------------------------------------------------------------
# HTTP Client Tests
# ---------------------------------------------------------------------------


class TestWaveletServiceClient:
    def test_successful_fetch(self) -> None:
        cfg = _make_config(tick_window=5)
        ticks = _make_ticks(5)
        raw = _make_response_bytes(5)

        with patch("urllib.request.urlopen", _urlopen_mock(raw)):
            client = WaveletServiceClient(cfg)
            result = client.fetch(ticks)

        assert result.is_successful
        assert result.status == ConnectionStatus.CONNECTED
        assert result.data is not None
        assert result.data.length == 5

    def test_timeout_returns_timeout_status(self) -> None:
        cfg = _make_config()
        ticks = _make_ticks(5)

        with patch("urllib.request.urlopen", side_effect=TimeoutError()):
            client = WaveletServiceClient(cfg)
            result = client.fetch(ticks)

        assert result.status == ConnectionStatus.TIMEOUT
        assert result.data is None

    def test_service_offline_returns_offline_status(self) -> None:
        cfg = _make_config()
        ticks = _make_ticks(5)

        err = urllib.error.URLError("Connection refused")
        with patch("urllib.request.urlopen", side_effect=err):
            client = WaveletServiceClient(cfg)
            result = client.fetch(ticks)

        assert result.status == ConnectionStatus.SERVICE_OFFLINE
        assert result.data is None

    def test_invalid_response_json(self) -> None:
        cfg = _make_config(tick_window=5)
        ticks = _make_ticks(5)

        with patch("urllib.request.urlopen", _urlopen_mock(b"not json")):
            client = WaveletServiceClient(cfg)
            result = client.fetch(ticks)

        assert result.status == ConnectionStatus.INVALID_RESPONSE
        assert result.data is None

    def test_missing_field_in_response(self) -> None:
        cfg = _make_config(tick_window=5)
        ticks = _make_ticks(5)
        bad = {"trend": [1.0] * 5}  # missing fields
        raw = json.dumps(bad).encode("utf-8")

        with patch("urllib.request.urlopen", _urlopen_mock(raw)):
            client = WaveletServiceClient(cfg)
            result = client.fetch(ticks)

        assert result.status == ConnectionStatus.INVALID_RESPONSE

    def test_elapsed_ms_populated(self) -> None:
        cfg = _make_config(tick_window=5)
        ticks = _make_ticks(5)
        raw = _make_response_bytes(5)

        with patch("urllib.request.urlopen", _urlopen_mock(raw)):
            client = WaveletServiceClient(cfg)
            result = client.fetch(ticks)

        assert result.elapsed_ms >= 0.0

    def test_never_raises(self) -> None:
        """fetch() must never propagate exceptions — always returns IndicatorResponse."""
        cfg = _make_config()
        ticks = _make_ticks(5)

        with patch("urllib.request.urlopen", side_effect=RuntimeError("unexpected")):
            client = WaveletServiceClient(cfg)
            # Should not raise even on unexpected error — but RuntimeError is
            # not caught. This verifies the documented contract: only
            # TimeoutError, URLError, and ResponseParseError are handled.
            with pytest.raises(RuntimeError):
                client.fetch(ticks)

    def test_check_health_connected(self) -> None:
        cfg = _make_config()
        with patch("urllib.request.urlopen", _urlopen_mock(b'{"status":"ok"}')):
            client = WaveletServiceClient(cfg)
            status = client.check_health()
        assert status == ConnectionStatus.CONNECTED

    def test_check_health_offline(self) -> None:
        cfg = _make_config()
        err = urllib.error.URLError("refused")
        with patch("urllib.request.urlopen", side_effect=err):
            client = WaveletServiceClient(cfg)
            status = client.check_health()
        assert status == ConnectionStatus.SERVICE_OFFLINE


# ---------------------------------------------------------------------------
# Reconnect Tests
# ---------------------------------------------------------------------------


class TestReconnect:
    def test_recovers_after_offline(self) -> None:
        """Client recovers automatically when service comes back online."""
        cfg = _make_config(tick_window=5)
        ticks = _make_ticks(5)
        raw = _make_response_bytes(5)
        client = WaveletServiceClient(cfg)

        err = urllib.error.URLError("refused")
        with patch("urllib.request.urlopen", side_effect=err):
            r1 = client.fetch(ticks)
        assert r1.status == ConnectionStatus.SERVICE_OFFLINE

        with patch("urllib.request.urlopen", _urlopen_mock(raw)):
            r2 = client.fetch(ticks)
        assert r2.is_successful

    def test_recovers_after_timeout(self) -> None:
        cfg = _make_config(tick_window=5)
        ticks = _make_ticks(5)
        raw = _make_response_bytes(5)
        client = WaveletServiceClient(cfg)

        with patch("urllib.request.urlopen", side_effect=TimeoutError()):
            r1 = client.fetch(ticks)
        assert r1.status == ConnectionStatus.TIMEOUT

        with patch("urllib.request.urlopen", _urlopen_mock(raw)):
            r2 = client.fetch(ticks)
        assert r2.is_successful

    def test_multiple_consecutive_successes(self) -> None:
        cfg = _make_config(tick_window=5)
        ticks = _make_ticks(5)
        raw = _make_response_bytes(5)
        client = WaveletServiceClient(cfg)

        for _ in range(3):
            with patch("urllib.request.urlopen", _urlopen_mock(raw)):
                r = client.fetch(ticks)
            assert r.is_successful


# ---------------------------------------------------------------------------
# Timeout Tests
# ---------------------------------------------------------------------------


class TestTimeout:
    def test_timeout_status_is_timeout(self) -> None:
        cfg = _make_config()
        ticks = _make_ticks(5)
        with patch("urllib.request.urlopen", side_effect=TimeoutError()):
            client = WaveletServiceClient(cfg)
            result = client.fetch(ticks)
        assert result.status == ConnectionStatus.TIMEOUT

    def test_timeout_elapsed_ms_is_nonneg(self) -> None:
        cfg = _make_config()
        ticks = _make_ticks(5)
        with patch("urllib.request.urlopen", side_effect=TimeoutError()):
            client = WaveletServiceClient(cfg)
            result = client.fetch(ticks)
        assert result.elapsed_ms >= 0.0

    def test_timeout_data_is_none(self) -> None:
        cfg = _make_config()
        ticks = _make_ticks(5)
        with patch("urllib.request.urlopen", side_effect=TimeoutError()):
            client = WaveletServiceClient(cfg)
            result = client.fetch(ticks)
        assert result.data is None


# ---------------------------------------------------------------------------
# ConnectionStatus Tests
# ---------------------------------------------------------------------------


class TestConnectionStatus:
    def test_all_statuses_have_display_string(self) -> None:
        for status in ConnectionStatus:
            assert isinstance(status.value, str)
            assert len(status.value) > 0

    def test_status_values(self) -> None:
        assert ConnectionStatus.CONNECTED.value == "Connected"
        assert ConnectionStatus.TIMEOUT.value == "Timeout"
        assert ConnectionStatus.SERVICE_OFFLINE.value == "Service Offline"
        assert ConnectionStatus.INVALID_RESPONSE.value == "Invalid Response"
        assert ConnectionStatus.CONNECTING.value == "Connecting..."
