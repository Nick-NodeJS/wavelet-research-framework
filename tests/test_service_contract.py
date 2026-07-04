"""Contract tests, smoke tests and regression tests for the Wavelet Service.

Covers:
- Issue 3: timestamp precision (ms integer from MT5 round-trips correctly)
- Issue 6: frozen HTTP contract schema validation
- Issue 7: service <-> MT5 contract integration tests
- Issue 8: smoke tests (startup, /health, /wavelet, errors, config)
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import pytest

from wavelet_research.service.app import create_app
from wavelet_research.service.config import ServiceConfig
from wavelet_research.service.contract import (
    CONTRACT_VERSION,
    SUPPORTED_WAVELET,
    WAVELET_RESPONSE_FIELDS,
    validate_health_contract,
    validate_request_contract,
    validate_response_contract,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONFIG = ServiceConfig(wavelet="db4", window=512, level=2, volatility_window=256)


def _make_client(config: ServiceConfig = _CONFIG):
    app = create_app(config)
    app.config["TESTING"] = True
    return app.test_client()


def _make_ticks(n: int, seed: int = 42, use_ms_timestamp: bool = False) -> list[dict]:
    """Generate synthetic tick dicts.

    Parameters
    ----------
    use_ms_timestamp : bool
        If True, use integer ms timestamps (as MT5 sends them).
        If False, use ISO string timestamps.
    """
    rng = np.random.default_rng(seed)
    mid = 1.10 + np.cumsum(rng.normal(0, 0.00002, n))
    spread = 0.00010
    base_ms = 1_751_670_000_000  # 2025-07-04 00:00:00 UTC in ms
    rows = []
    for i in range(n):
        bid = float(mid[i] - spread / 2)
        ask = float(mid[i] + spread / 2)
        if use_ms_timestamp:
            ts = str(base_ms + i * 100)  # 100ms apart
        else:
            ts = str(pd.Timestamp("2026-06-01") + pd.Timedelta(milliseconds=i * 100))
        rows.append({"time": ts, "bid": bid, "ask": ask, "mid": float(mid[i])})
    return rows


# ---------------------------------------------------------------------------
# Issue 6: Contract Schema Tests
# ---------------------------------------------------------------------------


class TestRequestContract:
    def test_valid_request_no_violations(self) -> None:
        body = {"ticks": [{"bid": 1.10, "ask": 1.11, "mid": 1.105}]}
        assert validate_request_contract(body) == []

    def test_missing_ticks_field(self) -> None:
        violations = validate_request_contract({"data": []})
        assert any("ticks" in v for v in violations)

    def test_non_dict_body(self) -> None:
        violations = validate_request_contract([1, 2, 3])
        assert len(violations) > 0

    def test_ticks_not_list(self) -> None:
        violations = validate_request_contract({"ticks": "not_a_list"})
        assert any("array" in v for v in violations)

    def test_tick_missing_bid(self) -> None:
        body = {"ticks": [{"ask": 1.11}]}
        violations = validate_request_contract(body)
        assert any("bid" in v for v in violations)

    def test_tick_missing_ask(self) -> None:
        body = {"ticks": [{"bid": 1.10}]}
        violations = validate_request_contract(body)
        assert any("ask" in v for v in violations)

    def test_tick_not_dict(self) -> None:
        body = {"ticks": ["not_a_dict"]}
        violations = validate_request_contract(body)
        assert len(violations) > 0

    def test_optional_time_and_mid_allowed(self) -> None:
        body = {"ticks": [{"bid": 1.10, "ask": 1.11, "time": "2026-01-01", "mid": 1.105}]}
        assert validate_request_contract(body) == []

    def test_multiple_ticks_all_valid(self) -> None:
        ticks = [{"bid": 1.10 + i * 0.0001, "ask": 1.11 + i * 0.0001} for i in range(5)]
        assert validate_request_contract({"ticks": ticks}) == []


class TestResponseContract:
    def test_valid_response_no_violations(self) -> None:
        body = {f: [1.0, 2.0] for f in WAVELET_RESPONSE_FIELDS}
        assert validate_response_contract(body) == []

    def test_missing_field(self) -> None:
        body = {f: [1.0] for f in WAVELET_RESPONSE_FIELDS if f != "trend"}
        violations = validate_response_contract(body)
        assert any("trend" in v for v in violations)

    def test_field_not_array(self) -> None:
        body = {f: [1.0] for f in WAVELET_RESPONSE_FIELDS}
        body["trend"] = "not_a_list"
        violations = validate_response_contract(body)
        assert any("array" in v for v in violations)

    def test_unequal_array_lengths(self) -> None:
        body = {f: [1.0] for f in WAVELET_RESPONSE_FIELDS}
        body["z_score"] = [1.0, 2.0]  # wrong length
        violations = validate_response_contract(body)
        assert any("unequal" in v.lower() for v in violations)

    def test_non_dict_body(self) -> None:
        violations = validate_response_contract([1, 2, 3])
        assert len(violations) > 0


class TestHealthContract:
    def test_valid_health_no_violations(self) -> None:
        body = {"status": "ok", "wavelet": "db4", "version": "0.2.0"}
        assert validate_health_contract(body) == []

    def test_wrong_status(self) -> None:
        body = {"status": "error", "wavelet": "db4", "version": "0.2.0"}
        violations = validate_health_contract(body)
        assert any("ok" in v for v in violations)

    def test_wrong_wavelet(self) -> None:
        body = {"status": "ok", "wavelet": "haar", "version": "0.2.0"}
        violations = validate_health_contract(body)
        assert any("db4" in v for v in violations)

    def test_missing_fields(self) -> None:
        violations = validate_health_contract({"status": "ok"})
        assert len(violations) > 0


# ---------------------------------------------------------------------------
# Issue 7: Contract Integration Tests (Python Service <-> MT5 contract)
# ---------------------------------------------------------------------------


class TestContractIntegration:
    def test_health_response_passes_contract(self) -> None:
        client = _make_client()
        resp = client.get("/health")
        body = resp.get_json()
        violations = validate_health_contract(body)
        assert violations == [], f"Contract violations: {violations}"

    def test_wavelet_response_passes_contract(self) -> None:
        client = _make_client()
        ticks = _make_ticks(600)
        resp = client.post("/wavelet", json={"ticks": ticks})
        assert resp.status_code == 200
        body = resp.get_json()
        violations = validate_response_contract(body)
        assert violations == [], f"Contract violations: {violations}"

    def test_response_array_length_equals_request_length(self) -> None:
        client = _make_client()
        n = 600
        ticks = _make_ticks(n)
        resp = client.post("/wavelet", json={"ticks": ticks})
        body = resp.get_json()
        for field in WAVELET_RESPONSE_FIELDS:
            assert len(body[field]) == n, f"{field} length mismatch"

    def test_ms_timestamp_accepted(self) -> None:
        """MT5 sends integer ms timestamps — service must accept them."""
        client = _make_client()
        ticks = _make_ticks(600, use_ms_timestamp=True)
        resp = client.post("/wavelet", json={"ticks": ticks})
        assert resp.status_code == 200

    def test_iso_timestamp_accepted(self) -> None:
        """ISO string timestamps must also be accepted."""
        client = _make_client()
        ticks = _make_ticks(600, use_ms_timestamp=False)
        resp = client.post("/wavelet", json={"ticks": ticks})
        assert resp.status_code == 200

    def test_deterministic_with_ms_timestamps(self) -> None:
        """Same ms timestamps must produce identical responses."""
        client = _make_client()
        ticks = _make_ticks(600, use_ms_timestamp=True)
        r1 = client.post("/wavelet", json={"ticks": ticks}).get_json()
        r2 = client.post("/wavelet", json={"ticks": ticks}).get_json()
        assert r1["trend"] == r2["trend"]

    def test_floating_point_precision(self) -> None:
        """Response values must be finite floats (no inf, no nan)."""
        client = _make_client()
        ticks = _make_ticks(600)
        body = client.post("/wavelet", json={"ticks": ticks}).get_json()
        for field in WAVELET_RESPONSE_FIELDS:
            for val in body[field]:
                assert isinstance(val, float), f"{field} contains non-float"
                assert val == val, f"{field} contains NaN"  # NaN != NaN
                assert abs(val) < 1e15, f"{field} contains inf-like value"

    def test_malformed_array_rejected(self) -> None:
        """Response with unequal arrays must fail contract validation."""
        bad_body = {f: [1.0] * 5 for f in WAVELET_RESPONSE_FIELDS}
        bad_body["z_score"] = [1.0] * 3
        violations = validate_response_contract(bad_body)
        assert violations != []

    def test_wavelet_field_always_db4(self) -> None:
        client = _make_client()
        body = client.get("/health").get_json()
        assert body["wavelet"] == SUPPORTED_WAVELET


# ---------------------------------------------------------------------------
# Issue 3: Timestamp Precision Regression Tests
# ---------------------------------------------------------------------------


class TestTimestampPrecision:
    def test_ms_timestamp_round_trip_deterministic(self) -> None:
        """ms-integer timestamps produce the same result as ISO timestamps
        when mid prices are identical — the timestamp is not used in wavelet math."""
        client = _make_client()
        rng = np.random.default_rng(0)
        mid = 1.10 + np.cumsum(rng.normal(0, 0.00002, 600))
        spread = 0.0001
        base_ms = 1_751_670_000_000

        ticks_ms = [
            {
                "time": str(base_ms + i * 100),
                "bid": float(mid[i] - spread / 2),
                "ask": float(mid[i] + spread / 2),
                "mid": float(mid[i]),
            }
            for i in range(600)
        ]
        ticks_iso = [
            {
                "time": str(pd.Timestamp("2026-06-01") + pd.Timedelta(milliseconds=i * 100)),
                "bid": t["bid"],
                "ask": t["ask"],
                "mid": t["mid"],
            }
            for i, t in enumerate(ticks_ms)
        ]

        r_ms = client.post("/wavelet", json={"ticks": ticks_ms}).get_json()
        r_iso = client.post("/wavelet", json={"ticks": ticks_iso}).get_json()

        # Trend depends only on mid price, not timestamp
        assert r_ms["trend"] == r_iso["trend"]

    def test_ms_timestamp_preserved_in_request(self) -> None:
        """TickRequest stores the raw time string unchanged."""
        from wavelet_research.service.validation import parse_tick
        raw = {"bid": 1.10, "ask": 1.11, "time": "1751670000000"}
        tick = parse_tick(raw, 0)
        assert tick.time == "1751670000000"

    def test_iso_timestamp_preserved_in_request(self) -> None:
        from wavelet_research.service.validation import parse_tick
        raw = {"bid": 1.10, "ask": 1.11, "time": "2026-07-04T12:00:00.123"}
        tick = parse_tick(raw, 0)
        assert tick.time == "2026-07-04T12:00:00.123"

    def test_missing_timestamp_accepted(self) -> None:
        """Timestamp is optional — ticks without time field are valid."""
        from wavelet_research.service.validation import parse_tick
        raw = {"bid": 1.10, "ask": 1.11}
        tick = parse_tick(raw, 0)
        assert tick.time == ""


# ---------------------------------------------------------------------------
# Issue 8: Smoke Tests
# ---------------------------------------------------------------------------


class TestServiceSmoke:
    def test_app_creates_without_error(self) -> None:
        app = create_app(_CONFIG)
        assert app is not None

    def test_health_endpoint_200(self) -> None:
        client = _make_client()
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_returns_json(self) -> None:
        client = _make_client()
        resp = client.get("/health")
        assert resp.content_type.startswith("application/json")
        data = resp.get_json()
        assert isinstance(data, dict)

    def test_wavelet_endpoint_200_with_valid_input(self) -> None:
        client = _make_client()
        ticks = _make_ticks(600)
        resp = client.post("/wavelet", json={"ticks": ticks})
        assert resp.status_code == 200

    def test_wavelet_malformed_json_400(self) -> None:
        client = _make_client()
        resp = client.post("/wavelet", data="bad json", content_type="application/json")
        assert resp.status_code == 400

    def test_wavelet_missing_ticks_400(self) -> None:
        client = _make_client()
        resp = client.post("/wavelet", json={})
        assert resp.status_code == 400

    def test_wavelet_empty_ticks_400(self) -> None:
        client = _make_client()
        resp = client.post("/wavelet", json={"ticks": []})
        assert resp.status_code == 400

    def test_wavelet_insufficient_ticks_422(self) -> None:
        client = _make_client()
        ticks = _make_ticks(5)  # less than window=512
        resp = client.post("/wavelet", json={"ticks": ticks})
        assert resp.status_code == 422

    def test_wavelet_invalid_price_400(self) -> None:
        client = _make_client()
        bad_ticks = [{"bid": -1.0, "ask": 1.10}]
        resp = client.post("/wavelet", json={"ticks": bad_ticks})
        assert resp.status_code == 400

    def test_unsupported_wavelet_config_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsupported wavelet"):
            ServiceConfig(wavelet="bior1.1")

    def test_error_response_contains_error_field(self) -> None:
        client = _make_client()
        resp = client.post("/wavelet", json={})
        data = resp.get_json()
        assert "error" in data

    def test_unknown_route_404(self) -> None:
        client = _make_client()
        resp = client.get("/unknown")
        assert resp.status_code == 404
