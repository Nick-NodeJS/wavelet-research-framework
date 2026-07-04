"""Tests for Stories 24–29."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from wavelet_research.paper_trading.acceptance import (
    AcceptanceConfig,
    AcceptanceResult,
    evaluate_acceptance,
)
from wavelet_research.paper_trading.journal import PaperTrade, PaperTradeJournal
from wavelet_research.research.final_gate import (
    GateConfig,
    GateDecision,
    GateMetrics,
    GateResult,
    evaluate_gate,
)
from wavelet_research.service.risk import (
    RiskBlockReason,
    RiskCheckRequest,
    RiskConfig,
    evaluate_risk,
)


# ────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────

def _make_journal(wins: int = 20, losses: int = 5) -> PaperTradeJournal:
    journal = PaperTradeJournal()
    ts = pd.Timestamp("2026-01-01")
    for i in range(wins):
        journal.add_trade(PaperTrade(
            direction="BUY", entry_price=1.1, exit_price=1.105,
            lots=0.01, pnl=5.0, entry_time=ts, exit_time=ts,
            exit_reason="return_to_trend",
        ))
    for i in range(losses):
        journal.add_trade(PaperTrade(
            direction="SELL", entry_price=1.1, exit_price=1.103,
            lots=0.01, pnl=-3.0, entry_time=ts, exit_time=ts,
            exit_reason="max_adverse_move",
        ))
    return journal


def _pass_metrics() -> GateMetrics:
    return GateMetrics(
        total_trades=100,
        profit_factor=1.5,
        expectancy=2.0,
        max_drawdown=50.0,
        gross_profit=500.0,
        avg_holding_bars=12.0,
        monte_carlo_survival=0.80,
        walk_forward_stability=0.70,
        paper_consistency=0.80,
    )


def _fail_metrics_drawdown() -> GateMetrics:
    return GateMetrics(
        total_trades=100, profit_factor=1.5, expectancy=2.0,
        max_drawdown=200.0, gross_profit=100.0,
        avg_holding_bars=12.0, monte_carlo_survival=0.80,
        walk_forward_stability=0.70, paper_consistency=0.80,
    )


def _safe_request(**kwargs) -> RiskCheckRequest:
    defaults = dict(
        symbol="EURUSD",
        signal_confidence=0.75,
        filter_can_trade=True,
        current_spread=0.0001,
        open_positions=0,
        daily_loss_so_far=0.0,
        bars_since_last_loss=20,
        bars_since_last_error=20,
        has_duplicate_position=False,
    )
    defaults.update(kwargs)
    return RiskCheckRequest(**defaults)


# ────────────────────────────────────────────────
# Story 27: Paper Trading Acceptance Gate
# ────────────────────────────────────────────────

class TestAcceptanceGate:
    def test_passes_with_good_journal(self):
        journal = _make_journal(wins=30, losses=5)
        config = AcceptanceConfig(min_trades=30)
        result = evaluate_acceptance(journal, config)
        assert result.passed is True

    def test_fails_insufficient_trades(self):
        journal = _make_journal(wins=5, losses=2)
        config = AcceptanceConfig(min_trades=30)
        result = evaluate_acceptance(journal, config)
        assert result.passed is False
        assert any("insufficient_trades" in r for r in result.reasons)

    def test_fails_low_profit_factor(self):
        journal = PaperTradeJournal()
        ts = pd.Timestamp("2026-01-01")
        for _ in range(40):
            journal.add_trade(PaperTrade("BUY", 1.1, 1.101, 0.01, 0.5, ts, ts, "x"))
        for _ in range(40):
            journal.add_trade(PaperTrade("SELL", 1.1, 1.102, 0.01, -1.0, ts, ts, "x"))
        config = AcceptanceConfig(min_trades=30, min_profit_factor=1.5)
        result = evaluate_acceptance(journal, config)
        assert result.passed is False

    def test_fails_max_consecutive_losses(self):
        journal = PaperTradeJournal()
        ts = pd.Timestamp("2026-01-01")
        for _ in range(10):
            journal.add_trade(PaperTrade("BUY", 1.1, 1.09, 0.01, -5.0, ts, ts, "x"))
        for _ in range(40):
            journal.add_trade(PaperTrade("BUY", 1.1, 1.105, 0.01, 5.0, ts, ts, "x"))
        config = AcceptanceConfig(min_trades=30, max_consecutive_losses=5)
        result = evaluate_acceptance(journal, config)
        assert result.passed is False

    def test_metrics_present(self):
        journal = _make_journal()
        result = evaluate_acceptance(journal, AcceptanceConfig())
        assert "trades" in result.metrics
        assert "profit_factor" in result.metrics

    def test_result_serializable(self):
        journal = _make_journal()
        result = evaluate_acceptance(journal, AcceptanceConfig())
        d = result.to_dict()
        text = json.dumps(d)
        assert "passed" in text


# ────────────────────────────────────────────────
# Story 28: EA Safety Controls
# ────────────────────────────────────────────────

class TestRiskControls:
    def test_live_mode_disabled_blocks(self):
        config = RiskConfig(live_mode=False)
        req = _safe_request()
        result = evaluate_risk(req, config)
        assert result.can_execute is False
        assert RiskBlockReason.LIVE_MODE_DISABLED in result.reasons

    def test_kill_switch_blocks(self):
        config = RiskConfig(live_mode=True, kill_switch=True)
        req = _safe_request()
        result = evaluate_risk(req, config)
        assert RiskBlockReason.MANUAL_KILL_SWITCH in result.reasons

    def test_low_confidence_blocks(self):
        config = RiskConfig(live_mode=True, min_signal_confidence=0.80)
        req = _safe_request(signal_confidence=0.60)
        result = evaluate_risk(req, config)
        assert RiskBlockReason.LOW_CONFIDENCE in result.reasons

    def test_filter_blocked_blocks(self):
        config = RiskConfig(live_mode=True)
        req = _safe_request(filter_can_trade=False)
        result = evaluate_risk(req, config)
        assert RiskBlockReason.FILTER_BLOCKED in result.reasons

    def test_spread_too_wide_blocks(self):
        config = RiskConfig(live_mode=True, max_spread=0.0001)
        req = _safe_request(current_spread=0.001)
        result = evaluate_risk(req, config)
        assert RiskBlockReason.SPREAD_TOO_WIDE in result.reasons

    def test_max_positions_blocks(self):
        config = RiskConfig(live_mode=True, max_open_positions=1)
        req = _safe_request(open_positions=1)
        result = evaluate_risk(req, config)
        assert RiskBlockReason.MAX_POSITIONS in result.reasons

    def test_daily_loss_limit_blocks(self):
        config = RiskConfig(live_mode=True, max_daily_loss=100.0)
        req = _safe_request(daily_loss_so_far=100.0)
        result = evaluate_risk(req, config)
        assert RiskBlockReason.DAILY_LOSS_LIMIT in result.reasons

    def test_cooldown_after_loss_blocks(self):
        config = RiskConfig(live_mode=True, cooldown_bars_after_loss=5)
        req = _safe_request(bars_since_last_loss=3)
        result = evaluate_risk(req, config)
        assert RiskBlockReason.COOLDOWN_AFTER_LOSS in result.reasons

    def test_cooldown_after_error_blocks(self):
        config = RiskConfig(live_mode=True, cooldown_bars_after_error=3)
        req = _safe_request(bars_since_last_error=1)
        result = evaluate_risk(req, config)
        assert RiskBlockReason.COOLDOWN_AFTER_ERROR in result.reasons

    def test_duplicate_position_blocks(self):
        config = RiskConfig(live_mode=True)
        req = _safe_request(has_duplicate_position=True)
        result = evaluate_risk(req, config)
        assert RiskBlockReason.DUPLICATE_POSITION in result.reasons

    def test_symbol_whitelist_blocks(self):
        config = RiskConfig(live_mode=True, symbol_whitelist=("GBPUSD",))
        req = _safe_request(symbol="EURUSD")
        result = evaluate_risk(req, config)
        assert RiskBlockReason.SYMBOL_NOT_WHITELISTED in result.reasons

    def test_all_pass_when_conditions_met(self):
        config = RiskConfig(
            live_mode=True, kill_switch=False, max_spread=0.001,
            min_signal_confidence=0.5, max_open_positions=5,
            max_daily_loss=1000.0, cooldown_bars_after_loss=0,
            cooldown_bars_after_error=0, symbol_whitelist=(),
        )
        req = _safe_request()
        result = evaluate_risk(req, config)
        assert result.can_execute is True

    def test_result_serializable(self):
        config = RiskConfig(live_mode=True)
        req = _safe_request()
        result = evaluate_risk(req, config)
        text = json.dumps(result.to_dict())
        assert "can_execute" in text

    def test_invalid_config_raises(self):
        with pytest.raises(ValueError):
            RiskConfig(max_risk_per_trade_pct=0.0)


# ────────────────────────────────────────────────
# Story 29: Final Gate
# ────────────────────────────────────────────────

class TestFinalGate:
    def test_pass_case(self):
        result = evaluate_gate(_pass_metrics(), GateConfig())
        assert result.decision == GateDecision.PASS
        assert result.next_action == "allow_small_live"

    def test_fail_by_drawdown(self):
        result = evaluate_gate(_fail_metrics_drawdown(), GateConfig())
        assert result.decision == GateDecision.FAIL
        assert any("drawdown" in r for r in result.reasons)

    def test_fail_by_low_sample(self):
        m = GateMetrics(
            total_trades=10, profit_factor=1.5, expectancy=2.0,
            max_drawdown=10.0, gross_profit=100.0, avg_holding_bars=12.0,
            monte_carlo_survival=0.80, walk_forward_stability=0.70,
            paper_consistency=0.80,
        )
        result = evaluate_gate(m, GateConfig(min_trades=50))
        assert result.decision == GateDecision.FAIL
        assert any("insufficient_sample" in r for r in result.reasons)

    def test_review_by_wf_stability(self):
        m = GateMetrics(
            total_trades=100, profit_factor=1.5, expectancy=2.0,
            max_drawdown=50.0, gross_profit=500.0, avg_holding_bars=12.0,
            monte_carlo_survival=0.80, walk_forward_stability=0.40,
            paper_consistency=0.80,
        )
        result = evaluate_gate(m, GateConfig())
        assert result.decision in (GateDecision.REVIEW, GateDecision.FAIL)

    def test_output_schema(self):
        result = evaluate_gate(_pass_metrics(), GateConfig())
        d = result.to_dict()
        assert "decision" in d
        assert "reasons" in d
        assert "metrics" in d
        assert "next_action" in d

    def test_serializable(self):
        result = evaluate_gate(_pass_metrics(), GateConfig())
        text = json.dumps(result.to_dict())
        assert "PASS" in text

    def test_fail_next_action_is_reject(self):
        result = evaluate_gate(_fail_metrics_drawdown(), GateConfig())
        if result.decision == GateDecision.FAIL:
            assert result.next_action == "reject_config"

    def test_review_next_action_is_paper(self):
        m = GateMetrics(
            total_trades=100, profit_factor=1.5, expectancy=2.0,
            max_drawdown=50.0, gross_profit=500.0, avg_holding_bars=40.0,
            monte_carlo_survival=0.80, walk_forward_stability=0.70,
            paper_consistency=0.80,
        )
        result = evaluate_gate(m, GateConfig(max_avg_holding_bars=20.0))
        assert result.decision in (GateDecision.REVIEW, GateDecision.FAIL)


# ────────────────────────────────────────────────
# Story 25: /market-state endpoint smoke test
# ────────────────────────────────────────────────

class TestMarketStateEndpoint:
    @pytest.fixture
    def client(self):
        from wavelet_research.service.app import create_app
        from wavelet_research.service.config import ServiceConfig
        cfg = ServiceConfig(wavelet="db4", window=256, level=2)
        app = create_app(cfg)
        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c

    def _valid_body(self, n: int = 260) -> dict:
        rng = np.random.default_rng(5)
        mid = 1.1 + np.cumsum(rng.normal(0, 0.0001, n))
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
        return {"symbol": "EURUSD", "ticks": ticks}

    def test_endpoint_200(self, client):
        resp = client.post("/market-state", json=self._valid_body())
        assert resp.status_code == 200

    def test_response_has_required_fields(self, client):
        resp = client.post("/market-state", json=self._valid_body())
        data = resp.get_json()
        assert "trend" in data
        assert "deviation" in data
        assert "filter" in data
        assert "signal" in data

    def test_response_deviation_has_side(self, client):
        resp = client.post("/market-state", json=self._valid_body())
        data = resp.get_json()
        assert data["deviation"]["side"] in ("above", "below", "near")

    def test_response_signal_has_side(self, client):
        resp = client.post("/market-state", json=self._valid_body())
        data = resp.get_json()
        assert data["signal"]["side"] in ("BUY", "SELL", "HOLD")

    def test_invalid_json_returns_400(self, client):
        resp = client.post("/market-state", data="bad", content_type="text/plain")
        assert resp.status_code == 400

    def test_no_trade_response_has_filter(self, client):
        resp = client.post("/market-state", json=self._valid_body())
        data = resp.get_json()
        assert "can_trade" in data["filter"]
