"""Tests for the Paper Trading Framework (Story 12)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wavelet_research.engine.config import WaveletEngineConfig
from wavelet_research.engine.models import Tick
from wavelet_research.mt5.expert_advisor import EAConfig, ExecutionMode
from wavelet_research.mt5.risk import RiskConfig
from wavelet_research.paper_trading.core import PaperTrader
from wavelet_research.paper_trading.journal import PaperTrade, PaperTradeJournal
from wavelet_research.paper_trading.replay import MarketReplay
from wavelet_research.signal.config import SignalConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mid = 1.10 + np.cumsum(rng.normal(0, 0.00005, n))
    spread = 0.00010
    return pd.DataFrame({
        "time": pd.date_range("2026-06-01", periods=n, freq="1s"),
        "bid": mid - spread / 2,
        "ask": mid + spread / 2,
        "mid": mid,
        "spread": np.full(n, spread),
    })


def _make_ea_config(buy_z: float = 0.5, sell_z: float = 0.5) -> EAConfig:
    return EAConfig(
        wavelet_config=WaveletEngineConfig(wavelet="haar", window=256, level=2),
        signal_config=SignalConfig(
            buy_z_threshold=buy_z, sell_z_threshold=sell_z,
            slope_filter_enabled=False,
        ),
        risk_config=RiskConfig(max_open_positions=5, max_daily_loss=1000.0),
        stop_loss_pips=50.0,
        take_profit_pips=50.0,
    )


# ---------------------------------------------------------------------------
# Market Replay Tests
# ---------------------------------------------------------------------------


class TestMarketReplay:
    def test_creates_from_data(self) -> None:
        data = _make_dataset(100)
        replay = MarketReplay(data)
        assert replay.total_ticks == 100
        assert replay.current_index == 0
        assert not replay.is_complete

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            MarketReplay(pd.DataFrame())

    def test_next_tick(self) -> None:
        data = _make_dataset(10)
        replay = MarketReplay(data)
        tick = replay.next_tick()
        assert isinstance(tick, Tick)
        assert replay.current_index == 1

    def test_iteration(self) -> None:
        data = _make_dataset(50)
        replay = MarketReplay(data)
        ticks = list(replay)
        assert len(ticks) == 50
        assert replay.is_complete

    def test_progress(self) -> None:
        data = _make_dataset(100)
        replay = MarketReplay(data)
        assert replay.progress == 0.0
        for _ in range(50):
            replay.next_tick()
        assert replay.progress == pytest.approx(0.5)

    def test_reset(self) -> None:
        data = _make_dataset(50)
        replay = MarketReplay(data)
        list(replay)
        assert replay.is_complete
        replay.reset()
        assert replay.current_index == 0
        assert not replay.is_complete

    def test_none_when_complete(self) -> None:
        data = _make_dataset(5)
        replay = MarketReplay(data)
        for _ in range(5):
            replay.next_tick()
        assert replay.next_tick() is None


# ---------------------------------------------------------------------------
# Journal Tests
# ---------------------------------------------------------------------------


class TestPaperTradeJournal:
    def test_empty_journal(self) -> None:
        j = PaperTradeJournal()
        assert j.count == 0
        assert j.total_pnl == 0.0
        assert j.win_rate == 0.0
        assert j.expectancy == 0.0

    def test_add_trade(self) -> None:
        j = PaperTradeJournal()
        t = PaperTrade(
            direction="BUY", entry_price=1.10,
            exit_price=1.1010, lots=0.01, pnl=10.0,
            entry_time=pd.Timestamp("2026-06-01"),
            exit_time=pd.Timestamp("2026-06-01 00:01"),
            exit_reason="take_profit",
        )
        j.add_trade(t)
        assert j.count == 1
        assert j.total_pnl == 10.0

    def test_metrics(self) -> None:
        j = PaperTradeJournal()
        now = pd.Timestamp("2026-06-01")
        j.add_trade(PaperTrade("BUY", 1.10, 1.1010, 0.01, 10.0, now, now, "tp"))
        j.add_trade(PaperTrade("SELL", 1.10, 1.1010, 0.01, -5.0, now, now, "sl"))
        j.add_trade(PaperTrade("BUY", 1.10, 1.1020, 0.01, 20.0, now, now, "tp"))
        assert j.count == 3
        assert j.total_pnl == 25.0
        assert j.win_rate == pytest.approx(2 / 3)
        assert j.profit_factor == pytest.approx(30.0 / 5.0)

    def test_to_dataframe(self) -> None:
        j = PaperTradeJournal()
        now = pd.Timestamp("2026-06-01")
        j.add_trade(PaperTrade("BUY", 1.10, 1.1010, 0.01, 10.0, now, now, "tp"))
        df = j.to_dataframe()
        assert len(df) == 1
        assert "pnl" in df.columns

    def test_max_drawdown(self) -> None:
        j = PaperTradeJournal()
        now = pd.Timestamp("2026-06-01")
        j.add_trade(PaperTrade("BUY", 1.10, 1.11, 0.01, 10.0, now, now, "tp"))
        j.add_trade(PaperTrade("BUY", 1.10, 1.09, 0.01, -20.0, now, now, "sl"))
        j.add_trade(PaperTrade("BUY", 1.10, 1.09, 0.01, -5.0, now, now, "sl"))
        assert j.max_drawdown == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# Paper Trader Tests
# ---------------------------------------------------------------------------


class TestPaperTrader:
    def test_runs_session(self) -> None:
        data = _make_dataset(n=500)
        cfg = _make_ea_config()
        trader = PaperTrader(cfg)
        journal = trader.run(data)
        assert isinstance(journal, PaperTradeJournal)

    def test_balance_changes(self) -> None:
        data = _make_dataset(n=500)
        cfg = _make_ea_config(buy_z=0.3, sell_z=0.3)
        trader = PaperTrader(cfg, initial_balance=10000.0)
        trader.run(data)
        # Balance should differ from initial if trades occurred
        if trader.total_trades > 0:
            assert trader.balance != 10000.0

    def test_deterministic(self) -> None:
        data = _make_dataset(n=400)
        cfg = _make_ea_config()
        j1 = PaperTrader(cfg).run(data)
        j2 = PaperTrader(cfg).run(data)
        assert j1.count == j2.count
        assert j1.total_pnl == j2.total_pnl

    def test_journal_has_trades(self) -> None:
        data = _make_dataset(n=500)
        cfg = _make_ea_config(buy_z=0.3, sell_z=0.3)
        trader = PaperTrader(cfg)
        journal = trader.run(data)
        if journal.count > 0:
            trade = journal.trades[0]
            assert trade.direction in ("BUY", "SELL")
            assert trade.exit_reason in ("stop_loss", "take_profit", "end_of_data")

    def test_validation_metrics(self) -> None:
        data = _make_dataset(n=500)
        cfg = _make_ea_config(buy_z=0.3, sell_z=0.3)
        trader = PaperTrader(cfg)
        journal = trader.run(data)
        if journal.count > 0:
            assert 0.0 <= journal.win_rate <= 1.0
            assert journal.profit_factor >= 0.0
