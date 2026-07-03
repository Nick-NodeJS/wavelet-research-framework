"""Tests for the MT5 Expert Advisor (Story 11)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wavelet_research.engine.config import WaveletEngineConfig
from wavelet_research.engine.models import Tick
from wavelet_research.mt5.expert_advisor import (
    EAConfig,
    ExecutionMode,
    MT5ExpertAdvisor,
    OrderRequest,
)
from wavelet_research.mt5.risk import RiskConfig, compute_position_size
from wavelet_research.signal.config import SignalConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tick(mid: float, index: int = 0) -> Tick:
    return Tick(
        time=pd.Timestamp("2026-06-01") + pd.Timedelta(seconds=index),
        bid=mid - 0.00005,
        ask=mid + 0.00005,
        mid=mid,
        spread=0.00010,
    )


def _make_ticks(n: int = 400, seed: int = 42) -> list[Tick]:
    rng = np.random.default_rng(seed)
    mid_values = 1.10 + np.cumsum(rng.normal(0, 0.00002, n))
    return [_make_tick(float(m), i) for i, m in enumerate(mid_values)]


def _make_ea_config(
    buy_z: float = 1.0, sell_z: float = 1.0,
) -> EAConfig:
    return EAConfig(
        wavelet_config=WaveletEngineConfig(wavelet="haar", window=256, level=2),
        signal_config=SignalConfig(
            buy_z_threshold=buy_z, sell_z_threshold=sell_z,
            slope_filter_enabled=False,
        ),
        risk_config=RiskConfig(max_open_positions=3),
    )


# ---------------------------------------------------------------------------
# Risk Config Tests
# ---------------------------------------------------------------------------


class TestRiskConfig:
    def test_valid(self) -> None:
        cfg = RiskConfig()
        assert cfg.max_risk_per_trade == 0.02

    def test_invalid_risk(self) -> None:
        with pytest.raises(ValueError, match="max_risk_per_trade"):
            RiskConfig(max_risk_per_trade=0.0)

    def test_invalid_positions(self) -> None:
        with pytest.raises(ValueError, match="max_open_positions"):
            RiskConfig(max_open_positions=0)

    def test_invalid_daily_loss(self) -> None:
        with pytest.raises(ValueError, match="max_daily_loss"):
            RiskConfig(max_daily_loss=-1.0)

    def test_invalid_lot_size(self) -> None:
        with pytest.raises(ValueError, match="lot_size"):
            RiskConfig(lot_size=0.0)


class TestPositionSizing:
    def test_compute_size(self) -> None:
        lots = compute_position_size(
            balance=10000.0, risk_per_trade=0.02,
            stop_loss_pips=20.0, pip_value=10.0,
        )
        assert lots == 1.0

    def test_zero_stop_loss(self) -> None:
        assert compute_position_size(10000.0, 0.02, 0.0, 10.0) == 0.0

    def test_minimum_lot(self) -> None:
        lots = compute_position_size(100.0, 0.01, 1000.0, 10.0)
        assert lots >= 0.01


# ---------------------------------------------------------------------------
# EA Config Tests
# ---------------------------------------------------------------------------


class TestEAConfig:
    def test_valid(self) -> None:
        cfg = _make_ea_config()
        assert cfg.mode == ExecutionMode.PAPER

    def test_invalid_stop_loss(self) -> None:
        with pytest.raises(ValueError, match="stop_loss_pips"):
            EAConfig(
                wavelet_config=WaveletEngineConfig(wavelet="haar", window=256, level=2),
                signal_config=SignalConfig(),
                stop_loss_pips=0.0,
            )

    def test_invalid_take_profit(self) -> None:
        with pytest.raises(ValueError, match="take_profit_pips"):
            EAConfig(
                wavelet_config=WaveletEngineConfig(wavelet="haar", window=256, level=2),
                signal_config=SignalConfig(),
                take_profit_pips=-5.0,
            )


# ---------------------------------------------------------------------------
# Expert Advisor Tests
# ---------------------------------------------------------------------------


class TestMT5ExpertAdvisor:
    def test_warmup_no_orders(self) -> None:
        ea = MT5ExpertAdvisor(_make_ea_config())
        for i in range(10):
            result = ea.on_tick(_make_tick(1.10, i))
        assert result is None
        assert len(ea.orders) == 0

    def test_generates_orders_after_warmup(self) -> None:
        ea = MT5ExpertAdvisor(_make_ea_config(buy_z=0.5, sell_z=0.5))
        ticks = _make_ticks(400)
        for tick in ticks:
            ea.on_tick(tick)
        # With low thresholds, should generate some orders
        assert len(ea.orders) > 0

    def test_order_fields(self) -> None:
        ea = MT5ExpertAdvisor(_make_ea_config(buy_z=0.5, sell_z=0.5))
        ticks = _make_ticks(400)
        for tick in ticks:
            result = ea.on_tick(tick)
            if result is not None:
                break
        if result is not None:
            assert isinstance(result, OrderRequest)
            assert result.direction in ("BUY", "SELL")
            assert result.lots > 0
            assert result.stop_loss_pips > 0
            assert result.confidence >= 0.0

    def test_max_positions_respected(self) -> None:
        cfg = EAConfig(
            wavelet_config=WaveletEngineConfig(wavelet="haar", window=256, level=2),
            signal_config=SignalConfig(
                buy_z_threshold=0.5, sell_z_threshold=0.5,
                slope_filter_enabled=False,
            ),
            risk_config=RiskConfig(max_open_positions=1),
        )
        ea = MT5ExpertAdvisor(cfg)
        ticks = _make_ticks(400)
        for tick in ticks:
            ea.on_tick(tick)
        # Can only have max 1 open, but orders accumulate
        # After first order without close, no more should come
        assert ea.open_positions <= 1

    def test_close_position_updates_balance(self) -> None:
        ea = MT5ExpertAdvisor(_make_ea_config(), initial_balance=10000.0)
        ea._open_positions = 1
        ea.close_position(50.0)
        assert ea.balance == 10050.0
        assert ea.open_positions == 0

    def test_close_position_loss_tracks_daily(self) -> None:
        ea = MT5ExpertAdvisor(_make_ea_config(), initial_balance=10000.0)
        ea._open_positions = 1
        ea.close_position(-30.0)
        assert ea.daily_loss == 30.0
        assert ea.balance == 9970.0

    def test_daily_loss_limit(self) -> None:
        cfg = EAConfig(
            wavelet_config=WaveletEngineConfig(wavelet="haar", window=256, level=2),
            signal_config=SignalConfig(
                buy_z_threshold=0.5, sell_z_threshold=0.5,
                slope_filter_enabled=False,
            ),
            risk_config=RiskConfig(max_daily_loss=10.0, max_open_positions=5),
        )
        ea = MT5ExpertAdvisor(cfg)
        ea._daily_loss = 10.0
        ticks = _make_ticks(400)
        for tick in ticks:
            ea.on_tick(tick)
        # No new orders after daily loss exceeded
        assert len(ea.orders) == 0

    def test_reset_daily(self) -> None:
        ea = MT5ExpertAdvisor(_make_ea_config())
        ea._daily_loss = 50.0
        ea.reset_daily()
        assert ea.daily_loss == 0.0

    def test_deterministic(self) -> None:
        cfg = _make_ea_config(buy_z=1.0, sell_z=1.0)
        ticks = _make_ticks(300)

        ea1 = MT5ExpertAdvisor(cfg)
        ea2 = MT5ExpertAdvisor(cfg)
        for tick in ticks:
            ea1.on_tick(tick)
            ea2.on_tick(tick)
        assert len(ea1.orders) == len(ea2.orders)
        for o1, o2 in zip(ea1.orders, ea2.orders):
            assert o1.direction == o2.direction
            assert o1.lots == o2.lots

    def test_paper_mode(self) -> None:
        cfg = _make_ea_config()
        assert cfg.mode == ExecutionMode.PAPER
        ea = MT5ExpertAdvisor(cfg)
        assert ea.config.mode == ExecutionMode.PAPER
