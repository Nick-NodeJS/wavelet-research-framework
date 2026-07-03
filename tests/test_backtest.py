"""Tests for the Backtesting Engine (Story 6)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wavelet_research.backtest.config import BacktestConfig, ExitStrategy
from wavelet_research.backtest.core import BacktestEngine
from wavelet_research.backtest.metrics import compute_report
from wavelet_research.backtest.models import (
    BacktestReport,
    EquityCurve,
    Trade,
    TradeJournal,
)
from wavelet_research.backtest.position import (
    OpenPosition,
    check_exit,
    close_position,
    compute_entry_price,
)
from wavelet_research.engine.config import WaveletEngineConfig
from wavelet_research.engine.core import WaveletEngine
from wavelet_research.engine.models import Tick, WaveletPoint
from wavelet_research.signal.config import SignalConfig
from wavelet_research.signal.core import SignalEngine
from wavelet_research.signal.models import Signal, SignalDecision


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tick(
    mid: float, index: int = 0, spread: float = 0.00010
) -> Tick:
    return Tick(
        time=pd.Timestamp("2026-06-01") + pd.Timedelta(seconds=index),
        bid=mid - spread / 2,
        ask=mid + spread / 2,
        mid=mid,
        spread=spread,
    )


def _make_dataset(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic tick dataset."""
    rng = np.random.default_rng(seed)
    mid = 1.10 + np.cumsum(rng.normal(0, 0.00002, n))
    spread = 0.00010
    return pd.DataFrame({
        "time": pd.date_range("2026-06-01", periods=n, freq="100ms"),
        "bid": mid - spread / 2,
        "ask": mid + spread / 2,
        "mid": mid,
        "spread": np.full(n, spread),
    })


def _make_trending_dataset(
    direction: str = "up", n: int = 500, seed: int = 42
) -> pd.DataFrame:
    """Create a dataset with a strong trend for predictable signals."""
    rng = np.random.default_rng(seed)
    if direction == "up":
        trend = np.linspace(1.10, 1.12, n)
    else:
        trend = np.linspace(1.12, 1.10, n)
    noise = rng.normal(0, 0.00001, n)
    mid = trend + noise
    spread = 0.00010
    return pd.DataFrame({
        "time": pd.date_range("2026-06-01", periods=n, freq="100ms"),
        "bid": mid - spread / 2,
        "ask": mid + spread / 2,
        "mid": mid,
        "spread": np.full(n, spread),
    })


def _make_wavelet_point(
    trend: float = 1.10,
    deviation: float = 0.0,
    z_score: float = 0.0,
    slope: float = 0.0,
) -> WaveletPoint:
    return WaveletPoint(
        trend=trend, deviation=deviation,
        z_score=z_score, slope=slope,
        energy=0.001, noise=0.001,
    )


# ---------------------------------------------------------------------------
# Config Tests
# ---------------------------------------------------------------------------


class TestBacktestConfig:
    def test_valid_config(self) -> None:
        cfg = BacktestConfig()
        assert cfg.max_hold_ticks == 100
        assert cfg.commission_pips == 0.0

    def test_invalid_pip_size(self) -> None:
        with pytest.raises(ValueError, match="pip_size"):
            BacktestConfig(pip_size=0.0)

    def test_invalid_max_hold(self) -> None:
        with pytest.raises(ValueError, match="max_hold_ticks"):
            BacktestConfig(max_hold_ticks=0)

    def test_invalid_take_profit(self) -> None:
        with pytest.raises(ValueError, match="take_profit_pips"):
            BacktestConfig(take_profit_pips=-1.0)

    def test_invalid_stop_loss(self) -> None:
        with pytest.raises(ValueError, match="stop_loss_pips"):
            BacktestConfig(stop_loss_pips=0.0)

    def test_invalid_commission(self) -> None:
        with pytest.raises(ValueError, match="commission_pips"):
            BacktestConfig(commission_pips=-0.1)

    def test_invalid_slippage(self) -> None:
        with pytest.raises(ValueError, match="slippage_pips"):
            BacktestConfig(slippage_pips=-0.1)

    def test_immutable(self) -> None:
        cfg = BacktestConfig()
        with pytest.raises(Exception):
            cfg.max_hold_ticks = 200  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Position Tests
# ---------------------------------------------------------------------------


class TestPosition:
    def test_entry_price_buy(self) -> None:
        tick = _make_tick(1.10)
        price = compute_entry_price(tick, Signal.BUY, 0.0, 0.00001)
        assert price == tick.ask

    def test_entry_price_sell(self) -> None:
        tick = _make_tick(1.10)
        price = compute_entry_price(tick, Signal.SELL, 0.0, 0.00001)
        assert price == tick.bid

    def test_entry_price_with_slippage_buy(self) -> None:
        tick = _make_tick(1.10)
        price = compute_entry_price(tick, Signal.BUY, 1.0, 0.00001)
        assert price == pytest.approx(tick.ask + 0.00001)

    def test_entry_price_with_slippage_sell(self) -> None:
        tick = _make_tick(1.10)
        price = compute_entry_price(tick, Signal.SELL, 1.0, 0.00001)
        assert price == pytest.approx(tick.bid - 0.00001)

    def test_buy_position_pnl(self) -> None:
        tick_entry = _make_tick(1.10000, 0)
        tick_exit = _make_tick(1.10020, 1)
        config = BacktestConfig(pip_size=0.00001)

        pos = OpenPosition(
            side=Signal.BUY,
            entry_index=0,
            entry_price=tick_entry.ask,
            entry_time=tick_entry.time,
            entry_trend=1.10,
        )
        pos.update(tick_exit, config.pip_size)
        pnl = pos.compute_exit_pnl(tick_exit, config)
        # BUY at ask 1.10005, exit at bid 1.10015 => 10 pips
        expected_gross = (tick_exit.bid - tick_entry.ask) / config.pip_size
        assert pnl == pytest.approx(expected_gross)

    def test_sell_position_pnl(self) -> None:
        tick_entry = _make_tick(1.10020, 0)
        tick_exit = _make_tick(1.10000, 1)
        config = BacktestConfig(pip_size=0.00001)

        pos = OpenPosition(
            side=Signal.SELL,
            entry_index=0,
            entry_price=tick_entry.bid,
            entry_time=tick_entry.time,
            entry_trend=1.10,
        )
        pos.update(tick_exit, config.pip_size)
        pnl = pos.compute_exit_pnl(tick_exit, config)
        # SELL at bid 1.10015, exit at ask 1.10005 => 10 pips
        expected_gross = (tick_entry.bid - tick_exit.ask) / config.pip_size
        assert pnl == pytest.approx(expected_gross)

    def test_mae_mfe_tracking(self) -> None:
        config = BacktestConfig(pip_size=0.00001)
        entry_tick = _make_tick(1.10000, 0)
        pos = OpenPosition(
            side=Signal.BUY,
            entry_index=0,
            entry_price=entry_tick.ask,
            entry_time=entry_tick.time,
            entry_trend=1.10,
        )
        # Price goes up then down
        pos.update(_make_tick(1.10020, 1), config.pip_size)  # +15 pips (bid)
        pos.update(_make_tick(1.09980, 2), config.pip_size)  # -25 pips (bid)
        pos.update(_make_tick(1.10010, 3), config.pip_size)  # +5 pips (bid)

        assert pos.mfe_pips > 0
        assert pos.mae_pips > 0  # MAE is positive (absolute)

    def test_holding_ticks(self) -> None:
        entry_tick = _make_tick(1.10, 0)
        pos = OpenPosition(
            side=Signal.BUY,
            entry_index=0,
            entry_price=entry_tick.ask,
            entry_time=entry_tick.time,
            entry_trend=1.10,
        )
        for i in range(5):
            pos.update(_make_tick(1.10, i + 1), 0.00001)
        assert pos.ticks_held == 5


# ---------------------------------------------------------------------------
# Exit Strategy Tests
# ---------------------------------------------------------------------------


class TestExitStrategies:
    def test_max_hold_exit(self) -> None:
        config = BacktestConfig(max_hold_ticks=3)
        tick = _make_tick(1.10, 3)
        pos = OpenPosition(
            side=Signal.BUY, entry_index=0,
            entry_price=tick.ask, entry_time=tick.time,
            entry_trend=1.10,
        )
        for i in range(3):
            pos.update(_make_tick(1.10, i), config.pip_size)
        reason = check_exit(pos, tick, config, None, None)
        assert reason == "max_hold"

    def test_take_profit_exit(self) -> None:
        config = BacktestConfig(
            exit_strategy=ExitStrategy.FIXED_TP_SL,
            take_profit_pips=10.0,
            stop_loss_pips=10.0,
            pip_size=0.00001,
        )
        entry_tick = _make_tick(1.10000, 0)
        pos = OpenPosition(
            side=Signal.BUY, entry_index=0,
            entry_price=entry_tick.ask, entry_time=entry_tick.time,
            entry_trend=1.10,
        )
        # Price moves up well past TP (entry_ask=1.10005, exit_bid=1.10025 => 20 pips)
        exit_tick = _make_tick(1.10030, 1)
        pos.update(exit_tick, config.pip_size)
        reason = check_exit(pos, exit_tick, config, None, None)
        assert reason == "take_profit"

    def test_stop_loss_exit(self) -> None:
        config = BacktestConfig(
            exit_strategy=ExitStrategy.FIXED_TP_SL,
            take_profit_pips=10.0,
            stop_loss_pips=10.0,
            pip_size=0.00001,
        )
        entry_tick = _make_tick(1.10000, 0)
        pos = OpenPosition(
            side=Signal.BUY, entry_index=0,
            entry_price=entry_tick.ask, entry_time=entry_tick.time,
            entry_trend=1.10,
        )
        # Price moves down significantly
        exit_tick = _make_tick(1.09980, 1)
        pos.update(exit_tick, config.pip_size)
        reason = check_exit(pos, exit_tick, config, None, None)
        assert reason == "stop_loss"

    def test_opposite_signal_exit(self) -> None:
        config = BacktestConfig(exit_strategy=ExitStrategy.OPPOSITE_SIGNAL)
        tick = _make_tick(1.10, 1)
        pos = OpenPosition(
            side=Signal.BUY, entry_index=0,
            entry_price=tick.ask, entry_time=tick.time,
            entry_trend=1.10,
        )
        pos.update(tick, config.pip_size)
        sell_decision = SignalDecision(
            signal=Signal.SELL, confidence=0.8, reason="test",
            z_score=3.0, trend_slope=-0.001, energy=0.01, noise=0.001,
        )
        reason = check_exit(pos, tick, config, None, sell_decision)
        assert reason == "opposite_signal"

    def test_return_to_trend_exit_buy(self) -> None:
        config = BacktestConfig(exit_strategy=ExitStrategy.RETURN_TO_TREND)
        # BUY position, price returns to trend
        entry_tick = _make_tick(1.09990, 0)  # below trend
        pos = OpenPosition(
            side=Signal.BUY, entry_index=0,
            entry_price=entry_tick.ask, entry_time=entry_tick.time,
            entry_trend=1.10000,
        )
        exit_tick = _make_tick(1.10010, 1)  # above trend
        pos.update(exit_tick, config.pip_size)
        wp = _make_wavelet_point(trend=1.10000)
        reason = check_exit(pos, exit_tick, config, wp, None)
        assert reason == "return_to_trend"

    def test_return_to_trend_exit_sell(self) -> None:
        config = BacktestConfig(exit_strategy=ExitStrategy.RETURN_TO_TREND)
        entry_tick = _make_tick(1.10010, 0)  # above trend
        pos = OpenPosition(
            side=Signal.SELL, entry_index=0,
            entry_price=entry_tick.bid, entry_time=entry_tick.time,
            entry_trend=1.10000,
        )
        exit_tick = _make_tick(1.09990, 1)  # below trend
        pos.update(exit_tick, config.pip_size)
        wp = _make_wavelet_point(trend=1.10000)
        reason = check_exit(pos, exit_tick, config, wp, None)
        assert reason == "return_to_trend"

    def test_no_exit_when_conditions_not_met(self) -> None:
        config = BacktestConfig(
            exit_strategy=ExitStrategy.FIXED_TP_SL,
            take_profit_pips=100.0,
            stop_loss_pips=100.0,
            max_hold_ticks=1000,
        )
        entry_tick = _make_tick(1.10000, 0)
        pos = OpenPosition(
            side=Signal.BUY, entry_index=0,
            entry_price=entry_tick.ask, entry_time=entry_tick.time,
            entry_trend=1.10,
        )
        tick = _make_tick(1.10001, 1)
        pos.update(tick, config.pip_size)
        reason = check_exit(pos, tick, config, None, None)
        assert reason is None


# ---------------------------------------------------------------------------
# Costs Tests
# ---------------------------------------------------------------------------


class TestCosts:
    def test_commission_deducted(self) -> None:
        config = BacktestConfig(pip_size=0.00001, commission_pips=2.0)
        entry_tick = _make_tick(1.10000, 0)
        exit_tick = _make_tick(1.10010, 1)  # +5 pips gross for BUY

        pos = OpenPosition(
            side=Signal.BUY, entry_index=0,
            entry_price=entry_tick.ask, entry_time=entry_tick.time,
            entry_trend=1.10,
        )
        gross = pos.compute_gross_pnl(exit_tick, config.pip_size)
        net = pos.compute_exit_pnl(exit_tick, config)
        assert net == pytest.approx(gross - 2.0)

    def test_slippage_deducted(self) -> None:
        config = BacktestConfig(pip_size=0.00001, slippage_pips=1.0)
        entry_tick = _make_tick(1.10000, 0)
        exit_tick = _make_tick(1.10010, 1)

        pos = OpenPosition(
            side=Signal.BUY, entry_index=0,
            entry_price=entry_tick.ask, entry_time=entry_tick.time,
            entry_trend=1.10,
        )
        gross = pos.compute_gross_pnl(exit_tick, config.pip_size)
        net = pos.compute_exit_pnl(exit_tick, config)
        # 2 * slippage_pips deducted
        assert net == pytest.approx(gross - 2.0)

    def test_commission_and_slippage_combined(self) -> None:
        config = BacktestConfig(
            pip_size=0.00001, commission_pips=1.0, slippage_pips=0.5
        )
        entry_tick = _make_tick(1.10000, 0)
        exit_tick = _make_tick(1.10010, 1)

        pos = OpenPosition(
            side=Signal.BUY, entry_index=0,
            entry_price=entry_tick.ask, entry_time=entry_tick.time,
            entry_trend=1.10,
        )
        gross = pos.compute_gross_pnl(exit_tick, config.pip_size)
        net = pos.compute_exit_pnl(exit_tick, config)
        total_costs = 1.0 + 2 * 0.5  # commission + 2*slippage
        assert net == pytest.approx(gross - total_costs)


# ---------------------------------------------------------------------------
# Metrics / Report Tests
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_empty_trades(self) -> None:
        report = compute_report([])
        assert report.trades == 0
        assert report.total_pnl == 0.0
        assert report.max_drawdown == 0.0

    def test_report_fields(self) -> None:
        trades = [
            Trade(
                entry_index=0, exit_index=5, side="BUY",
                entry_price=1.10005, exit_price=1.10015,
                pnl_pips=10.0, gross_pnl_pips=10.0,
                mae_pips=2.0, mfe_pips=15.0,
                holding_ticks=5, exit_reason="max_hold",
            ),
            Trade(
                entry_index=10, exit_index=15, side="SELL",
                entry_price=1.10015, exit_price=1.10010,
                pnl_pips=-5.0, gross_pnl_pips=-5.0,
                mae_pips=8.0, mfe_pips=3.0,
                holding_ticks=5, exit_reason="max_hold",
            ),
        ]
        report = compute_report(trades)
        assert report.trades == 2
        assert report.win_rate == pytest.approx(0.5)
        assert report.total_pnl == pytest.approx(5.0)
        assert report.expectancy == pytest.approx(2.5)
        assert report.average_trade == pytest.approx(2.5)
        assert report.average_mae == pytest.approx(5.0)
        assert report.average_mfe == pytest.approx(9.0)
        assert report.average_holding_time == pytest.approx(5.0)
        assert report.journal.count == 2
        assert len(report.equity_curve.values) == 2

    def test_drawdown_calculation(self) -> None:
        trades = [
            Trade(
                entry_index=0, exit_index=1, side="BUY",
                entry_price=1.10, exit_price=1.10,
                pnl_pips=10.0, gross_pnl_pips=10.0,
                mae_pips=0.0, mfe_pips=10.0,
                holding_ticks=1, exit_reason="tp",
            ),
            Trade(
                entry_index=2, exit_index=3, side="BUY",
                entry_price=1.10, exit_price=1.10,
                pnl_pips=-8.0, gross_pnl_pips=-8.0,
                mae_pips=8.0, mfe_pips=0.0,
                holding_ticks=1, exit_reason="sl",
            ),
            Trade(
                entry_index=4, exit_index=5, side="BUY",
                entry_price=1.10, exit_price=1.10,
                pnl_pips=5.0, gross_pnl_pips=5.0,
                mae_pips=0.0, mfe_pips=5.0,
                holding_ticks=1, exit_reason="tp",
            ),
        ]
        report = compute_report(trades)
        # Equity: 10, 2, 7 => drawdown from 10 to 2 = 8
        assert report.max_drawdown == pytest.approx(8.0)

    def test_equity_curve(self) -> None:
        trades = [
            Trade(
                entry_index=0, exit_index=1, side="BUY",
                entry_price=1.10, exit_price=1.10,
                pnl_pips=10.0, gross_pnl_pips=10.0,
                mae_pips=0.0, mfe_pips=10.0,
                holding_ticks=1, exit_reason="tp",
            ),
            Trade(
                entry_index=2, exit_index=3, side="SELL",
                entry_price=1.10, exit_price=1.10,
                pnl_pips=-3.0, gross_pnl_pips=-3.0,
                mae_pips=3.0, mfe_pips=0.0,
                holding_ticks=1, exit_reason="sl",
            ),
        ]
        report = compute_report(trades)
        assert report.equity_curve.values == pytest.approx((10.0, 7.0))
        assert report.equity_curve.peak == pytest.approx(10.0)
        assert report.equity_curve.final == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# BacktestEngine Integration Tests
# ---------------------------------------------------------------------------


class TestBacktestEngine:
    def test_empty_data_raises(self) -> None:
        config = BacktestConfig()
        engine = BacktestEngine(config)
        wavelet_cfg = WaveletEngineConfig(
            wavelet="haar", window=256, level=2
        )
        wavelet_engine = WaveletEngine(wavelet_cfg)
        signal_cfg = SignalConfig(buy_z_threshold=2.0, sell_z_threshold=2.0)
        signal_engine = SignalEngine(signal_cfg)
        with pytest.raises(ValueError, match="empty"):
            engine.run(pd.DataFrame(), wavelet_engine, signal_engine)

    def test_run_produces_report(self) -> None:
        data = _make_dataset(n=500)
        config = BacktestConfig(
            pip_size=0.00001,
            exit_strategy=ExitStrategy.MAX_HOLD,
            max_hold_ticks=10,
        )
        wavelet_cfg = WaveletEngineConfig(
            wavelet="haar", window=256, level=2
        )
        signal_cfg = SignalConfig(
            buy_z_threshold=1.5, sell_z_threshold=1.5,
            slope_filter_enabled=False,
        )
        engine = BacktestEngine(config)
        wavelet_engine = WaveletEngine(wavelet_cfg)
        signal_engine = SignalEngine(signal_cfg)

        report = engine.run(data, wavelet_engine, signal_engine)
        assert isinstance(report, BacktestReport)
        assert report.trades >= 0

    def test_deterministic_replay(self) -> None:
        data = _make_dataset(n=500)
        config = BacktestConfig(
            pip_size=0.00001,
            exit_strategy=ExitStrategy.MAX_HOLD,
            max_hold_ticks=10,
        )
        wavelet_cfg = WaveletEngineConfig(
            wavelet="haar", window=256, level=2
        )
        signal_cfg = SignalConfig(
            buy_z_threshold=1.5, sell_z_threshold=1.5,
            slope_filter_enabled=False,
        )

        # Run 1
        engine1 = BacktestEngine(config)
        we1 = WaveletEngine(wavelet_cfg)
        se1 = SignalEngine(signal_cfg)
        report1 = engine1.run(data, we1, se1)

        # Run 2
        engine2 = BacktestEngine(config)
        we2 = WaveletEngine(wavelet_cfg)
        se2 = SignalEngine(signal_cfg)
        report2 = engine2.run(data, we2, se2)

        assert report1.trades == report2.trades
        assert report1.total_pnl == report2.total_pnl
        assert report1.win_rate == report2.win_rate
        assert report1.profit_factor == report2.profit_factor

    def test_buy_lifecycle(self) -> None:
        """Verify a BUY trade goes through entry-hold-exit correctly."""
        data = _make_dataset(n=400)
        config = BacktestConfig(
            pip_size=0.00001,
            exit_strategy=ExitStrategy.MAX_HOLD,
            max_hold_ticks=5,
        )
        wavelet_cfg = WaveletEngineConfig(
            wavelet="haar", window=256, level=2
        )
        signal_cfg = SignalConfig(
            buy_z_threshold=0.5, sell_z_threshold=100.0,
            slope_filter_enabled=False,
        )
        engine = BacktestEngine(config)
        report = engine.run(
            data, WaveletEngine(wavelet_cfg), SignalEngine(signal_cfg)
        )
        # With low threshold and sell disabled, should get some BUY trades
        if report.trades > 0:
            for trade in report.journal.trades:
                assert trade.side == "BUY"
                assert trade.holding_ticks <= config.max_hold_ticks

    def test_sell_lifecycle(self) -> None:
        """Verify a SELL trade goes through entry-hold-exit correctly."""
        data = _make_dataset(n=400)
        config = BacktestConfig(
            pip_size=0.00001,
            exit_strategy=ExitStrategy.MAX_HOLD,
            max_hold_ticks=5,
        )
        wavelet_cfg = WaveletEngineConfig(
            wavelet="haar", window=256, level=2
        )
        signal_cfg = SignalConfig(
            buy_z_threshold=100.0, sell_z_threshold=0.5,
            slope_filter_enabled=False,
        )
        engine = BacktestEngine(config)
        report = engine.run(
            data, WaveletEngine(wavelet_cfg), SignalEngine(signal_cfg)
        )
        if report.trades > 0:
            for trade in report.journal.trades:
                assert trade.side == "SELL"
                assert trade.holding_ticks <= config.max_hold_ticks

    def test_tp_sl_exit_strategy(self) -> None:
        data = _make_dataset(n=500)
        config = BacktestConfig(
            pip_size=0.00001,
            exit_strategy=ExitStrategy.FIXED_TP_SL,
            take_profit_pips=5.0,
            stop_loss_pips=5.0,
            max_hold_ticks=50,
        )
        wavelet_cfg = WaveletEngineConfig(
            wavelet="haar", window=256, level=2
        )
        signal_cfg = SignalConfig(
            buy_z_threshold=1.0, sell_z_threshold=1.0,
            slope_filter_enabled=False,
        )
        engine = BacktestEngine(config)
        report = engine.run(
            data, WaveletEngine(wavelet_cfg), SignalEngine(signal_cfg)
        )
        if report.trades > 0:
            for trade in report.journal.trades:
                assert trade.exit_reason in (
                    "take_profit", "stop_loss", "max_hold", "end_of_data"
                )

    def test_journal_has_all_trades(self) -> None:
        data = _make_dataset(n=500)
        config = BacktestConfig(
            pip_size=0.00001,
            exit_strategy=ExitStrategy.MAX_HOLD,
            max_hold_ticks=5,
        )
        wavelet_cfg = WaveletEngineConfig(
            wavelet="haar", window=256, level=2
        )
        signal_cfg = SignalConfig(
            buy_z_threshold=1.0, sell_z_threshold=1.0,
            slope_filter_enabled=False,
        )
        engine = BacktestEngine(config)
        report = engine.run(
            data, WaveletEngine(wavelet_cfg), SignalEngine(signal_cfg)
        )
        assert report.journal.count == report.trades
        assert len(report.journal.pnl_values()) == report.trades

    def test_costs_applied(self) -> None:
        data = _make_dataset(n=500)
        config_no_cost = BacktestConfig(
            pip_size=0.00001,
            exit_strategy=ExitStrategy.MAX_HOLD,
            max_hold_ticks=10,
            commission_pips=0.0,
            slippage_pips=0.0,
        )
        config_with_cost = BacktestConfig(
            pip_size=0.00001,
            exit_strategy=ExitStrategy.MAX_HOLD,
            max_hold_ticks=10,
            commission_pips=2.0,
            slippage_pips=1.0,
        )
        wavelet_cfg = WaveletEngineConfig(
            wavelet="haar", window=256, level=2
        )
        signal_cfg = SignalConfig(
            buy_z_threshold=1.0, sell_z_threshold=1.0,
            slope_filter_enabled=False,
        )

        report_no = BacktestEngine(config_no_cost).run(
            data, WaveletEngine(wavelet_cfg), SignalEngine(signal_cfg)
        )
        report_with = BacktestEngine(config_with_cost).run(
            data, WaveletEngine(wavelet_cfg), SignalEngine(signal_cfg)
        )

        if report_no.trades > 0 and report_with.trades > 0:
            # Total PnL with costs should be less
            assert report_with.total_pnl < report_no.total_pnl
