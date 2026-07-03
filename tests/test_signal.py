"""Tests for the Signal Engine (Story 5)."""

from __future__ import annotations

import pytest

from wavelet_research.engine.models import WaveletPoint
from wavelet_research.signal.config import SignalConfig
from wavelet_research.signal.core import SignalEngine
from wavelet_research.signal.models import Signal, SignalDecision
from wavelet_research.signal.rules import (
    check_buy_threshold,
    check_energy_filter,
    check_noise_filter,
    check_sell_threshold,
    check_slope_filter,
    compute_confidence,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_point(
    z_score: float = 0.0,
    slope: float = 0.0,
    energy: float = 0.001,
    noise: float = 0.001,
    trend: float = 1.10,
    deviation: float = 0.0,
) -> WaveletPoint:
    return WaveletPoint(
        trend=trend,
        deviation=deviation,
        z_score=z_score,
        slope=slope,
        energy=energy,
        noise=noise,
    )


def _default_config(**kwargs) -> SignalConfig:
    defaults = dict(
        buy_z_threshold=2.0,
        sell_z_threshold=2.0,
        min_confidence=0.0,
        slope_filter_enabled=True,
        energy_filter_enabled=False,
        noise_filter_enabled=False,
        max_noise=1.0,
        min_energy=0.0,
        allow_buy=True,
        allow_sell=True,
    )
    defaults.update(kwargs)
    return SignalConfig(**defaults)


# ---------------------------------------------------------------------------
# Config Validation Tests
# ---------------------------------------------------------------------------


class TestSignalConfig:
    def test_valid_config(self) -> None:
        cfg = _default_config()
        assert cfg.buy_z_threshold == 2.0

    def test_invalid_buy_threshold(self) -> None:
        with pytest.raises(ValueError, match="buy_z_threshold"):
            SignalConfig(buy_z_threshold=0.0)

    def test_invalid_sell_threshold(self) -> None:
        with pytest.raises(ValueError, match="sell_z_threshold"):
            SignalConfig(sell_z_threshold=-1.0)

    def test_invalid_min_confidence_low(self) -> None:
        with pytest.raises(ValueError, match="min_confidence"):
            SignalConfig(min_confidence=-0.1)

    def test_invalid_min_confidence_high(self) -> None:
        with pytest.raises(ValueError, match="min_confidence"):
            SignalConfig(min_confidence=1.1)

    def test_invalid_max_noise(self) -> None:
        with pytest.raises(ValueError, match="max_noise"):
            SignalConfig(max_noise=0.0)

    def test_invalid_min_energy(self) -> None:
        with pytest.raises(ValueError, match="min_energy"):
            SignalConfig(min_energy=-0.01)

    def test_immutable(self) -> None:
        cfg = _default_config()
        with pytest.raises(Exception):
            cfg.buy_z_threshold = 5.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Rule Tests
# ---------------------------------------------------------------------------


class TestRules:
    def test_buy_threshold_crossed(self) -> None:
        cfg = _default_config(buy_z_threshold=2.0)
        point = _make_point(z_score=-2.5)
        assert check_buy_threshold(point, cfg) is True

    def test_buy_threshold_not_crossed(self) -> None:
        cfg = _default_config(buy_z_threshold=2.0)
        point = _make_point(z_score=-1.5)
        assert check_buy_threshold(point, cfg) is False

    def test_buy_threshold_exact_boundary(self) -> None:
        cfg = _default_config(buy_z_threshold=2.0)
        point = _make_point(z_score=-2.0)
        assert check_buy_threshold(point, cfg) is True

    def test_sell_threshold_crossed(self) -> None:
        cfg = _default_config(sell_z_threshold=2.0)
        point = _make_point(z_score=2.5)
        assert check_sell_threshold(point, cfg) is True

    def test_sell_threshold_not_crossed(self) -> None:
        cfg = _default_config(sell_z_threshold=2.0)
        point = _make_point(z_score=1.5)
        assert check_sell_threshold(point, cfg) is False

    def test_sell_threshold_exact_boundary(self) -> None:
        cfg = _default_config(sell_z_threshold=2.0)
        point = _make_point(z_score=2.0)
        assert check_sell_threshold(point, cfg) is True

    def test_slope_filter_buy_positive_slope(self) -> None:
        point = _make_point(slope=0.001)
        assert check_slope_filter(point, Signal.BUY) is True

    def test_slope_filter_buy_negative_slope(self) -> None:
        point = _make_point(slope=-0.001)
        assert check_slope_filter(point, Signal.BUY) is False

    def test_slope_filter_sell_negative_slope(self) -> None:
        point = _make_point(slope=-0.001)
        assert check_slope_filter(point, Signal.SELL) is True

    def test_slope_filter_sell_positive_slope(self) -> None:
        point = _make_point(slope=0.001)
        assert check_slope_filter(point, Signal.SELL) is False

    def test_slope_filter_zero_slope(self) -> None:
        point = _make_point(slope=0.0)
        assert check_slope_filter(point, Signal.BUY) is True
        assert check_slope_filter(point, Signal.SELL) is True

    def test_energy_filter_passes(self) -> None:
        cfg = _default_config(min_energy=0.001)
        point = _make_point(energy=0.01)
        assert check_energy_filter(point, cfg) is True

    def test_energy_filter_rejects(self) -> None:
        cfg = _default_config(min_energy=0.1)
        point = _make_point(energy=0.01)
        assert check_energy_filter(point, cfg) is False

    def test_noise_filter_passes(self) -> None:
        cfg = _default_config(max_noise=1.0)
        point = _make_point(noise=0.5)
        assert check_noise_filter(point, cfg) is True

    def test_noise_filter_rejects(self) -> None:
        cfg = _default_config(max_noise=0.01)
        point = _make_point(noise=0.5)
        assert check_noise_filter(point, cfg) is False


# ---------------------------------------------------------------------------
# Confidence Tests
# ---------------------------------------------------------------------------


class TestConfidence:
    def test_confidence_in_range(self) -> None:
        cfg = _default_config()
        point = _make_point(z_score=-3.0, slope=0.001, energy=0.01, noise=0.001)
        conf = compute_confidence(point, cfg, Signal.BUY)
        assert 0.0 <= conf <= 1.0

    def test_higher_z_gives_higher_confidence(self) -> None:
        cfg = _default_config(buy_z_threshold=2.0)
        point_weak = _make_point(z_score=-2.1, slope=0.001, energy=0.01, noise=0.001)
        point_strong = _make_point(z_score=-4.0, slope=0.001, energy=0.01, noise=0.001)
        conf_weak = compute_confidence(point_weak, cfg, Signal.BUY)
        conf_strong = compute_confidence(point_strong, cfg, Signal.BUY)
        assert conf_strong > conf_weak

    def test_noise_reduces_confidence(self) -> None:
        cfg = _default_config()
        point_clean = _make_point(z_score=-3.0, slope=0.001, energy=0.01, noise=0.0)
        point_noisy = _make_point(z_score=-3.0, slope=0.001, energy=0.01, noise=0.05)
        conf_clean = compute_confidence(point_clean, cfg, Signal.BUY)
        conf_noisy = compute_confidence(point_noisy, cfg, Signal.BUY)
        assert conf_clean >= conf_noisy

    def test_slope_alignment_bonus(self) -> None:
        cfg = _default_config()
        point_aligned = _make_point(z_score=-3.0, slope=0.001, energy=0.0, noise=0.0)
        point_misaligned = _make_point(z_score=-3.0, slope=-0.001, energy=0.0, noise=0.0)
        conf_aligned = compute_confidence(point_aligned, cfg, Signal.BUY)
        conf_misaligned = compute_confidence(point_misaligned, cfg, Signal.BUY)
        assert conf_aligned > conf_misaligned


# ---------------------------------------------------------------------------
# SignalEngine Core Tests
# ---------------------------------------------------------------------------


class TestSignalEngine:
    def test_buy_signal(self) -> None:
        cfg = _default_config()
        engine = SignalEngine(cfg)
        point = _make_point(z_score=-2.5, slope=0.001)
        decision = engine.decide(point)
        assert decision.signal == Signal.BUY
        assert decision.confidence > 0

    def test_sell_signal(self) -> None:
        cfg = _default_config()
        engine = SignalEngine(cfg)
        point = _make_point(z_score=2.5, slope=-0.001)
        decision = engine.decide(point)
        assert decision.signal == Signal.SELL
        assert decision.confidence > 0

    def test_hold_signal(self) -> None:
        cfg = _default_config()
        engine = SignalEngine(cfg)
        point = _make_point(z_score=0.5)
        decision = engine.decide(point)
        assert decision.signal == Signal.HOLD
        assert decision.confidence == 0.0

    def test_hold_reason(self) -> None:
        cfg = _default_config()
        engine = SignalEngine(cfg)
        point = _make_point(z_score=0.5)
        decision = engine.decide(point)
        assert "no_threshold_crossed" in decision.reason

    def test_buy_disabled(self) -> None:
        cfg = _default_config(allow_buy=False)
        engine = SignalEngine(cfg)
        point = _make_point(z_score=-3.0, slope=0.001)
        decision = engine.decide(point)
        assert decision.signal == Signal.HOLD

    def test_sell_disabled(self) -> None:
        cfg = _default_config(allow_sell=False)
        engine = SignalEngine(cfg)
        point = _make_point(z_score=3.0, slope=-0.001)
        decision = engine.decide(point)
        assert decision.signal == Signal.HOLD

    def test_slope_filter_rejects_buy(self) -> None:
        cfg = _default_config(slope_filter_enabled=True)
        engine = SignalEngine(cfg)
        # z-score crosses BUY threshold but slope is negative
        point = _make_point(z_score=-3.0, slope=-0.001)
        decision = engine.decide(point)
        assert decision.signal == Signal.HOLD
        assert "slope_filter" in decision.reason

    def test_slope_filter_rejects_sell(self) -> None:
        cfg = _default_config(slope_filter_enabled=True)
        engine = SignalEngine(cfg)
        # z-score crosses SELL threshold but slope is positive
        point = _make_point(z_score=3.0, slope=0.001)
        decision = engine.decide(point)
        assert decision.signal == Signal.HOLD
        assert "slope_filter" in decision.reason

    def test_slope_filter_disabled(self) -> None:
        cfg = _default_config(slope_filter_enabled=False)
        engine = SignalEngine(cfg)
        # Would be rejected by slope filter, but it's disabled
        point = _make_point(z_score=-3.0, slope=-0.001)
        decision = engine.decide(point)
        assert decision.signal == Signal.BUY

    def test_energy_filter_rejects(self) -> None:
        cfg = _default_config(energy_filter_enabled=True, min_energy=0.1)
        engine = SignalEngine(cfg)
        point = _make_point(z_score=-3.0, slope=0.001, energy=0.001)
        decision = engine.decide(point)
        assert decision.signal == Signal.HOLD
        assert "energy_filter" in decision.reason

    def test_energy_filter_passes(self) -> None:
        cfg = _default_config(energy_filter_enabled=True, min_energy=0.001)
        engine = SignalEngine(cfg)
        point = _make_point(z_score=-3.0, slope=0.001, energy=0.01)
        decision = engine.decide(point)
        assert decision.signal == Signal.BUY

    def test_noise_filter_rejects(self) -> None:
        cfg = _default_config(noise_filter_enabled=True, max_noise=0.001)
        engine = SignalEngine(cfg)
        point = _make_point(z_score=-3.0, slope=0.001, noise=0.5)
        decision = engine.decide(point)
        assert decision.signal == Signal.HOLD
        assert "noise_filter" in decision.reason

    def test_noise_filter_passes(self) -> None:
        cfg = _default_config(noise_filter_enabled=True, max_noise=1.0)
        engine = SignalEngine(cfg)
        point = _make_point(z_score=-3.0, slope=0.001, noise=0.5)
        decision = engine.decide(point)
        assert decision.signal == Signal.BUY

    def test_min_confidence_rejects(self) -> None:
        cfg = _default_config(min_confidence=0.99)
        engine = SignalEngine(cfg)
        # Just barely crosses threshold — low confidence
        point = _make_point(z_score=-2.01, slope=0.0, energy=0.0, noise=0.01)
        decision = engine.decide(point)
        assert decision.signal == Signal.HOLD
        assert "min_confidence" in decision.reason

    def test_decision_fields_populated(self) -> None:
        cfg = _default_config()
        engine = SignalEngine(cfg)
        point = _make_point(z_score=-3.0, slope=0.001, energy=0.01, noise=0.002)
        decision = engine.decide(point)
        assert decision.z_score == point.z_score
        assert decision.trend_slope == point.slope
        assert decision.energy == point.energy
        assert decision.noise == point.noise


# ---------------------------------------------------------------------------
# Determinism Tests
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_input_same_output(self) -> None:
        cfg = _default_config()
        engine = SignalEngine(cfg)
        point = _make_point(z_score=-2.5, slope=0.001, energy=0.01, noise=0.002)
        d1 = engine.decide(point)
        d2 = engine.decide(point)
        assert d1.signal == d2.signal
        assert d1.confidence == d2.confidence
        assert d1.reason == d2.reason

    def test_deterministic_across_instances(self) -> None:
        cfg = _default_config()
        engine1 = SignalEngine(cfg)
        engine2 = SignalEngine(cfg)
        point = _make_point(z_score=3.0, slope=-0.001, energy=0.01, noise=0.002)
        d1 = engine1.decide(point)
        d2 = engine2.decide(point)
        assert d1.signal == d2.signal
        assert d1.confidence == d2.confidence


# ---------------------------------------------------------------------------
# Multi-scale Tests
# ---------------------------------------------------------------------------


class TestMultiScale:
    def test_decide_many_single_point(self) -> None:
        cfg = _default_config()
        engine = SignalEngine(cfg)
        point = _make_point(z_score=-3.0, slope=0.001)
        decision = engine.decide_many([point])
        assert decision.signal == Signal.BUY

    def test_decide_many_selects_strongest(self) -> None:
        cfg = _default_config()
        engine = SignalEngine(cfg)
        weak = _make_point(z_score=-1.5, slope=0.001)
        strong = _make_point(z_score=-4.0, slope=0.001)
        mild = _make_point(z_score=-0.5, slope=0.001)
        decision = engine.decide_many([weak, strong, mild])
        # Should decide based on strongest z-score
        assert decision.signal == Signal.BUY
        assert decision.z_score == strong.z_score

    def test_decide_many_metadata(self) -> None:
        cfg = _default_config()
        engine = SignalEngine(cfg)
        points = [
            _make_point(z_score=-3.0, slope=0.001),
            _make_point(z_score=-2.5, slope=0.001),
            _make_point(z_score=-1.0, slope=0.001),
        ]
        decision = engine.decide_many(points)
        assert decision.metadata["scale_count"] == 3
        assert "agreement_ratio" in decision.metadata
        assert "all_z_scores" in decision.metadata

    def test_decide_many_empty_raises(self) -> None:
        cfg = _default_config()
        engine = SignalEngine(cfg)
        with pytest.raises(ValueError, match="At least one"):
            engine.decide_many([])

    def test_decide_many_mixed_directions(self) -> None:
        cfg = _default_config(slope_filter_enabled=False)
        engine = SignalEngine(cfg)
        buy_point = _make_point(z_score=-3.0)
        sell_point = _make_point(z_score=4.0)  # Stronger absolute z
        decision = engine.decide_many([buy_point, sell_point])
        # Should pick the strongest absolute z-score (sell_point)
        assert decision.signal == Signal.SELL
