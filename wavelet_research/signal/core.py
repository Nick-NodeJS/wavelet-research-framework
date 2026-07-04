"""Core Signal Engine for deterministic signal generation.

Converts WaveletPoint data into BUY, SELL, or HOLD decisions
using configurable threshold rules and optional filters.
"""

from __future__ import annotations

import logging
from typing import Sequence

from wavelet_research.deviation.models import DeviationPoint
from wavelet_research.deviation_stats.models import DeviationQueryResult
from wavelet_research.engine.models import WaveletPoint
from wavelet_research.filters.models import FilterResult
from wavelet_research.signal.config import SignalConfig
from wavelet_research.signal.models import Signal, SignalDecision
from wavelet_research.signal.rules import (
    check_buy_threshold,
    check_energy_filter,
    check_noise_filter,
    check_sell_threshold,
    check_slope_filter,
    compute_confidence,
)

logger = logging.getLogger(__name__)


def _hold_decision(point: WaveletPoint, reason: str) -> SignalDecision:
    """Create a HOLD decision.

    Parameters
    ----------
    point : WaveletPoint
        Current wavelet features.
    reason : str
        Reason for holding.

    Returns
    -------
    SignalDecision
        HOLD decision with zero confidence.
    """
    return SignalDecision(
        signal=Signal.HOLD,
        confidence=0.0,
        reason=reason,
        z_score=point.z_score,
        trend_slope=point.slope,
        energy=point.energy,
        noise=point.noise,
    )


class SignalEngine:
    """Deterministic signal generation engine.

    Converts WaveletPoint data into SignalDecision using configurable
    threshold-based rules and optional filters (slope, energy, noise).

    Parameters
    ----------
    config : SignalConfig
        Immutable signal configuration.
    """

    def __init__(self, config: SignalConfig) -> None:
        self._config = config

    @property
    def config(self) -> SignalConfig:
        """Signal configuration."""
        return self._config

    def decide(self, point: WaveletPoint) -> SignalDecision:
        """Generate a signal decision from a single WaveletPoint.

        Parameters
        ----------
        point : WaveletPoint
            Current wavelet features.

        Returns
        -------
        SignalDecision
            BUY, SELL, or HOLD decision with confidence and metadata.
        """
        # Check BUY
        if self._config.allow_buy and check_buy_threshold(point, self._config):
            return self._evaluate_signal(point, Signal.BUY)

        # Check SELL
        if self._config.allow_sell and check_sell_threshold(point, self._config):
            return self._evaluate_signal(point, Signal.SELL)

        return _hold_decision(point, "no_threshold_crossed")

    def decide_many(self, points: Sequence[WaveletPoint]) -> SignalDecision:
        """Generate a signal decision from multiple WaveletPoints (multi-scale).

        Aggregates by selecting the point with the strongest absolute z-score
        and deciding based on that. This supports future multi-scale input
        without hardcoding specific timeframes.

        Parameters
        ----------
        points : Sequence[WaveletPoint]
            WaveletPoints from different scales/windows.

        Returns
        -------
        SignalDecision
            Aggregated decision with metadata including scale count.

        Raises
        ------
        ValueError
            If points is empty.
        """
        if not points:
            raise ValueError("At least one WaveletPoint is required.")

        if len(points) == 1:
            return self.decide(points[0])

        # Select the point with strongest absolute z-score
        strongest = max(points, key=lambda p: abs(p.z_score))
        decision = self.decide(strongest)

        # Enrich metadata with multi-scale info
        all_z_scores = [p.z_score for p in points]
        agreement_count = sum(
            1 for z in all_z_scores
            if (z > 0) == (strongest.z_score > 0)
        )

        metadata = dict(decision.metadata)
        metadata["scale_count"] = len(points)
        metadata["agreement_ratio"] = agreement_count / len(points)
        metadata["all_z_scores"] = all_z_scores

        return SignalDecision(
            signal=decision.signal,
            confidence=decision.confidence,
            reason=decision.reason,
            z_score=decision.z_score,
            trend_slope=decision.trend_slope,
            energy=decision.energy,
            noise=decision.noise,
            metadata=metadata,
        )

    def decide_with_context(
        self,
        point: WaveletPoint,
        deviation: DeviationPoint,
        stats: DeviationQueryResult,
        filter_result: FilterResult,
    ) -> SignalDecision:
        """Generate a signal using trend-relative entry rules (Story 22).

        Combines the base threshold rules with:
        - filter engine gate (can_trade)
        - minimum normalized deviation
        - historical return probability gate
        - minimum stats sample size

        Parameters
        ----------
        point : WaveletPoint
            Current wavelet features.
        deviation : DeviationPoint
            Normalized deviation measures.
        stats : DeviationQueryResult
            Historical stats for this market state.
        filter_result : FilterResult
            Result from the filter engine.

        Returns
        -------
        SignalDecision
            Enriched decision with entry context in metadata.
        """
        if not filter_result.can_trade:
            return SignalDecision(
                signal=Signal.HOLD,
                confidence=0.0,
                reason="filter_blocked",
                z_score=point.z_score,
                trend_slope=point.slope,
                energy=point.energy,
                noise=point.noise,
                metadata={
                    "filter_reasons": [r.value for r in filter_result.reasons],
                    "normalized_deviation": deviation.z_score,
                    "historical_probability": stats.return_to_trend_probability,
                    "expected_bars_to_return": stats.median_bars_to_return,
                },
            )

        if abs(deviation.z_score) < self._config.min_normalized_deviation:
            return SignalDecision(
                signal=Signal.HOLD,
                confidence=0.0,
                reason="deviation_too_small",
                z_score=point.z_score,
                trend_slope=point.slope,
                energy=point.energy,
                noise=point.noise,
                metadata={"normalized_deviation": deviation.z_score},
            )

        if stats.sample_size < self._config.min_stats_sample_size:
            return SignalDecision(
                signal=Signal.HOLD,
                confidence=0.0,
                reason="insufficient_history",
                z_score=point.z_score,
                trend_slope=point.slope,
                energy=point.energy,
                noise=point.noise,
                metadata={"sample_size": stats.sample_size},
            )

        if stats.return_to_trend_probability < self._config.min_return_probability:
            return SignalDecision(
                signal=Signal.HOLD,
                confidence=0.0,
                reason="low_return_probability",
                z_score=point.z_score,
                trend_slope=point.slope,
                energy=point.energy,
                noise=point.noise,
                metadata={"historical_probability": stats.return_to_trend_probability},
            )

        base_decision = self.decide(point)

        if base_decision.signal == Signal.HOLD:
            return base_decision

        meta = dict(base_decision.metadata)
        meta["normalized_deviation"] = deviation.z_score
        meta["historical_probability"] = stats.return_to_trend_probability
        meta["expected_bars_to_return"] = stats.median_bars_to_return
        meta["filter_reasons"] = []
        meta["entry_reason"] = (
            "below_trend_return_probability"
            if base_decision.signal == Signal.BUY
            else "above_trend_return_probability"
        )

        return SignalDecision(
            signal=base_decision.signal,
            confidence=base_decision.confidence,
            reason=base_decision.reason,
            z_score=base_decision.z_score,
            trend_slope=base_decision.trend_slope,
            energy=base_decision.energy,
            noise=base_decision.noise,
            metadata=meta,
        )

    def _evaluate_signal(
        self, point: WaveletPoint, direction: Signal
    ) -> SignalDecision:
        """Evaluate a candidate signal through all filters.

        Parameters
        ----------
        point : WaveletPoint
            Current wavelet features.
        direction : Signal
            Proposed direction (BUY or SELL).

        Returns
        -------
        SignalDecision
            Confirmed signal or HOLD if filters reject.
        """
        # Slope filter
        if self._config.slope_filter_enabled:
            if not check_slope_filter(point, direction):
                return _hold_decision(point, f"{direction.value}_rejected_by_slope_filter")

        # Energy filter
        if self._config.energy_filter_enabled:
            if not check_energy_filter(point, self._config):
                return _hold_decision(point, f"{direction.value}_rejected_by_energy_filter")

        # Noise filter
        if self._config.noise_filter_enabled:
            if not check_noise_filter(point, self._config):
                return _hold_decision(point, f"{direction.value}_rejected_by_noise_filter")

        # Compute confidence
        confidence = compute_confidence(point, self._config, direction)

        # Minimum confidence check
        if confidence < self._config.min_confidence:
            return _hold_decision(point, f"{direction.value}_below_min_confidence")

        reason = (
            f"{direction.value}_z{point.z_score:.4f}"
            f"_slope{point.slope:.6f}"
            f"_conf{confidence:.4f}"
        )

        return SignalDecision(
            signal=direction,
            confidence=confidence,
            reason=reason,
            z_score=point.z_score,
            trend_slope=point.slope,
            energy=point.energy,
            noise=point.noise,
        )
