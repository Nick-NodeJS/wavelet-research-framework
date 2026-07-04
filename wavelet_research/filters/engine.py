"""No-trade filter engine — evaluates all filter rules."""

from __future__ import annotations

from wavelet_research.deviation.models import DeviationPoint
from wavelet_research.deviation_stats.models import DeviationQueryResult
from wavelet_research.filters.models import (
    FilterConfig,
    FilterReason,
    FilterResult,
    FilterSeverity,
)
from wavelet_research.filters.rules import (
    check_cooldown,
    check_deviation_too_small,
    check_high_adverse_excursion,
    check_low_return_probability,
    check_low_sample_size,
    check_spread_too_wide,
    check_trend_too_flat,
    check_trend_unstable,
)
from wavelet_research.trend_quality.models import TrendQualityState

_BLOCK_REASONS = frozenset({
    FilterReason.LOW_SAMPLE_SIZE,
    FilterReason.TREND_UNSTABLE,
    FilterReason.SPREAD_TOO_WIDE,
    FilterReason.RECENT_SIGNAL_COOLDOWN,
    FilterReason.HIGH_ADVERSE_EXCURSION,
})
_WARNING_REASONS = frozenset({
    FilterReason.LOW_RETURN_PROBABILITY,
    FilterReason.DEVIATION_TOO_SMALL,
    FilterReason.TREND_TOO_FLAT,
})


class FilterEngine:
    """Evaluates no-trade filter rules and returns a FilterResult.

    Parameters
    ----------
    config : FilterConfig
        Filter configuration thresholds.
    """

    def __init__(self, config: FilterConfig) -> None:
        self._config = config
        self._bars_since_last_signal: int = 999

    @property
    def config(self) -> FilterConfig:
        """Filter configuration."""
        return self._config

    def evaluate(
        self,
        deviation_point: DeviationPoint,
        trend_quality_state: TrendQualityState,
        stats: DeviationQueryResult,
    ) -> FilterResult:
        """Evaluate all filter rules for the current market state.

        Parameters
        ----------
        deviation_point : DeviationPoint
            Current deviation measures.
        trend_quality_state : TrendQualityState
            Current trend quality.
        stats : DeviationQueryResult
            Historical stats for this market state.

        Returns
        -------
        FilterResult
            Combined filter decision with all triggered reasons.
        """
        triggered: list[FilterReason] = []

        if check_low_sample_size(stats, self._config):
            triggered.append(FilterReason.LOW_SAMPLE_SIZE)
        if check_low_return_probability(stats, self._config):
            triggered.append(FilterReason.LOW_RETURN_PROBABILITY)
        if check_high_adverse_excursion(stats, self._config):
            triggered.append(FilterReason.HIGH_ADVERSE_EXCURSION)
        if check_trend_unstable(trend_quality_state):
            triggered.append(FilterReason.TREND_UNSTABLE)
        if check_spread_too_wide(deviation_point, self._config):
            triggered.append(FilterReason.SPREAD_TOO_WIDE)
        if check_deviation_too_small(deviation_point, self._config):
            triggered.append(FilterReason.DEVIATION_TOO_SMALL)
        if check_trend_too_flat(deviation_point, self._config):
            triggered.append(FilterReason.TREND_TOO_FLAT)
        if check_cooldown(self._bars_since_last_signal, self._config):
            triggered.append(FilterReason.RECENT_SIGNAL_COOLDOWN)

        blocking = [r for r in triggered if r in _BLOCK_REASONS]
        can_trade = len(blocking) == 0

        if blocking:
            severity = FilterSeverity.BLOCK
        elif triggered:
            severity = FilterSeverity.WARNING
        else:
            severity = FilterSeverity.PASS

        return FilterResult(
            can_trade=can_trade,
            reasons=tuple(triggered),
            severity=severity,
        )

    def record_signal(self) -> None:
        """Notify the engine that a signal was emitted (resets cooldown)."""
        self._bars_since_last_signal = 0

    def tick(self) -> None:
        """Advance one bar (increments cooldown counter)."""
        self._bars_since_last_signal += 1
