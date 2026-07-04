"""Individual filter rule functions. Pure functions — no state."""

from __future__ import annotations

from wavelet_research.deviation.models import DeviationPoint
from wavelet_research.deviation_stats.models import DeviationQueryResult
from wavelet_research.filters.models import FilterConfig, FilterReason
from wavelet_research.trend_quality.models import TrendQualityState


def check_low_sample_size(
    stats: DeviationQueryResult, config: FilterConfig
) -> bool:
    """Return True (blocked) when historical sample is too small."""
    return stats.sample_size < config.min_sample_size


def check_low_return_probability(
    stats: DeviationQueryResult, config: FilterConfig
) -> bool:
    """Return True when return-to-trend probability is below threshold."""
    return (
        stats.sample_size > 0
        and stats.return_to_trend_probability < config.min_return_probability
    )


def check_high_adverse_excursion(
    stats: DeviationQueryResult, config: FilterConfig
) -> bool:
    """Return True when expected adverse excursion is too large."""
    return (
        stats.sample_size > 0
        and stats.expected_adverse_excursion > config.max_adverse_excursion
    )


def check_trend_unstable(state: TrendQualityState) -> bool:
    """Return True when the trend quality state is unstable."""
    return not state.is_stable


def check_spread_too_wide(dp: DeviationPoint, config: FilterConfig) -> bool:
    """Return True when spread exceeds the configured maximum."""
    return dp.spread > config.max_spread


def check_deviation_too_small(dp: DeviationPoint, config: FilterConfig) -> bool:
    """Return True when absolute z-score is below minimum."""
    return abs(dp.z_score) < config.min_normalized_deviation


def check_trend_too_flat(dp: DeviationPoint, config: FilterConfig) -> bool:
    """Return True when trend slope is near zero."""
    return abs(dp.trend_slope) < config.min_slope_abs


def check_cooldown(bars_since_last_signal: int, config: FilterConfig) -> bool:
    """Return True when still within cooldown period after last signal."""
    return bars_since_last_signal < config.cooldown_bars
