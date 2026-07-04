"""Domain models for the trend quality audit."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class TrendVerdict(str, Enum):
    """Overall audit verdict."""

    PASS = "pass"
    REVIEW = "review"
    FAIL = "fail"


@dataclass(frozen=True)
class TrendQualityMetrics:
    """Raw metrics from the trend audit.

    Parameters
    ----------
    repaint_max : float
        Maximum absolute change in a previously emitted trend value.
    repaint_mean : float
        Mean absolute change across all recheck comparisons.
    trend_lag_estimate : int
        Estimated lag in bars between price move and trend response.
    trend_smoothness : float
        Ratio of trend std-dev to price std-dev (lower = smoother).
    trend_direction_stability : float
        Fraction of consecutive bars where slope sign is unchanged [0..1].
    price_cross_frequency : float
        Fraction of bars where price crosses trend [0..1].
    mean_abs_distance : float
        Mean absolute distance between price and trend.
    normalized_mean_abs_distance : float
        mean_abs_distance / mean_price_level.
    """

    repaint_max: float
    repaint_mean: float
    trend_lag_estimate: int
    trend_smoothness: float
    trend_direction_stability: float
    price_cross_frequency: float
    mean_abs_distance: float
    normalized_mean_abs_distance: float


@dataclass(frozen=True)
class TrendQualityReport:
    """Complete trend quality audit report.

    Parameters
    ----------
    metrics : TrendQualityMetrics
        Computed quality metrics.
    trend_quality_score : float
        Composite score in [0.0, 1.0]. Higher is better.
    recommendation : TrendVerdict
        Overall verdict.
    repaint_max : float
        Convenience alias for metrics.repaint_max.
    lag_estimate_bars : int
        Convenience alias for metrics.trend_lag_estimate.
    smoothness : float
        Convenience alias for metrics.trend_smoothness.
    cross_frequency : float
        Convenience alias for metrics.price_cross_frequency.
    """

    metrics: TrendQualityMetrics
    trend_quality_score: float
    recommendation: TrendVerdict
    repaint_max: float
    lag_estimate_bars: int
    smoothness: float
    cross_frequency: float

    def to_dict(self) -> dict[str, object]:
        """Serialise to JSON-compatible dictionary.

        Returns
        -------
        dict[str, object]
            Plain key-value pairs matching the story's output schema.
        """
        return {
            "trend_quality_score": self.trend_quality_score,
            "repaint_max": self.repaint_max,
            "lag_estimate_bars": self.lag_estimate_bars,
            "smoothness": self.smoothness,
            "cross_frequency": self.cross_frequency,
            "recommendation": self.recommendation.value,
            "metrics": {
                "repaint_mean": self.metrics.repaint_mean,
                "trend_direction_stability": self.metrics.trend_direction_stability,
                "mean_abs_distance": self.metrics.mean_abs_distance,
                "normalized_mean_abs_distance": self.metrics.normalized_mean_abs_distance,
            },
        }


@dataclass(frozen=True)
class TrendQualityState:
    """Live trend quality state for a single bar/tick.

    Used by filters and signal engine.

    Parameters
    ----------
    is_stable : bool
        True when trend is considered reliable.
    repaint_risk : float
        Estimated repaint risk in current conditions.
    slope_consistent : bool
        True when recent slope direction is consistent.
    """

    is_stable: bool
    repaint_risk: float
    slope_consistent: bool
