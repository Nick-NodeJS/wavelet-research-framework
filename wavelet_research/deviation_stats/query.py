"""Deterministic bucket-based query interface for historical deviation stats."""

from __future__ import annotations

import statistics

from wavelet_research.deviation_stats.models import DeviationEvent, DeviationQueryResult

_HIGH_CONFIDENCE = 200
_MEDIUM_CONFIDENCE = 50
_LOW_CONFIDENCE = 10
_DEV_BIN_WIDTH = 0.5


class DeviationStatsIndex:
    """In-memory bucket index for querying historical deviation stats.

    Uses binned normalized_deviation + volatility_bucket for lookup.
    Deterministic — no randomness, no ML.

    Parameters
    ----------
    events : list[DeviationEvent]
        Pre-collected historical events.
    dev_bin_width : float
        Width of the normalized_deviation bucket.
    """

    def __init__(
        self,
        events: list[DeviationEvent],
        dev_bin_width: float = _DEV_BIN_WIDTH,
    ) -> None:
        self._dev_bin_width = dev_bin_width
        self._index: dict[tuple[int, str], list[DeviationEvent]] = {}
        for event in events:
            key = self._make_key(event.normalized_deviation, event.volatility_bucket)
            self._index.setdefault(key, []).append(event)

    def query(
        self,
        normalized_deviation: float,
        volatility_bucket: str = "normal",
        trend_slope: float = 0.0,
    ) -> DeviationQueryResult:
        """Query historical statistics for the current market state.

        Parameters
        ----------
        normalized_deviation : float
            Current z-score.
        volatility_bucket : str
            "low" | "normal" | "high".
        trend_slope : float
            Current trend slope (not used in first version, reserved).

        Returns
        -------
        DeviationQueryResult
            Aggregated statistics. Low confidence when sample < 10.
        """
        key = self._make_key(normalized_deviation, volatility_bucket)
        events = self._index.get(key, [])

        if not events:
            return DeviationQueryResult(
                sample_size=0,
                return_to_trend_probability=0.0,
                median_bars_to_return=0.0,
                expected_return=0.0,
                expected_adverse_excursion=0.0,
                confidence_level="insufficient",
            )

        n = len(events)
        returned = [e for e in events if e.returned_to_trend]
        prob = len(returned) / n
        bars = statistics.median([e.bars_to_return for e in returned]) if returned else 0.0
        exp_ret = sum(e.future_return_5 for e in events) / n
        exp_mae = sum(e.max_adverse_excursion for e in events) / n

        if n >= _HIGH_CONFIDENCE:
            confidence = "high"
        elif n >= _MEDIUM_CONFIDENCE:
            confidence = "medium"
        elif n >= _LOW_CONFIDENCE:
            confidence = "low"
        else:
            confidence = "insufficient"

        return DeviationQueryResult(
            sample_size=n,
            return_to_trend_probability=prob,
            median_bars_to_return=float(bars),
            expected_return=exp_ret,
            expected_adverse_excursion=exp_mae,
            confidence_level=confidence,
        )

    def _make_key(self, normalized_deviation: float, volatility_bucket: str) -> tuple[int, str]:
        bin_idx = int(normalized_deviation / self._dev_bin_width)
        return (bin_idx, volatility_bucket)
