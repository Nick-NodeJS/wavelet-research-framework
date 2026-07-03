"""Ranking logic for experiment results.

Sorts experiment results by configurable criteria with default priority:
1. Profit Factor (descending)
2. Expectancy (descending)
3. Max Drawdown (ascending — lower is better)
4. Total PnL (descending)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from wavelet_research.research.experiment import ExperimentResult


class SortDirection(Enum):
    """Sort direction for ranking criteria."""

    ASCENDING = "ascending"
    DESCENDING = "descending"


@dataclass(frozen=True)
class RankingCriterion:
    """A single ranking criterion.

    Parameters
    ----------
    metric_name : str
        Name of the metric field on TradeMetrics.
    direction : SortDirection
        Sort direction (ascending or descending).
    """

    metric_name: str
    direction: SortDirection


DEFAULT_RANKING_CRITERIA: tuple[RankingCriterion, ...] = (
    RankingCriterion("profit_factor", SortDirection.DESCENDING),
    RankingCriterion("expectancy", SortDirection.DESCENDING),
    RankingCriterion("max_drawdown", SortDirection.ASCENDING),
    RankingCriterion("total_pnl", SortDirection.DESCENDING),
)


def rank_results(
    results: Sequence[ExperimentResult],
    criteria: Sequence[RankingCriterion] | None = None,
) -> list[ExperimentResult]:
    """Rank experiment results by configurable criteria.

    Parameters
    ----------
    results : Sequence[ExperimentResult]
        List of experiment results to rank.
    criteria : Sequence[RankingCriterion] or None
        Ranking criteria in priority order. Uses DEFAULT_RANKING_CRITERIA if None.

    Returns
    -------
    list[ExperimentResult]
        Sorted results (best first).
    """
    if criteria is None:
        criteria = DEFAULT_RANKING_CRITERIA

    if not results:
        return []

    def sort_key(result: ExperimentResult) -> tuple[float, ...]:
        key_parts: list[float] = []
        for criterion in criteria:
            value = getattr(result.metrics, criterion.metric_name)
            if criterion.direction == SortDirection.DESCENDING:
                key_parts.append(-value)
            else:
                key_parts.append(value)
        return tuple(key_parts)

    return sorted(results, key=sort_key)
