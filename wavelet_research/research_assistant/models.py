"""Domain models for the AI Research Assistant."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FailureExplanation:
    """Explanation for why an experiment failed constraints.

    Parameters
    ----------
    config_identifier : str
        Config identifier.
    reasons : tuple[str, ...]
        List of failure reasons.
    suggestion : str
        Suggested fix.
    """

    config_identifier: str
    reasons: tuple[str, ...]
    suggestion: str


@dataclass(frozen=True)
class Recommendation:
    """Recommended new experiment to run.

    Parameters
    ----------
    description : str
        Description of the recommendation.
    rationale : str
        Why this experiment is recommended.
    parameter_changes : dict[str, object]
        Suggested parameter changes.
    priority : str
        Priority level: 'high', 'medium', 'low'.
    """

    description: str
    rationale: str
    parameter_changes: dict[str, object]
    priority: str


@dataclass(frozen=True)
class AnalysisReport:
    """Complete analysis report for experiment results.

    Parameters
    ----------
    total_experiments : int
        Total experiments analyzed.
    passed_count : int
        Experiments passing constraints.
    failed_count : int
        Experiments failing constraints.
    best_profit_factor : float
        Best profit factor observed.
    best_expectancy : float
        Best expectancy observed.
    worst_drawdown : float
        Worst drawdown observed.
    failure_explanations : tuple[FailureExplanation, ...]
        Explanations for failures.
    recommendations : tuple[Recommendation, ...]
        Recommended next experiments.
    insights : tuple[str, ...]
        Key insights from analysis.
    """

    total_experiments: int
    passed_count: int
    failed_count: int
    best_profit_factor: float
    best_expectancy: float
    worst_drawdown: float
    failure_explanations: tuple[FailureExplanation, ...]
    recommendations: tuple[Recommendation, ...]
    insights: tuple[str, ...]


@dataclass(frozen=True)
class ComparisonReport:
    """Comparison between two parameter sets.

    Parameters
    ----------
    config_a_identifier : str
        First config identifier.
    config_b_identifier : str
        Second config identifier.
    pnl_difference : float
        PnL difference (A - B).
    profit_factor_difference : float
        Profit factor difference.
    drawdown_difference : float
        Drawdown difference.
    winner : str
        Which config is better overall.
    explanation : str
        Why the winner is better.
    """

    config_a_identifier: str
    config_b_identifier: str
    pnl_difference: float
    profit_factor_difference: float
    drawdown_difference: float
    winner: str
    explanation: str
