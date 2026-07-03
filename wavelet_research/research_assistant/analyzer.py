"""Experiment analysis: identifies patterns, explains failures, recommends experiments.

Read-only analysis — never modifies experiment results.
"""

from __future__ import annotations

import logging

import numpy as np

from wavelet_research.optimizer.config import ConstraintConfig
from wavelet_research.optimizer.models import ScoredConfig
from wavelet_research.orchestrator.results import ExperimentReport
from wavelet_research.research_assistant.models import (
    AnalysisReport,
    FailureExplanation,
    Recommendation,
)

logger = logging.getLogger(__name__)


class ExperimentAnalyzer:
    """Analyzes completed experiment results.

    Provides failure explanations and recommendations
    without modifying any experiment data.

    Parameters
    ----------
    constraints : ConstraintConfig
        Constraints used for filtering.
    """

    def __init__(self, constraints: ConstraintConfig) -> None:
        self._constraints = constraints

    def analyze(
        self, reports: list[ExperimentReport]
    ) -> AnalysisReport:
        """Analyze a batch of experiment reports.

        Parameters
        ----------
        reports : list[ExperimentReport]
            Completed experiment reports.

        Returns
        -------
        AnalysisReport
            Complete analysis.

        Raises
        ------
        ValueError
            If reports is empty.
        """
        if not reports:
            raise ValueError("Reports must not be empty.")

        passed = []
        failed = []
        explanations: list[FailureExplanation] = []

        for r in reports:
            reasons = self._check_failures(r)
            if reasons:
                failed.append(r)
                explanations.append(FailureExplanation(
                    config_identifier=r.config.identifier,
                    reasons=tuple(reasons),
                    suggestion=self._suggest_fix(reasons),
                ))
            else:
                passed.append(r)

        pf_values = [r.report.profit_factor for r in reports
                     if np.isfinite(r.report.profit_factor)]
        exp_values = [r.report.expectancy for r in reports]
        dd_values = [r.report.max_drawdown for r in reports]

        best_pf = max(pf_values) if pf_values else 0.0
        best_exp = max(exp_values) if exp_values else 0.0
        worst_dd = max(dd_values) if dd_values else 0.0

        recommendations = self._generate_recommendations(passed, failed)
        insights = self._generate_insights(reports, passed, failed)

        return AnalysisReport(
            total_experiments=len(reports),
            passed_count=len(passed),
            failed_count=len(failed),
            best_profit_factor=best_pf,
            best_expectancy=best_exp,
            worst_drawdown=worst_dd,
            failure_explanations=tuple(explanations),
            recommendations=tuple(recommendations),
            insights=tuple(insights),
        )

    def _check_failures(self, report: ExperimentReport) -> list[str]:
        """Check why a report fails constraints."""
        reasons: list[str] = []
        r = report.report
        c = self._constraints

        if r.trades < c.min_trades:
            reasons.append(
                f"Too few trades: {r.trades} < {c.min_trades}"
            )
        if r.max_drawdown > c.max_drawdown:
            reasons.append(
                f"Drawdown too high: {r.max_drawdown:.2f} > {c.max_drawdown:.2f}"
            )
        if r.profit_factor < c.min_profit_factor:
            reasons.append(
                f"Profit factor too low: {r.profit_factor:.2f} < {c.min_profit_factor:.2f}"
            )
        if r.expectancy < c.min_expectancy:
            reasons.append(
                f"Expectancy too low: {r.expectancy:.2f} < {c.min_expectancy:.2f}"
            )
        return reasons

    def _suggest_fix(self, reasons: list[str]) -> str:
        """Generate a suggestion based on failure reasons."""
        suggestions: list[str] = []
        for reason in reasons:
            if "Too few trades" in reason:
                suggestions.append("Lower z-score thresholds to generate more signals")
            elif "Drawdown too high" in reason:
                suggestions.append("Reduce max_hold_ticks or add tighter stop loss")
            elif "Profit factor too low" in reason:
                suggestions.append("Increase z-score thresholds for higher-quality entries")
            elif "Expectancy too low" in reason:
                suggestions.append("Try different wavelet or exit strategy")
        return "; ".join(suggestions) if suggestions else "Review parameters"

    def _generate_recommendations(
        self,
        passed: list[ExperimentReport],
        failed: list[ExperimentReport],
    ) -> list[Recommendation]:
        """Generate recommended new experiments."""
        recommendations: list[Recommendation] = []

        if not passed and failed:
            recommendations.append(Recommendation(
                description="Broaden search with lower thresholds",
                rationale="All experiments failed constraints; signals may be too restrictive",
                parameter_changes={"buy_z_threshold": 1.0, "sell_z_threshold": 1.0},
                priority="high",
            ))

        if passed:
            best = max(passed, key=lambda r: r.report.profit_factor
                       if np.isfinite(r.report.profit_factor) else 0.0)
            wc = best.config.wavelet_config
            recommendations.append(Recommendation(
                description=f"Explore variants around best config ({best.config.identifier})",
                rationale="Best performing config found; search nearby parameter space",
                parameter_changes={
                    "wavelet": wc.wavelet,
                    "window": wc.window,
                },
                priority="medium",
            ))

        return recommendations

    def _generate_insights(
        self,
        all_reports: list[ExperimentReport],
        passed: list[ExperimentReport],
        failed: list[ExperimentReport],
    ) -> list[str]:
        """Generate key insights."""
        insights: list[str] = []

        pass_rate = len(passed) / len(all_reports) * 100
        insights.append(f"Pass rate: {pass_rate:.1f}% ({len(passed)}/{len(all_reports)})")

        if passed:
            pnls = [r.report.total_pnl for r in passed]
            insights.append(
                f"Passed experiments PnL range: [{min(pnls):.2f}, {max(pnls):.2f}]"
            )

        if failed:
            insights.append(
                f"Most common failure: {self._most_common_failure(failed)}"
            )

        return insights

    def _most_common_failure(self, failed: list[ExperimentReport]) -> str:
        """Identify the most common failure reason."""
        reason_counts: dict[str, int] = {}
        for r in failed:
            reasons = self._check_failures(r)
            for reason in reasons:
                category = reason.split(":")[0]
                reason_counts[category] = reason_counts.get(category, 0) + 1
        if not reason_counts:
            return "Unknown"
        return max(reason_counts, key=reason_counts.get)  # type: ignore[arg-type]
