"""Parameter set comparison for the AI Research Assistant.

Read-only comparisons — never modifies experiment results.
"""

from __future__ import annotations

import numpy as np

from wavelet_research.orchestrator.results import ExperimentReport
from wavelet_research.research_assistant.models import ComparisonReport


class ParameterComparator:
    """Compares two experiment configurations.

    Provides detailed comparison of performance metrics
    and identifies the winner with explanation.
    """

    def compare(
        self,
        report_a: ExperimentReport,
        report_b: ExperimentReport,
    ) -> ComparisonReport:
        """Compare two experiment reports.

        Parameters
        ----------
        report_a : ExperimentReport
            First experiment.
        report_b : ExperimentReport
            Second experiment.

        Returns
        -------
        ComparisonReport
            Comparison result.
        """
        ra = report_a.report
        rb = report_b.report

        pnl_diff = ra.total_pnl - rb.total_pnl
        pf_a = ra.profit_factor if np.isfinite(ra.profit_factor) else 0.0
        pf_b = rb.profit_factor if np.isfinite(rb.profit_factor) else 0.0
        pf_diff = pf_a - pf_b
        dd_diff = ra.max_drawdown - rb.max_drawdown

        # Score: higher PnL, higher PF, lower DD
        score_a = pf_a + ra.expectancy - ra.max_drawdown * 0.1
        score_b = pf_b + rb.expectancy - rb.max_drawdown * 0.1

        if score_a >= score_b:
            winner = report_a.config.identifier
            explanation = self._explain_winner(ra, rb, "A")
        else:
            winner = report_b.config.identifier
            explanation = self._explain_winner(rb, ra, "B")

        return ComparisonReport(
            config_a_identifier=report_a.config.identifier,
            config_b_identifier=report_b.config.identifier,
            pnl_difference=pnl_diff,
            profit_factor_difference=pf_diff,
            drawdown_difference=dd_diff,
            winner=winner,
            explanation=explanation,
        )

    def _explain_winner(self, winner_report, loser_report, label: str) -> str:
        """Generate explanation for why one config wins."""
        reasons: list[str] = []

        if winner_report.total_pnl > loser_report.total_pnl:
            reasons.append(
                f"higher PnL ({winner_report.total_pnl:.2f} vs {loser_report.total_pnl:.2f})"
            )
        if winner_report.profit_factor > loser_report.profit_factor:
            reasons.append(
                f"better profit factor ({winner_report.profit_factor:.2f} vs {loser_report.profit_factor:.2f})"
            )
        if winner_report.max_drawdown < loser_report.max_drawdown:
            reasons.append(
                f"lower drawdown ({winner_report.max_drawdown:.2f} vs {loser_report.max_drawdown:.2f})"
            )

        if not reasons:
            reasons.append("marginally better composite score")

        return f"Config {label} wins due to: " + ", ".join(reasons)
