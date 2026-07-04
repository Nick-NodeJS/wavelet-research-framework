"""Final statistical gate for strategy readiness (Story 29).

Produces a single PASS / REVIEW / FAIL decision from backtest metrics,
walk-forward stability, and paper trading consistency.
No ML, no subjectivity — pure deterministic threshold checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class GateDecision(str, Enum):
    """Final gate verdict."""

    PASS = "PASS"
    REVIEW = "REVIEW"
    FAIL = "FAIL"


@dataclass(frozen=True)
class GateConfig:
    """Thresholds for the final gate.

    Parameters
    ----------
    min_trades : int
        Minimum trade count across the evaluation period.
    min_profit_factor : float
        Minimum profit factor.
    min_expectancy : float
        Minimum average PnL per trade.
    max_drawdown_pct : float
        Maximum drawdown as fraction of gross profit.
    min_monte_carlo_survival : float
        Minimum Monte Carlo survival probability [0..1].
    max_avg_holding_bars : float
        Maximum average holding period in bars.
    min_walk_forward_stability : float
        Minimum OOS / IS ratio [0..1].
    min_paper_consistency : float
        Minimum agreement between backtest and paper directions [0..1].
    """

    min_trades: int = 50
    min_profit_factor: float = 1.3
    min_expectancy: float = 0.0
    max_drawdown_pct: float = 0.30
    min_monte_carlo_survival: float = 0.70
    max_avg_holding_bars: float = 30.0
    min_walk_forward_stability: float = 0.60
    min_paper_consistency: float = 0.70


@dataclass(frozen=True)
class GateMetrics:
    """Metrics fed into the final gate.

    Parameters
    ----------
    total_trades : int
    profit_factor : float
    expectancy : float
    max_drawdown : float
    gross_profit : float
    avg_holding_bars : float
    monte_carlo_survival : float
        Fraction of MC runs with positive PnL.
    walk_forward_stability : float
        OOS profit factor / IS profit factor ratio.
    paper_consistency : float
        Fraction of signals consistent between backtest and paper run.
    """

    total_trades: int
    profit_factor: float
    expectancy: float
    max_drawdown: float
    gross_profit: float
    avg_holding_bars: float
    monte_carlo_survival: float
    walk_forward_stability: float
    paper_consistency: float


@dataclass(frozen=True)
class GateResult:
    """Result of the final gate evaluation.

    Parameters
    ----------
    decision : GateDecision
        PASS, REVIEW, or FAIL.
    reasons : tuple[str, ...]
        Failure or review reasons.
    metrics : GateMetrics
        Input metrics used.
    next_action : str
        Recommended next step.
    """

    decision: GateDecision
    reasons: tuple[str, ...]
    metrics: GateMetrics
    next_action: str

    def to_dict(self) -> dict[str, object]:
        """Serialise to JSON-compatible dict."""
        return {
            "decision": self.decision.value,
            "reasons": list(self.reasons),
            "metrics": {
                "total_trades": self.metrics.total_trades,
                "profit_factor": self.metrics.profit_factor,
                "expectancy": self.metrics.expectancy,
                "max_drawdown": self.metrics.max_drawdown,
                "avg_holding_bars": self.metrics.avg_holding_bars,
                "monte_carlo_survival": self.metrics.monte_carlo_survival,
                "walk_forward_stability": self.metrics.walk_forward_stability,
                "paper_consistency": self.metrics.paper_consistency,
            },
            "next_action": self.next_action,
        }


def evaluate_gate(metrics: GateMetrics, config: GateConfig) -> GateResult:
    """Evaluate the final statistical gate.

    Parameters
    ----------
    metrics : GateMetrics
        Strategy performance metrics.
    config : GateConfig
        Gate thresholds.

    Returns
    -------
    GateResult
        Single PASS / REVIEW / FAIL decision.
    """
    hard_failures: list[str] = []
    reviews: list[str] = []

    if metrics.total_trades < config.min_trades:
        hard_failures.append(
            f"insufficient_sample: {metrics.total_trades} < {config.min_trades}"
        )

    if metrics.profit_factor < config.min_profit_factor:
        hard_failures.append(
            f"low_profit_factor: {metrics.profit_factor:.3f} < {config.min_profit_factor}"
        )

    if metrics.expectancy < config.min_expectancy:
        hard_failures.append(
            f"negative_expectancy: {metrics.expectancy:.4f}"
        )

    drawdown_pct = (
        metrics.max_drawdown / metrics.gross_profit
        if metrics.gross_profit > 0
        else float("inf")
    )
    if drawdown_pct > config.max_drawdown_pct:
        hard_failures.append(
            f"excessive_drawdown: {drawdown_pct:.2%} > {config.max_drawdown_pct:.2%}"
        )

    if metrics.monte_carlo_survival < config.min_monte_carlo_survival:
        hard_failures.append(
            f"low_mc_survival: {metrics.monte_carlo_survival:.2%} < {config.min_monte_carlo_survival:.2%}"
        )

    if metrics.walk_forward_stability < config.min_walk_forward_stability:
        reviews.append(
            f"low_wf_stability: {metrics.walk_forward_stability:.2f} < {config.min_walk_forward_stability}"
        )

    if metrics.avg_holding_bars > config.max_avg_holding_bars:
        reviews.append(
            f"long_holding_period: {metrics.avg_holding_bars:.1f} bars"
        )

    if metrics.paper_consistency < config.min_paper_consistency:
        reviews.append(
            f"paper_inconsistency: {metrics.paper_consistency:.2%} < {config.min_paper_consistency:.2%}"
        )

    if hard_failures:
        decision = GateDecision.FAIL
        next_action = "reject_config"
        reasons = tuple(hard_failures + reviews)
    elif reviews:
        decision = GateDecision.REVIEW
        next_action = "paper_trade_more"
        reasons = tuple(reviews)
    else:
        decision = GateDecision.PASS
        next_action = "allow_small_live"
        reasons = ()

    return GateResult(
        decision=decision,
        reasons=reasons,
        metrics=metrics,
        next_action=next_action,
    )
