"""Paper trading acceptance validation gate (Story 27)."""

from __future__ import annotations

from dataclasses import dataclass

from wavelet_research.paper_trading.journal import PaperTradeJournal


@dataclass(frozen=True)
class AcceptanceConfig:
    """Configuration for the acceptance gate.

    Parameters
    ----------
    min_trades : int
        Minimum required trade count.
    min_profit_factor : float
        Minimum required profit factor.
    min_expectancy : float
        Minimum expected PnL per trade.
    max_drawdown : float
        Maximum allowed drawdown.
    max_consecutive_losses : int
        Maximum allowed consecutive losses.
    """

    min_trades: int = 30
    min_profit_factor: float = 1.2
    min_expectancy: float = 0.0
    max_drawdown: float = 100.0
    max_consecutive_losses: int = 8


@dataclass(frozen=True)
class AcceptanceResult:
    """Result of the acceptance gate evaluation.

    Parameters
    ----------
    passed : bool
        True when all criteria pass.
    reasons : tuple[str, ...]
        Failure reasons (empty when passed).
    metrics : dict[str, float]
        Computed metrics used in evaluation.
    """

    passed: bool
    reasons: tuple[str, ...]
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        """Serialise to JSON-compatible dict."""
        return {
            "passed": self.passed,
            "reasons": list(self.reasons),
            "metrics": self.metrics,
        }


def evaluate_acceptance(
    journal: PaperTradeJournal,
    config: AcceptanceConfig,
) -> AcceptanceResult:
    """Evaluate journal against acceptance criteria.

    Parameters
    ----------
    journal : PaperTradeJournal
        Completed paper trading journal.
    config : AcceptanceConfig
        Gate thresholds.

    Returns
    -------
    AcceptanceResult
        Pass/fail with reasons.
    """
    reasons: list[str] = []

    n = journal.count
    pf = journal.profit_factor
    exp = journal.expectancy
    dd = journal.max_drawdown

    if n < config.min_trades:
        reasons.append(f"insufficient_trades: {n} < {config.min_trades}")
    if pf < config.min_profit_factor:
        reasons.append(f"low_profit_factor: {pf:.3f} < {config.min_profit_factor}")
    if exp < config.min_expectancy:
        reasons.append(f"negative_expectancy: {exp:.4f} < {config.min_expectancy}")
    if dd > config.max_drawdown:
        reasons.append(f"excessive_drawdown: {dd:.4f} > {config.max_drawdown}")

    # Consecutive losses
    max_cons = _max_consecutive_losses(journal)
    if max_cons > config.max_consecutive_losses:
        reasons.append(
            f"too_many_consecutive_losses: {max_cons} > {config.max_consecutive_losses}"
        )

    return AcceptanceResult(
        passed=len(reasons) == 0,
        reasons=tuple(reasons),
        metrics={
            "trades": float(n),
            "profit_factor": pf,
            "expectancy": exp,
            "max_drawdown": dd,
            "max_consecutive_losses": float(max_cons),
        },
    )


def _max_consecutive_losses(journal: PaperTradeJournal) -> int:
    max_streak = current = 0
    for t in journal.trades:
        if t.pnl < 0:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak
