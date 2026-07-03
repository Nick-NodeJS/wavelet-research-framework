"""Monte Carlo trade-order simulation.

Shuffles trade PnL sequences to assess strategy robustness.
Deterministic when seeded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wavelet_research.validation.metrics import (
    compute_sharpe,
    compute_stability_score,
)


@dataclass(frozen=True)
class MonteCarloResult:
    """Result of a single Monte Carlo simulation.

    Parameters
    ----------
    total_pnl : float
        Total PnL of shuffled sequence.
    max_drawdown : float
        Max drawdown of shuffled sequence.
    sharpe_ratio : float
        Sharpe ratio of shuffled sequence.
    final_equity : float
        Final equity value.
    """

    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    final_equity: float


def _compute_drawdown(equity: np.ndarray) -> float:
    """Compute maximum drawdown from equity curve."""
    if len(equity) == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity)
    drawdown = running_max - equity
    return float(np.max(drawdown))


def run_monte_carlo(
    pnl_values: list[float],
    n_simulations: int = 1000,
    seed: int = 42,
) -> list[MonteCarloResult]:
    """Run Monte Carlo trade-order simulations.

    Shuffles the trade PnL sequence repeatedly to generate
    a distribution of possible outcomes.

    Parameters
    ----------
    pnl_values : list[float]
        Original per-trade PnL values.
    n_simulations : int
        Number of Monte Carlo iterations.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[MonteCarloResult]
        Results for each simulation.

    Raises
    ------
    ValueError
        If pnl_values is empty or n_simulations < 1.
    """
    if not pnl_values:
        raise ValueError("pnl_values must not be empty.")
    if n_simulations < 1:
        raise ValueError(f"n_simulations must be >= 1, got {n_simulations}")

    rng = np.random.default_rng(seed)
    pnl = np.asarray(pnl_values, dtype=np.float64)
    results: list[MonteCarloResult] = []

    for _ in range(n_simulations):
        shuffled = rng.permutation(pnl)
        equity = np.cumsum(shuffled)
        total = float(equity[-1])
        dd = _compute_drawdown(equity)
        sharpe = compute_sharpe(shuffled)

        results.append(MonteCarloResult(
            total_pnl=total,
            max_drawdown=dd,
            sharpe_ratio=sharpe,
            final_equity=total,
        ))

    return results
