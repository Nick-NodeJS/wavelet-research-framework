"""Configuration for the Parameter Optimization Engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from wavelet_research.backtest.config import BacktestConfig, ExitStrategy


class SearchMethod(str, Enum):
    """Supported search methods."""

    GRID = "grid"
    RANDOM = "random"


@dataclass(frozen=True)
class ObjectiveConfig:
    """Multi-objective scoring configuration.

    Parameters
    ----------
    profit_factor_weight : float
        Weight for profit factor in composite score.
    expectancy_weight : float
        Weight for expectancy.
    drawdown_weight : float
        Weight for max drawdown (penalized).
    stability_weight : float
        Weight for stability score.

    Raises
    ------
    ValueError
        If any weight is negative.
    """

    profit_factor_weight: float = 0.3
    expectancy_weight: float = 0.3
    drawdown_weight: float = 0.2
    stability_weight: float = 0.2

    def __post_init__(self) -> None:
        for name in ("profit_factor_weight", "expectancy_weight",
                     "drawdown_weight", "stability_weight"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")


@dataclass(frozen=True)
class ConstraintConfig:
    """Hard constraints for optimization.

    Parameters
    ----------
    min_trades : int
        Minimum number of trades required.
    max_drawdown : float
        Maximum allowed drawdown in pips.
    min_profit_factor : float
        Minimum profit factor.
    min_expectancy : float
        Minimum expectancy in pips.
    """

    min_trades: int = 5
    max_drawdown: float = 100.0
    min_profit_factor: float = 0.0
    min_expectancy: float = -float("inf")

    def __post_init__(self) -> None:
        if self.min_trades < 0:
            raise ValueError(f"min_trades must be >= 0, got {self.min_trades}")
        if self.max_drawdown <= 0:
            raise ValueError(
                f"max_drawdown must be > 0, got {self.max_drawdown}"
            )


@dataclass(frozen=True)
class OptimizerConfig:
    """Configuration for the Parameter Optimizer.

    Parameters
    ----------
    search_method : SearchMethod
        Search strategy (grid or random).
    wavelets : Sequence[str]
        Wavelet families to search.
    windows : Sequence[int]
        Window sizes to search.
    levels : Sequence[int]
        Decomposition levels to search.
    volatility_windows : Sequence[int]
        Volatility windows to search.
    buy_z_thresholds : Sequence[float]
        BUY z-score thresholds.
    sell_z_thresholds : Sequence[float]
        SELL z-score thresholds.
    exit_strategies : Sequence[ExitStrategy]
        Exit strategies.
    max_hold_ticks_list : Sequence[int]
        Max hold periods.
    slope_filter_options : Sequence[bool]
        Slope filter variants.
    objective : ObjectiveConfig
        Objective function configuration.
    constraints : ConstraintConfig
        Hard constraints.
    max_iterations : int
        Maximum iterations for random search.
    seed : int
        Random seed for reproducibility.
    backtest_config : BacktestConfig
        Base backtest config for costs.

    Raises
    ------
    ValueError
        If configuration is invalid.
    """

    search_method: SearchMethod = SearchMethod.GRID
    wavelets: Sequence[str] = ("haar",)
    windows: Sequence[int] = (256,)
    levels: Sequence[int] = (2,)
    volatility_windows: Sequence[int] = (256,)
    buy_z_thresholds: Sequence[float] = (1.5, 2.0, 2.5)
    sell_z_thresholds: Sequence[float] = (1.5, 2.0, 2.5)
    exit_strategies: Sequence[ExitStrategy] = (ExitStrategy.MAX_HOLD,)
    max_hold_ticks_list: Sequence[int] = (10, 50)
    slope_filter_options: Sequence[bool] = (True, False)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    constraints: ConstraintConfig = field(default_factory=ConstraintConfig)
    max_iterations: int = 100
    seed: int = 42
    backtest_config: BacktestConfig = field(default_factory=BacktestConfig)

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError(
                f"max_iterations must be >= 1, got {self.max_iterations}"
            )
        if not self.wavelets:
            raise ValueError("wavelets must not be empty")
        if not self.windows:
            raise ValueError("windows must not be empty")
