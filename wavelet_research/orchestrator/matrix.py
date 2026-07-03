"""Parameter matrix generation for research experiments.

Generates all combinations of configurable parameters
as a list of PipelineConfig instances.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Sequence

from wavelet_research.backtest.config import BacktestConfig, ExitStrategy
from wavelet_research.engine.config import WaveletEngineConfig
from wavelet_research.orchestrator.config import PipelineConfig
from wavelet_research.signal.config import SignalConfig


@dataclass(frozen=True)
class ParameterMatrix:
    """Generates PipelineConfig combinations from parameter lists.

    Parameters
    ----------
    wavelets : Sequence[str]
        Wavelet families to test.
    windows : Sequence[int]
        Window sizes to test.
    levels : Sequence[int]
        Decomposition levels to test.
    volatility_windows : Sequence[int]
        Volatility windows to test.
    buy_z_thresholds : Sequence[float]
        BUY z-score thresholds to test.
    sell_z_thresholds : Sequence[float]
        SELL z-score thresholds to test.
    exit_strategies : Sequence[ExitStrategy]
        Exit strategies to test.
    max_hold_ticks_list : Sequence[int]
        Max hold periods to test.
    slope_filter_options : Sequence[bool]
        Slope filter on/off to test.
    backtest_config : BacktestConfig
        Base backtest config (for costs and other fixed params).

    Raises
    ------
    ValueError
        If any parameter list is empty.
    """

    wavelets: Sequence[str] = ("haar",)
    windows: Sequence[int] = (256,)
    levels: Sequence[int] = (2,)
    volatility_windows: Sequence[int] = (256,)
    buy_z_thresholds: Sequence[float] = (2.0,)
    sell_z_thresholds: Sequence[float] = (2.0,)
    exit_strategies: Sequence[ExitStrategy] = (ExitStrategy.MAX_HOLD,)
    max_hold_ticks_list: Sequence[int] = (100,)
    slope_filter_options: Sequence[bool] = (True,)
    backtest_config: BacktestConfig = field(default_factory=BacktestConfig)

    def __post_init__(self) -> None:
        """Validate that all parameter lists are non-empty."""
        if not self.wavelets:
            raise ValueError("wavelets must not be empty")
        if not self.windows:
            raise ValueError("windows must not be empty")
        if not self.levels:
            raise ValueError("levels must not be empty")
        if not self.buy_z_thresholds:
            raise ValueError("buy_z_thresholds must not be empty")
        if not self.sell_z_thresholds:
            raise ValueError("sell_z_thresholds must not be empty")

    def generate(self) -> list[PipelineConfig]:
        """Generate all parameter combinations.

        Returns
        -------
        list[PipelineConfig]
            All valid pipeline configurations.
        """
        configs: list[PipelineConfig] = []

        combinations = itertools.product(
            self.wavelets,
            self.windows,
            self.levels,
            self.volatility_windows,
            self.buy_z_thresholds,
            self.sell_z_thresholds,
            self.exit_strategies,
            self.max_hold_ticks_list,
            self.slope_filter_options,
        )

        for (
            wavelet, window, level, vol_window,
            buy_z, sell_z,
            exit_strat, max_hold, slope_filter,
        ) in combinations:
            wavelet_cfg = WaveletEngineConfig(
                wavelet=wavelet,
                window=window,
                level=level,
                volatility_window=vol_window,
            )
            signal_cfg = SignalConfig(
                buy_z_threshold=buy_z,
                sell_z_threshold=sell_z,
                slope_filter_enabled=slope_filter,
            )
            backtest_cfg = BacktestConfig(
                pip_size=self.backtest_config.pip_size,
                exit_strategy=exit_strat,
                max_hold_ticks=max_hold,
                take_profit_pips=self.backtest_config.take_profit_pips,
                stop_loss_pips=self.backtest_config.stop_loss_pips,
                commission_pips=self.backtest_config.commission_pips,
                slippage_pips=self.backtest_config.slippage_pips,
            )
            configs.append(PipelineConfig(
                wavelet_config=wavelet_cfg,
                signal_config=signal_cfg,
                backtest_config=backtest_cfg,
            ))

        return configs

    @property
    def total_combinations(self) -> int:
        """Total number of parameter combinations."""
        return (
            len(self.wavelets)
            * len(self.windows)
            * len(self.levels)
            * len(self.volatility_windows)
            * len(self.buy_z_thresholds)
            * len(self.sell_z_thresholds)
            * len(self.exit_strategies)
            * len(self.max_hold_ticks_list)
            * len(self.slope_filter_options)
        )
