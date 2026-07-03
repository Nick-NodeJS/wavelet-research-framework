"""Search strategies: grid and random."""

from __future__ import annotations

import itertools

import numpy as np

from wavelet_research.backtest.config import BacktestConfig, ExitStrategy
from wavelet_research.engine.config import WaveletEngineConfig
from wavelet_research.optimizer.config import OptimizerConfig, SearchMethod
from wavelet_research.orchestrator.config import PipelineConfig
from wavelet_research.signal.config import SignalConfig


def generate_grid_configs(config: OptimizerConfig) -> list[PipelineConfig]:
    """Generate all parameter combinations via grid search.

    Parameters
    ----------
    config : OptimizerConfig
        Optimizer configuration.

    Returns
    -------
    list[PipelineConfig]
        All grid combinations.
    """
    configs: list[PipelineConfig] = []

    combinations = itertools.product(
        config.wavelets,
        config.windows,
        config.levels,
        config.volatility_windows,
        config.buy_z_thresholds,
        config.sell_z_thresholds,
        config.exit_strategies,
        config.max_hold_ticks_list,
        config.slope_filter_options,
    )

    for (
        wavelet, window, level, vol_window,
        buy_z, sell_z, exit_strat, max_hold, slope_filter,
    ) in combinations:
        wavelet_cfg = WaveletEngineConfig(
            wavelet=wavelet, window=window,
            level=level, volatility_window=vol_window,
        )
        signal_cfg = SignalConfig(
            buy_z_threshold=buy_z, sell_z_threshold=sell_z,
            slope_filter_enabled=slope_filter,
        )
        backtest_cfg = BacktestConfig(
            pip_size=config.backtest_config.pip_size,
            exit_strategy=exit_strat,
            max_hold_ticks=max_hold,
            take_profit_pips=config.backtest_config.take_profit_pips,
            stop_loss_pips=config.backtest_config.stop_loss_pips,
            commission_pips=config.backtest_config.commission_pips,
            slippage_pips=config.backtest_config.slippage_pips,
        )
        configs.append(PipelineConfig(
            wavelet_config=wavelet_cfg,
            signal_config=signal_cfg,
            backtest_config=backtest_cfg,
        ))

    return configs


def generate_random_configs(config: OptimizerConfig) -> list[PipelineConfig]:
    """Generate random parameter combinations.

    Parameters
    ----------
    config : OptimizerConfig
        Optimizer configuration.

    Returns
    -------
    list[PipelineConfig]
        Random sample of configurations (up to max_iterations).
    """
    rng = np.random.default_rng(config.seed)
    configs: list[PipelineConfig] = []

    for _ in range(config.max_iterations):
        wavelet = str(rng.choice(list(config.wavelets)))
        window = int(rng.choice(list(config.windows)))
        level = int(rng.choice(list(config.levels)))
        vol_window = int(rng.choice(list(config.volatility_windows)))
        buy_z = float(rng.choice(list(config.buy_z_thresholds)))
        sell_z = float(rng.choice(list(config.sell_z_thresholds)))
        exit_idx = int(rng.integers(0, len(config.exit_strategies)))
        exit_strat = list(config.exit_strategies)[exit_idx]
        max_hold = int(rng.choice(list(config.max_hold_ticks_list)))
        slope_idx = int(rng.integers(0, len(config.slope_filter_options)))
        slope_filter = list(config.slope_filter_options)[slope_idx]

        wavelet_cfg = WaveletEngineConfig(
            wavelet=wavelet, window=window,
            level=level, volatility_window=vol_window,
        )
        signal_cfg = SignalConfig(
            buy_z_threshold=buy_z, sell_z_threshold=sell_z,
            slope_filter_enabled=slope_filter,
        )
        backtest_cfg = BacktestConfig(
            pip_size=config.backtest_config.pip_size,
            exit_strategy=exit_strat,
            max_hold_ticks=max_hold,
            take_profit_pips=config.backtest_config.take_profit_pips,
            stop_loss_pips=config.backtest_config.stop_loss_pips,
            commission_pips=config.backtest_config.commission_pips,
            slippage_pips=config.backtest_config.slippage_pips,
        )
        configs.append(PipelineConfig(
            wavelet_config=wavelet_cfg,
            signal_config=signal_cfg,
            backtest_config=backtest_cfg,
        ))

    return configs
