"""Configuration for the experiment pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from wavelet_research.backtest.config import BacktestConfig
from wavelet_research.engine.config import WaveletEngineConfig
from wavelet_research.signal.config import SignalConfig


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable configuration for a single experiment pipeline.

    Combines all three engine configurations into a single unit.

    Parameters
    ----------
    wavelet_config : WaveletEngineConfig
        Configuration for the Causal Wavelet Engine.
    signal_config : SignalConfig
        Configuration for the Signal Engine.
    backtest_config : BacktestConfig
        Configuration for the Backtesting Engine.
    """

    wavelet_config: WaveletEngineConfig
    signal_config: SignalConfig
    backtest_config: BacktestConfig

    @property
    def identifier(self) -> str:
        """Unique string identifier for this pipeline configuration."""
        wc = self.wavelet_config
        sc = self.signal_config
        bc = self.backtest_config
        return (
            f"{wc.wavelet}_W{wc.window}_L{wc.level}"
            f"_BZ{sc.buy_z_threshold}_SZ{sc.sell_z_threshold}"
            f"_E{bc.exit_strategy.value}_H{bc.max_hold_ticks}"
        )
