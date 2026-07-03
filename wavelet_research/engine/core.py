"""Core Causal Wavelet Engine for streaming tick processing.

Processes ticks sequentially using only historical data.
Never repaints. Never uses future samples.
"""

from __future__ import annotations

import logging
from typing import Optional

from wavelet_research.engine.buffer import RollingBuffer
from wavelet_research.engine.config import WaveletEngineConfig
from wavelet_research.engine.decomposition import (
    decompose,
    reconstruct_trend,
    safe_decomposition_level,
)
from wavelet_research.engine.features import extract_features
from wavelet_research.engine.models import Tick, WaveletPoint

logger = logging.getLogger(__name__)


class WaveletEngine:
    """Causal Wavelet Engine for streaming tick-by-tick processing.

    Processes each tick using only historical data up to and including
    the current tick. The engine maintains a rolling buffer and produces
    a WaveletPoint for each tick once the buffer is full.

    Parameters
    ----------
    config : WaveletEngineConfig
        Immutable engine configuration.

    Raises
    ------
    ValueError
        If the configuration is invalid.
    """

    def __init__(self, config: WaveletEngineConfig) -> None:
        self._config = config
        self._buffer = RollingBuffer(config.window)
        self._effective_level = safe_decomposition_level(
            config.wavelet, config.window, config.level
        )
        self._tick_count = 0

    @property
    def config(self) -> WaveletEngineConfig:
        """Engine configuration."""
        return self._config

    @property
    def tick_count(self) -> int:
        """Number of ticks processed."""
        return self._tick_count

    @property
    def is_ready(self) -> bool:
        """Whether the engine has accumulated enough history to produce output."""
        return self._buffer.is_full

    def update(self, tick: Tick) -> Optional[WaveletPoint]:
        """Process a single tick and optionally produce a WaveletPoint.

        Returns None until the rolling buffer is full (insufficient history).
        Once the buffer is full, returns a WaveletPoint for every tick.

        This method is strictly causal: only historical data up to and
        including the current tick is used. The engine never repaints.

        Parameters
        ----------
        tick : Tick
            Normalized tick with mid price.

        Returns
        -------
        WaveletPoint or None
            Computed features, or None if insufficient history.
        """
        self._buffer.append(tick.mid)
        self._tick_count += 1

        if not self._buffer.is_full:
            return None

        values = self._buffer.as_array()

        coefficients = decompose(
            values, self._config.wavelet, self._effective_level
        )
        trend_series = reconstruct_trend(
            coefficients, self._config.wavelet, len(values)
        )

        return extract_features(
            values=values,
            trend_series=trend_series,
            coefficients=coefficients,
            volatility_window=self._config.volatility_window,
        )

    def reset(self) -> None:
        """Reset the engine state, clearing the buffer.

        After reset, the engine needs to accumulate a full window of
        ticks before producing output again.
        """
        self._buffer = RollingBuffer(self._config.window)
        self._tick_count = 0
