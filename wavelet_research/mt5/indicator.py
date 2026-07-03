"""MT5 Indicator adapter.

Wraps WaveletEngine and SignalEngine to produce indicator buffers
for MT5 display: trend, deviation, z-score, energy, and signals.
No business logic duplication — delegates to existing engines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from wavelet_research.engine.config import WaveletEngineConfig
from wavelet_research.engine.core import WaveletEngine
from wavelet_research.engine.models import Tick, WaveletPoint
from wavelet_research.signal.config import SignalConfig
from wavelet_research.signal.core import SignalEngine
from wavelet_research.signal.models import Signal, SignalDecision


@dataclass
class IndicatorBuffer:
    """Buffer holding indicator values for one tick.

    Parameters
    ----------
    trend : float
        Wavelet trend value.
    deviation : float
        Deviation from trend.
    z_score : float
        Normalized z-score.
    energy : float
        Wavelet energy.
    noise : float
        Noise estimate.
    signal : int
        Signal direction: 1=BUY, -1=SELL, 0=HOLD.
    confidence : float
        Signal confidence [0.0, 1.0].
    """

    trend: float = 0.0
    deviation: float = 0.0
    z_score: float = 0.0
    energy: float = 0.0
    noise: float = 0.0
    signal: int = 0
    confidence: float = 0.0


_SIGNAL_MAP = {
    Signal.BUY: 1,
    Signal.SELL: -1,
    Signal.HOLD: 0,
}


class MT5Indicator:
    """MT5 Indicator using the Wavelet Engine.

    Processes ticks and produces indicator buffers for display.
    Delegates all computation to WaveletEngine and SignalEngine.

    Parameters
    ----------
    wavelet_config : WaveletEngineConfig
        Wavelet engine configuration.
    signal_config : SignalConfig
        Signal engine configuration.
    """

    def __init__(
        self,
        wavelet_config: WaveletEngineConfig,
        signal_config: SignalConfig,
    ) -> None:
        self._wavelet_engine = WaveletEngine(wavelet_config)
        self._signal_engine = SignalEngine(signal_config)
        self._history: list[IndicatorBuffer] = []

    @property
    def history(self) -> list[IndicatorBuffer]:
        """All indicator buffers produced so far."""
        return list(self._history)

    @property
    def buffer_count(self) -> int:
        """Number of buffers produced."""
        return len(self._history)

    def on_tick(self, tick: Tick) -> IndicatorBuffer:
        """Process a single tick and produce indicator values.

        Parameters
        ----------
        tick : Tick
            Incoming tick data.

        Returns
        -------
        IndicatorBuffer
            Indicator values for this tick.
        """
        wavelet_point = self._wavelet_engine.update(tick)

        if wavelet_point is None:
            buffer = IndicatorBuffer()
        else:
            decision = self._signal_engine.decide(wavelet_point)
            buffer = IndicatorBuffer(
                trend=wavelet_point.trend,
                deviation=wavelet_point.deviation,
                z_score=wavelet_point.z_score,
                energy=wavelet_point.energy,
                noise=wavelet_point.noise,
                signal=_SIGNAL_MAP[decision.signal],
                confidence=decision.confidence,
            )

        self._history.append(buffer)
        return buffer

    def reset(self) -> None:
        """Reset indicator state for new chart."""
        self._wavelet_engine = WaveletEngine(
            self._wavelet_engine._config
        )
        self._signal_engine = SignalEngine(
            self._signal_engine._config
        )
        self._history.clear()
