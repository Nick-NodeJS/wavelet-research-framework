"""Domain models for the no-trade filter engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class FilterReason(str, Enum):
    """Reason a trade was blocked."""

    LOW_SAMPLE_SIZE = "LOW_SAMPLE_SIZE"
    LOW_RETURN_PROBABILITY = "LOW_RETURN_PROBABILITY"
    HIGH_ADVERSE_EXCURSION = "HIGH_ADVERSE_EXCURSION"
    TREND_UNSTABLE = "TREND_UNSTABLE"
    VOLATILITY_SPIKE = "VOLATILITY_SPIKE"
    SPREAD_TOO_WIDE = "SPREAD_TOO_WIDE"
    DEVIATION_TOO_SMALL = "DEVIATION_TOO_SMALL"
    TREND_TOO_FLAT = "TREND_TOO_FLAT"
    RECENT_SIGNAL_COOLDOWN = "RECENT_SIGNAL_COOLDOWN"


class FilterSeverity(str, Enum):
    """Severity of the filter decision."""

    PASS = "pass"
    WARNING = "warning"
    BLOCK = "block"


@dataclass(frozen=True)
class FilterResult:
    """Result of a filter engine evaluation.

    Parameters
    ----------
    can_trade : bool
        True when all filters pass.
    reasons : tuple[FilterReason, ...]
        All triggered filter reasons.
    severity : FilterSeverity
        Worst severity among triggered filters.
    """

    can_trade: bool
    reasons: tuple[FilterReason, ...]
    severity: FilterSeverity

    def to_dict(self) -> dict[str, object]:
        """Serialise to JSON-compatible dict.

        Returns
        -------
        dict[str, object]
            Plain representation.
        """
        return {
            "can_trade": self.can_trade,
            "reasons": [r.value for r in self.reasons],
            "severity": self.severity.value,
        }


@dataclass(frozen=True)
class FilterConfig:
    """Configuration for the no-trade filter engine.

    Parameters
    ----------
    min_sample_size : int
        Minimum historical events required.
    min_return_probability : float
        Minimum return-to-trend probability [0..1].
    max_adverse_excursion : float
        Maximum acceptable expected adverse excursion.
    max_spread : float
        Maximum allowed spread (in price units).
    min_normalized_deviation : float
        Minimum absolute z-score to consider a trade.
    min_slope_abs : float
        Minimum absolute slope to consider trend non-flat.
    volatility_spike_multiplier : float
        Factor above recent volatility to flag spike.
    cooldown_bars : int
        Bars to wait after a signal before allowing a new one.
    """

    min_sample_size: int = 50
    min_return_probability: float = 0.55
    max_adverse_excursion: float = 2.0
    max_spread: float = 0.0003
    min_normalized_deviation: float = 1.0
    min_slope_abs: float = 1e-6
    volatility_spike_multiplier: float = 3.0
    cooldown_bars: int = 5
