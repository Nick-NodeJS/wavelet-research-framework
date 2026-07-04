# WAVELET_RESEARCH_STORY_21.md

# Story

Implement a no-trade filter engine that blocks statistically weak or noisy situations.

# Goal

The system should not try to trade every deviation. The main practical edge is avoiding bad contexts.

# Scope

Create or extend:

```text
wavelet_research/filters/
```

Recommended files:

```text
models.py
rules.py
engine.py
```

# Filter Categories

Implement simple deterministic filters:

```text
LOW_SAMPLE_SIZE
LOW_RETURN_PROBABILITY
HIGH_ADVERSE_EXCURSION
TREND_UNSTABLE
VOLATILITY_SPIKE
SPREAD_TOO_WIDE
DEVIATION_TOO_SMALL
TREND_TOO_FLAT
RECENT_SIGNAL_COOLDOWN
```

# Inputs

```text
DeviationPoint
TrendQualityState
HistoricalDeviationStats
Market microstructure data if available
```

# Output

```json
{
  "can_trade": false,
  "reasons": ["LOW_RETURN_PROBABILITY", "VOLATILITY_SPIKE"],
  "severity": "block|warning|pass"
}
```

# Integration

SignalEngine must call filter engine before producing BUY/SELL.

# Tests

Add tests for:

- each filter reason;
- multiple reasons;
- pass case;
- integration with signal engine;
- no-trade state visible in service response.

# Acceptance Criteria

- Signal engine can return HOLD/NO_TRADE with explicit reason.
- Trading is blocked when historical statistics are weak.
- All filter decisions are explainable.
