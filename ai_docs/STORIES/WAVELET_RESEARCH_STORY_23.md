# WAVELET_RESEARCH_STORY_23.md

# Story

Implement exit rules for return-to-trend trading.

# Goal

A trade should close when the reason for opening it is no longer valid, not only by fixed TP/SL.

# Scope

Extend backtester and paper trading exit logic.

# Exit Types

Implement:

```text
RETURN_TO_TREND
DEVIATION_NORMALIZED
TREND_INVALIDATION
MAX_ADVERSE_MOVE
MAX_HOLDING_BARS
SESSION_END_OPTIONAL
```

# Config

```text
exit_on_trend_touch = true
exit_deviation_threshold = 0.2
max_holding_bars = 20
max_adverse_normalized_deviation = 2.5
trend_invalidation_slope_change = optional
```

# Backtester Integration

Backtest must record exit reason per trade.

# Metrics

Add breakdown:

```text
PnL by exit reason
win rate by exit reason
average holding time by exit reason
MAE/MFE by exit reason
```

# Tests

Add tests for:

- exit on trend touch;
- exit by normalized deviation;
- exit by max holding bars;
- exit by adverse move;
- trade journal stores exit reason.

# Acceptance Criteria

- Backtester supports trend-relative exits.
- Paper trading uses the same exit logic.
- Exit reasons are visible in reports.
