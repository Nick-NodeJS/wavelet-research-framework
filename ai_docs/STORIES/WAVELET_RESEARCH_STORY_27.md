# WAVELET_RESEARCH_STORY_27.md

# Story

Run paper trading acceptance validation using calibrated trend strategy.

# Goal

Before real trading, prove that live-like sequential behavior matches historical expectations.

# Scope

Extend paper trading framework if needed.

# Required Run Modes

```text
historical replay paper run
real-time demo paper run
journal export
daily summary export
```

# Journal Fields

```text
timestamp
symbol
signal
entry_price
exit_price
entry_reason
exit_reason
normalized_deviation
historical_probability
filter_state
pnl
mae
mfe
holding_bars
```

# Acceptance Metrics

Configurable gate:

```text
min_trades
profit_factor
expectancy
max_drawdown
max_consecutive_losses
signal_consistency
```

# Tests

Add tests for:

- journal completeness;
- calibrated config loading;
- paper run determinism;
- acceptance gate pass/fail;
- export format.

# Acceptance Criteria

- Paper run produces inspectable journal.
- Acceptance gate can fail strategy objectively.
- Same signal/exit logic is used as service/backtest.
