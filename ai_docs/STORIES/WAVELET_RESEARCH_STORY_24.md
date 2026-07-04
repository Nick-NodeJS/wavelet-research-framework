# WAVELET_RESEARCH_STORY_24.md

# Story

Create a compact calibration command to derive practical thresholds from historical data.

# Goal

Avoid manual guessing of thresholds while also avoiding over-complex optimization.

# Scope

Add CLI command:

```bash
wavelet-research calibrate-trend-strategy --ticks data.csv --wavelet db4 --window 512 --output config/calibrated.json
```

# Calibration Targets

Find conservative values for:

```text
min_normalized_deviation
min_return_probability
min_stats_sample_size
max_adverse_normalized_deviation
max_holding_bars
volatility_spike_threshold
```

# Method

Use deterministic grid search with strict constraints:

```text
minimum trades
positive expectancy
acceptable drawdown
stable OOS result
low parameter sensitivity
```

# Output

Write a small JSON config usable by service/MT5:

```json
{
  "symbol": "EURUSD",
  "wavelet": "db4",
  "window": 512,
  "entry": {},
  "exit": {},
  "filters": {},
  "validation_summary": {}
}
```

# Tests

Add tests for:

- calibration output schema;
- deterministic run with seed;
- constraints applied;
- config load by signal engine;
- CLI smoke test.

# Acceptance Criteria

- User can calibrate strategy from historical CSV in one command.
- Output config is directly usable by real-time service.
- Calibration rejects unstable parameter sets.
