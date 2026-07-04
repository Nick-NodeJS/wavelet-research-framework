# WAVELET_RESEARCH_STORY_22.md

# Story

Implement simple trend-relative entry rules.

# Goal

Open trades only when price deviation from trend is large enough and historical return-to-trend statistics are favorable.

# Scope

Extend:

```text
wavelet_research/signal/
```

Do not create a separate strategy framework unless necessary.

# Entry Logic

First version:

```text
If price is below upward/acceptable trend by enough normalized deviation
and historical probability of return is high
and no-trade filters pass
then BUY.

If price is above downward/acceptable trend by enough normalized deviation
and historical probability of return is high
and no-trade filters pass
then SELL.
```

# Config

Add config fields:

```text
min_normalized_deviation
min_return_probability
min_stats_sample_size
allow_countertrend
max_spread
cooldown_bars
```

Default should be conservative.

# Output

SignalDecision should include:

```text
signal
confidence
entry_reason
normalized_deviation
historical_probability
expected_bars_to_return
filter_reasons
```

# Tests

Add tests for:

- BUY below trend;
- SELL above trend;
- blocked by filter;
- blocked by low historical probability;
- confidence calculation;
- deterministic decision.

# Acceptance Criteria

- Entry rules are explainable.
- Entry rules do not depend on future data.
- Entry rules use trend, deviation and historical stats only.
