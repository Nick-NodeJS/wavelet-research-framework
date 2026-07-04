# WAVELET_RESEARCH_STORY_19.md

# Story

Implement a normalized deviation engine that measures how far price is from the causal trend.

# Goal

Price always oscillates around trend. The system needs a stable, comparable deviation measure instead of raw price distance.

# Scope

Create:

```text
wavelet_research/deviation/
```

Recommended files:

```text
models.py
core.py
normalization.py
```

# Inputs

For every tick/bar:

```text
price
trend
spread
volatility estimate
ATR or rolling absolute return
trend slope
```

# Outputs

```text
raw_distance = price - trend
relative_distance = raw_distance / trend
volatility_normalized_distance = raw_distance / rolling_volatility
z_score_distance
side = above|below|near
```

# Design Rules

- No trading decision inside this module.
- No lookahead.
- Deterministic output.
- Works both in backtest and live service.

# Integration

Expose deviation result to:

- signal engine;
- service response;
- research/backtest pipeline;
- MT5 indicator response.

# Tests

Add tests for:

- zero volatility protection;
- positive/negative deviation;
- deterministic rolling calculation;
- no division by zero;
- service contract compatibility.

# Acceptance Criteria

- Every WaveletPoint can be converted into DeviationPoint.
- Deviation is normalized and comparable across time.
- Signal engine can consume deviation without recalculating it.
