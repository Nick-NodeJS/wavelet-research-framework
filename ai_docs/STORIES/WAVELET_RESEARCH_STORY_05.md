# WAVELET_RESEARCH_STORY_05.md

# Story

Implement the Signal Engine.

# Goal

Create a deterministic signal generation layer that converts WaveletPoint data into BUY, SELL, or HOLD decisions.

# Scope

Implement only:

- Signal configuration
- Signal decision model
- Signal engine
- Threshold-based signal rules
- Multi-scale signal aggregation support

Do NOT implement:

- Order execution
- Position management
- Backtesting
- MT5 integration
- Visualization
- AI logic

# Input

WaveletPoint or multiple WaveletPoint values from different windows/scales.

Required fields:

- trend
- deviation
- z_score
- slope
- energy
- noise

# Output

SignalDecision

Fields:

- signal
- confidence
- reason
- z_score
- trend_slope
- energy
- noise
- metadata

Allowed signals:

- BUY
- SELL
- HOLD

# Configuration

Create immutable SignalConfig.

Support:

- buy_z_threshold
- sell_z_threshold
- min_confidence
- slope_filter_enabled
- energy_filter_enabled
- noise_filter_enabled
- max_noise
- min_energy
- allow_buy
- allow_sell

# Core Rules v1

BUY condition:

- z_score <= -buy_z_threshold
- optional slope filter confirms improvement
- optional energy filter confirms movement
- optional noise filter does not reject signal

SELL condition:

- z_score >= sell_z_threshold
- optional slope filter confirms weakness
- optional energy filter confirms movement
- optional noise filter does not reject signal

Otherwise:

- HOLD

# Confidence

Implement deterministic confidence score.

Confidence should be based on:

- absolute z_score strength
- slope alignment
- energy confirmation
- noise penalty

Confidence must be normalized to:

0.0 ... 1.0

# Multi-scale Support

The engine must support future multi-scale input.

Design API so it can accept:

- one WaveletPoint
- collection of WaveletPoint values

Do not hardcode H1/H4/D1/W1.

# Public API

```python
config = SignalConfig(...)

engine = SignalEngine(config)

decision = engine.decide(point)
```

For future multi-scale:

```python
decision = engine.decide_many(points)
```

# Determinism

Same input must always produce the same output.

No randomness.

No time-dependent behavior.

# Validation

Reject:

- invalid thresholds
- invalid confidence range
- missing WaveletPoint fields
- unsupported signal direction

# Tests

Create tests for:

- BUY signal
- SELL signal
- HOLD signal
- threshold boundaries
- confidence calculation
- slope filter
- energy filter
- noise filter
- buy disabled
- sell disabled
- deterministic output
- invalid config
- multi-scale API smoke test

# Code Quality

- Python 3.12+
- Full typing
- Dataclasses
- Small focused classes
- No duplicated logic
- Ruff
- Black
- mypy
- High unit test coverage

# Acceptance Criteria

- Converts WaveletPoint into deterministic SignalDecision.
- Supports BUY, SELL, HOLD.
- Supports configurable filters.
- Produces normalized confidence.
- Ready for Backtester integration.
