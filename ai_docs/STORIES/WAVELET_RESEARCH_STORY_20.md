# WAVELET_RESEARCH_STORY_20.md

# Story

Build historical statistics for what typically happens after similar deviations from trend.

# Goal

For a current market state, the system should answer:

> Historically, after similar deviation/trend/volatility conditions, how often did price return to trend and how quickly?

# Scope

Create:

```text
wavelet_research/deviation_stats/
```

Recommended files:

```text
models.py
collector.py
index.py
query.py
storage.py
```

# Historical Event Record

For each bar/tick snapshot store:

```text
timestamp
symbol
timeframe/window
trend_value
price
normalized_deviation
trend_slope
volatility_bucket
future_return_1
future_return_3
future_return_5
future_return_10
future_return_20
returned_to_trend
bars_to_return
max_favorable_excursion
max_adverse_excursion
```

# Query Interface

Given current state:

```python
stats = deviation_stats.query(
    normalized_deviation=1.8,
    trend_slope=0.0002,
    volatility_bucket="normal",
)
```

Return:

```text
sample_size
return_to_trend_probability
median_bars_to_return
expected_return
expected_adverse_excursion
confidence_level
```

# Simplicity Constraint

Use bins/buckets and deterministic lookup first.
Do not add ML model here.

# Storage

Support CSV/JSONL or parquet if already available in dependencies.
Keep first version simple and inspectable.

# Tests

Add tests for:

- event collection without lookahead in current features;
- future label generation only during offline build;
- query by buckets;
- low sample size handling;
- deterministic results.

# Acceptance Criteria

- Historical stats can be built from existing CSV data.
- Current market state can query similar historical situations.
- Low-confidence situations are explicitly marked.
