# WAVELET_RESEARCH_STORY_18.md

# Story

Audit and harden the causal trend line produced by the current wavelet engine.

# Goal

Before building trading rules, verify that the trend itself is usable as the central reference line for trading.

The system must answer:

- Is the trend causal?
- Is it stable when new ticks arrive?
- Does it avoid repainting historical values?
- Is it smooth enough to filter noise?
- Is it responsive enough to be tradable?

# Scope

Implement a trend quality audit module under:

```text
wavelet_research/trend_quality/
```

Recommended files:

```text
wavelet_research/trend_quality/__init__.py
wavelet_research/trend_quality/models.py
wavelet_research/trend_quality/metrics.py
wavelet_research/trend_quality/audit.py
```

# Required Metrics

Compute at minimum:

```text
repaint_delta
trend_lag_estimate
trend_smoothness
trend_direction_stability
price_cross_frequency
mean_abs_distance
normalized_mean_abs_distance
```

# Causal / Repaint Check

Replay ticks sequentially.
For each timestamp store the trend value as first seen.
Later replay longer windows and verify that already emitted trend values do not materially change.

# CLI

Add CLI command:

```bash
wavelet-research trend-audit --ticks data/EURUSD.csv --wavelet db4 --window 512 --output reports/trend_audit.json
```

# Output

Write JSON report:

```json
{
  "trend_quality_score": 0.0,
  "repaint_max": 0.0,
  "lag_estimate_bars": 0,
  "smoothness": 0.0,
  "cross_frequency": 0.0,
  "recommendation": "pass|fail|review"
}
```

# Tests

Add tests for:

- no future data usage;
- repaint metric;
- trend quality report serialization;
- deterministic replay;
- CLI smoke test.

# Acceptance Criteria

- Trend audit runs on historical CSV.
- Report is deterministic.
- Repaint risk is explicitly measured.
- Story fails if the trend cannot be trusted as a trading reference.
