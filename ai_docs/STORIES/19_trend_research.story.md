# Story-019 — Configurable Wavelet Trend Modes

## Goal

Transform the current implementation from a single fixed wavelet trend (A2) into a configurable research platform capable of comparing multiple approximation levels without changing the infrastructure.

This story does **not** change the trading logic.

Its purpose is to determine which wavelet approximation best represents the underlying market trend.

---

# Background

Current implementation reconstructs the trend using only:

- db4
- decomposition level = 2
- approximation A2

Although mathematically correct, this may not be the optimal definition of "trend" for trading.

The system must become configurable.

---

# Functional Requirements

## Python

Replace the fixed trend reconstruction with selectable modes.

Supported modes:

```
A1
A2
A3
A4
```

Each mode reconstructs the signal using only the approximation coefficients of the selected level.

No detail coefficients are used.

---

## API

Current API remains backward compatible.

Add optional request field:

```json
{
    "trend_mode": "A2"
}
```

Default:

```
A2
```

Supported values:

```
A1
A2
A3
A4
```

Invalid values:

```
HTTP 400
```

---

## Wavelet Engine

Refactor reconstruction into:

```python
reconstruct_trend(
    coefficients,
    wavelet_name,
    signal_length,
    approximation_level
)
```

No duplicated code.

---

## MT5 Bridge EA

Read user input:

```
Trend Mode
```

Possible values:

```
A1
A2
A3
A4
```

Include it in every request.

No other behaviour changes.

---

## MT5 Indicator

Display current mode inside status:

```
WaveletTrend

Connected

Mode : A2

Latency : xx ms
```

No additional plots.

No trading logic.

---

# Acceptance Criteria

## Functional

- A1 works

- A2 works

- A3 works

- A4 works

without restarting MT5.

---

Switching the mode causes the next request to return the selected trend.

---

No repaint.

---

No look-ahead.

---

No API breaking changes.

---

Unit tests added.

---

Integration tests updated.

---

Regression tests pass.

---

# Non Functional

No measurable performance degradation.

Memory usage unchanged.

Existing API clients continue working.

---

# Out of Scope

Multiple trend lines.

EMA comparison.

Deviation analysis.

Trading signals.

Trend scoring.

Those belong to subsequent stories.

---

# Expected Outcome

After completion we can visually compare different wavelet trend definitions on identical market data and objectively determine which approximation level provides the most useful baseline for subsequent deviation and signal research.