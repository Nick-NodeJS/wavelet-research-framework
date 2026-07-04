# Story 20 — Wavelet Trend Calibration

## Goal

Calibrate the Wavelet Engine to produce a trend line that is visually and quantitatively superior to a Moving Average while remaining strictly causal (non-repainting).

This story does **not** introduce new algorithms.

The goal is to make the existing engine configurable so that different parameter combinations can be evaluated directly from MT5.

---

# Background

Current implementation already supports:

- db4 wavelet
- causal rolling window
- approximation reconstruction
- MT5 Bridge
- Trend visualization

However the resulting trend is still too flat and does not outperform a standard Moving Average.

The most probable reason is parameter selection rather than algorithm correctness.

---

# Scope

Parameterize the Wavelet Engine.

Support runtime selection of:

- rolling window
- wavelet family
- decomposition level
- reconstruction mode

without changing code.

---

# Functional Requirements

## 1. Window Size

Allow runtime selection.

Supported values:

- 128
- 256
- 512
- 1024
- 2048

The selected value must be sent by MT5 and used by the Python service.

---

## 2. Wavelet Family

Support:

- db2
- db4
- db6
- sym4
- sym6
- coif1

Configuration only.

No algorithm changes.

---

## 3. Decomposition Level

Support runtime values:

- 1
- 2
- 3
- 4

Current validation must remain.

---

## 4. Reconstruction Mode

Support runtime selection.

Current modes:

- A1
- A2
- A3
- A4

No additional modes in this story.

---

## 5. Logging

Every request must log

```
window
wavelet
level
trend_mode
elapsed_ms
```

Example

```
POST /wavelet
window=256
wavelet=db4
level=2
mode=A2
elapsed=118ms
```

---

## 6. MT5 EA

Expose new inputs

```
Window
Wavelet
Level
TrendMode
```

These values must be included in every request.

---

## 7. MT5 Indicator

No changes.

Indicator must continue displaying the returned trend.

---

# Manual Test Matrix

Test all combinations below.

## Window

- 128
- 256
- 512
- 1024
- 2048

with

```
db4
level=2
mode=A2
```

Capture screenshots.

---

## Wavelet

Compare

- db2
- db4
- db6
- sym4
- sym6
- coif1

using

```
window=256
level=2
mode=A2
```

---

## Levels

Compare

```
1
2
3
4
```

using

```
window=256
db4
A2
```

---

## Modes

Compare

```
A1
A2
A3
A4
```

using

```
window=256
db4
level=2
```

---

# Success Criteria

The team must identify one parameter combination that:

- follows price significantly better than current implementation
- is smoother than raw price
- remains strictly causal
- produces no repainting
- visually outperforms MA on the same chart

---

# Out of Scope

- New trend algorithms
- Multi-scale trend fusion
- Adaptive windows
- Kalman filtering
- Savitzky-Golay
- Hodrick-Prescott
- Signal generation

Those belong to future stories.

---

# Deliverables

- configurable Wavelet Engine
- configurable MT5 EA
- parameter propagation to Python
- logging
- comparison screenshots
- documented recommended parameter set