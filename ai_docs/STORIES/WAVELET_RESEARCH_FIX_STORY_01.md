# WAVELET_RESEARCH_FIX_STORY_01.md

# Story

Stabilize the current implementation before continuing feature development.

---

# Goal

Resolve architectural, packaging and integration issues identified during repository validation.

No new functionality shall be introduced.

Only quality improvements and bug fixes.

---

# Issue 1 — Remove Virtual Environment

The repository must never contain:

- .venv
- __pycache__
- *.pyc

Tasks:

- remove tracked virtual environment
- update .gitignore
- verify clean repository

---

# Issue 2 — README

Rewrite README to reflect the current architecture.

It must describe:

- Wavelet Engine
- Python Service
- MT5 Thin Indicator
- Research Framework
- Backtesting
- Validation
- Optimization

Include:

- architecture diagram
- installation
- running service
- MT5 integration
- development workflow

---

# Issue 3 — MQL5 Timestamp

Review timestamp serialization.

Use native MT5 millisecond timestamps.

Avoid precision loss.

Create regression test proving identical timestamps between MT5 and Python.

---

# Issue 4 — Price Precision

Remove hardcoded:

```cpp
DoubleToString(price, 5)
```

Use symbol precision:

```cpp
_Digits
```

All serialization must work correctly for symbols with different decimal precision.

---

# Issue 5 — Indicator Windows

Review indicator declaration.

Ensure:

- trend overlays the main chart
- Relative Deviation is displayed correctly
- Z-Score is displayed correctly
- Energy is displayed correctly

Refactor if multiple indicators are required due to MT5 limitations.

Do not rely on unsupported combinations.

---

# Issue 6 — HTTP Contract

Freeze API contract.

Create JSON schema for:

POST /wavelet

Validate:

- request
- response
- missing fields
- malformed arrays
- incompatible versions

---

# Issue 7 — Contract Tests

Add automated integration tests covering:

Python Service ↔ MT5 Contract

Verify:

- serialization
- timestamp precision
- floating point precision
- array sizes
- deterministic responses

---

# Issue 8 — Smoke Tests

Implement automated smoke tests:

- service startup
- /health
- /wavelet
- malformed request
- timeout
- unsupported configuration

---

# Acceptance Criteria

- Repository contains no generated files.
- README matches the implemented architecture.
- Timestamp serialization is deterministic.
- Price precision is symbol-independent.
- MT5 contract is frozen and tested.
- All integration tests pass.
- Ready for quantitative research.