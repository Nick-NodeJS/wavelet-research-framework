# WAVELET_RESEARCH_STORY_04.md

# Story

Implement the production-ready Causal Wavelet Engine.

# Goal

Build the mathematical core capable of processing streaming tick data using only historical information.

The engine must never repaint and must never use future samples.

# Scope

Implement only:

- Wavelet Engine
- Rolling window management
- Wavelet decomposition
- Trend reconstruction
- Feature extraction

Do NOT implement:

- Signal generation
- Trading logic
- Backtesting
- MT5 integration
- Visualization

# Supported Wavelets

- haar
- db4
- db6
- sym4
- coif3

The implementation must be extensible for additional families.

# Windows

Support configurable rolling windows:

- 256
- 512
- 1024
- 2048
- 4096

# Input

Streaming normalized ticks:

- time
- bid
- ask
- mid
- spread

# Output

WaveletPoint

Fields:

- trend
- deviation
- z_score
- slope
- energy
- noise

# Requirements

- Causal processing only
- No look-ahead bias
- No repainting
- Immutable configuration
- Deterministic output

# Trend

Reconstruct trend from approximation coefficients only.

# Features

Compute:

- trend
- deviation = mid - trend
- rolling volatility
- z_score
- trend slope
- wavelet energy
- noise estimate

# Public API

```python
config = WaveletConfig(...)

engine = WaveletEngine(config)

point = engine.update(tick)
```

# Performance

- Streaming implementation
- Efficient rolling buffer
- Minimize allocations
- Suitable for millions of ticks

# Validation

Reject:

- unsupported wavelet
- invalid level
- invalid window
- insufficient history

# Tests

Cover:

- all supported wavelets
- all supported windows
- deterministic output
- no repainting
- no future dependency
- trend reconstruction
- z-score
- slope
- energy
- noise
- invalid configurations

# Code Quality

- Python 3.12+
- Full typing
- SOLID
- Small focused classes
- Dependency Injection
- No duplicated logic
- Ruff
- Black
- mypy
- High unit test coverage

# Acceptance Criteria

- Processes ticks sequentially.
- Uses only historical data.
- Produces deterministic WaveletPoint.
- Fully covered by tests.
- Ready for Signal Engine integration.
