# Wavelet Research Framework

Production-grade wavelet-based trading research system for FX tick data.

---

## Architecture

```
Tick Data (CSV)
      │
      ▼
Wavelet Engine          ← causal rolling-window decomposition (db4)
      │
      ▼
Signal Engine           ← z-score thresholds, filters, confidence
      │
      ▼
Backtest Engine         ← sequential replay, virtual orders, metrics
      │
      ▼
Orchestrator            ← parameter matrix, parallel-safe experiments
      │
      ├── Walk-Forward Validator   ← IS/OOS splits, Monte Carlo, stability
      ├── Parameter Optimizer      ← grid / random search, constraints
      └── AI Research Assistant   ← failure analysis, recommendations

Python Wavelet Service  ← HTTP API (Flask) for MT5 integration
      │
      ▼ POST /wavelet
MT5 Thin Indicator      ← visualization client (MQL5), no calculations
```

---

## Components

### Wavelet Engine
Causal streaming decomposition using PyWavelets.
Fixed wavelet: **db4**. Outputs: trend, deviation, z-score, energy, noise.

### Signal Engine
Converts `WaveletPoint` output into BUY / SELL / HOLD decisions.
Configurable z-score thresholds, slope filter, energy filter, confidence scoring.

### Backtesting Engine
Sequential tick replay. Virtual order execution with configurable exit strategies:
`MAX_HOLD`, `TAKE_PROFIT`, `STOP_LOSS`, `COMBINED`. Full trade journal + metrics.

### Research Framework (Orchestrator)
Generates parameter matrices and runs experiments. Ranks results by profit factor,
expectancy, and drawdown. Persists results to CSV/Parquet.

### Walk-Forward Validator
In-sample / out-of-sample splits, rolling walk-forward folds, Monte Carlo
trade-order shuffle, robustness percentiles, parameter sensitivity (CV).

### Parameter Optimizer
Grid search and random search over the full parameter space with multi-objective
scoring and hard constraints (min trades, max drawdown, min profit factor).

### Python Wavelet Service
Local HTTP service exposing the Wavelet Engine via `POST /wavelet`.
Single computation backend for all MT5 integrations.

### MT5 Thin Indicator
MQL5 indicator that collects ticks, POSTs to the Python service, and draws
the returned trend and deviation buffers. Zero wavelet math in MQL5.

---

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

---

## Running the Wavelet Service

```bash
# Default: http://127.0.0.1:5000
wavelet-service

# Custom config via environment
WAVELET_WINDOW=512 WAVELET_LEVEL=2 SERVICE_PORT=5000 wavelet-service
```

Health check:
```bash
curl http://127.0.0.1:5000/health
# {"status": "ok", "wavelet": "db4", "version": "0.2.0"}
```

---

## Research CLI

```bash
# Run experiments
wavelet-research research --ticks data/ticks.csv --wavelets haar,db4 --windows 256,512

# Optimize parameters
wavelet-research optimize --ticks data/ticks.csv --random --max-iter 200

# Walk-forward validation
wavelet-research validate --ticks data/ticks.csv --wavelet db4 --window 512 --folds 5

# Paper trading
wavelet-research paper-trade --ticks data/ticks.csv --balance 10000
```

---

## MT5 Integration

1. Start the Python service: `wavelet-service`
2. Compile `mql5/WaveletThinIndicator.mq5` in MetaEditor
3. Allow WebRequests to `http://127.0.0.1:5000` in MT5 settings
4. Attach the indicator to a chart — it auto-requests wavelet data on each bar

**MT5 inputs:**
- `Server URL` — Python service address (default: `http://127.0.0.1:5000`)
- `Tick Window` — number of ticks per request (default: 2048)
- `Request Timeout (ms)` — HTTP timeout (default: 500)

---

## Development Workflow

```bash
# Run all tests
python -m pytest tests/ -q

# Run specific story tests
python -m pytest tests/test_service.py -v
python -m pytest tests/test_mt5_indicator_client.py -v

# Lint
ruff check wavelet_research/
```

---

## Input Format

```csv
time,bid,ask
2026-01-01 00:00:00.001,1.10001,1.10004
```

`mid` and `spread` are computed automatically if omitted.

---

## Key Design Constraints

- **Causal only** — no future data ever used
- **Deterministic** — seeded RNG, frozen dataclasses throughout
- **db4 fixed** — the production service uses db4 exclusively
- **No MQL5 math** — all calculations in Python, MT5 is visualization only
