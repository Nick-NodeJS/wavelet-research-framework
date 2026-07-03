# Wavelet Research v0

Research prototype for tick-based FX wavelet deviation strategy.

## Goal

Input:

```csv
time,bid,ask
2026-01-01 00:00:00.001,1.10001,1.10004
```

Output:

- wavelet trend
- normalized deviation / z-score
- signal
- simple backtest metrics
- best configs report

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python -m wavelet_research.cli \
  --ticks data/ticks.csv \
  --symbol EURUSD \
  --pip-size 0.0001 \
  --spread-pips 0.0 \
  --max-hold 500 \
  --output reports/report.csv
```

## Important

This v0 is for research only.

The algorithm is causal at the decision level: every signal is calculated on a rolling window ending at the current tick.
