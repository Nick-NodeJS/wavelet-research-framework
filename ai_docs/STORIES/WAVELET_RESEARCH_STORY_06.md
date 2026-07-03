# WAVELET_RESEARCH_STORY_06.md

# Story

Implement the Backtesting Engine.

# Goal

Build a deterministic backtesting engine capable of replaying historical ticks sequentially and evaluating SignalEngine decisions.

# Scope

Implement:

- sequential tick replay
- virtual order execution
- position lifecycle
- configurable exit strategies
- metrics collection
- trade journal

Do NOT implement:

- MT5 integration
- optimization
- AI
- visualization

# Input

- normalized tick stream
- SignalEngine
- BacktestConfig

# Exit Strategies

Support:

- fixed TP/SL
- return to wavelet trend
- opposite signal
- maximum holding ticks
- partial close (extensible)

# Trading Costs

Support:

- Bid/Ask execution
- spread
- commission
- slippage
- swap (optional)

# Metrics

Minimum:

- trades
- win rate
- profit factor
- expectancy
- total pnl
- max drawdown
- MAE
- MFE
- average trade
- average holding time

# Outputs

- BacktestReport
- TradeJournal
- EquityCurve

# API

```python
engine = BacktestEngine(config)

report = engine.run(dataset, signal_engine)
```

# Requirements

- deterministic
- sequential processing
- no future data
- configurable
- reusable

# Tests

Cover:

- BUY lifecycle
- SELL lifecycle
- all exit strategies
- costs
- drawdown
- MAE/MFE
- deterministic replay
- invalid configs

# Code Quality

- Python 3.12+
- SOLID
- full typing
- Ruff
- Black
- mypy
- high unit test coverage

# Acceptance Criteria

- Replays historical ticks sequentially.
- Executes trades correctly.
- Produces deterministic reports and journals.
- Ready for Research Framework integration.
