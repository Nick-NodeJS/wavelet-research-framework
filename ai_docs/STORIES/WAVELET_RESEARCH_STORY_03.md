
# Story

Implement the Wavelet Research Framework.

# Goal

Build a reusable research framework capable of evaluating thousands of wavelet configurations on historical tick data.

The framework must be independent of any specific wavelet implementation.

# Scope

Implement:

- research runner
- experiment configuration
- experiment execution
- metrics collection
- result persistence
- ranking

Do NOT implement:

- MT5 integration
- trading logic
- visualization
- optimization algorithms

# Architecture

research/
    runner.py
    experiment.py
    metrics.py
    ranking.py
    storage.py
    config.py

# Configuration

Support:

- wavelet family
- decomposition level
- window size
- volatility window
- normalization method
- signal threshold

Configurations must be immutable.

# Runner

The framework shall execute one configuration at a time and collect metrics.

Execution must be deterministic.

# Metrics

Collect at minimum:

- trades
- win_rate
- profit_factor
- expectancy
- max_drawdown
- total_pnl
- average_trade
- average_win
- average_loss

The framework must allow adding new metrics without modifying existing code.

# Storage

Persist every experiment result.

Support:

- CSV
- Parquet

# Ranking

Sort experiments by configurable criteria.

Default priority:

1. Profit Factor
2. Expectancy
3. Max Drawdown
4. Total PnL

# Public API

runner = ResearchRunner(configs)

results = runner.run(dataset)

# Performance

Designed for batch execution over thousands of configurations.

Avoid duplicated calculations where possible.

# Tests

Create tests for:

- single experiment
- multiple experiments
- deterministic execution
- ranking
- storage
- metrics
- invalid configuration

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

- Framework executes multiple experiments.
- Results are persisted.
- Experiments are ranked.
- Architecture is extensible.
- Ready for Wavelet Engine integration.
