# Wavelet Research Framework — Implementation Report

**Version:** 0.4.0  
**Date:** 2026-07-04  
**Status:** Stories 18–29 implemented and tested  

---

## Executive Summary

Complete implementation of the Wavelet Research Framework spanning 23 stories (05–29 + Fix Story 01). The system provides a production-ready, deterministic research pipeline from raw tick ingestion through trend quality auditing, normalized deviation measurement, historical statistics, no-trade filtering, trend-relative signal generation, and a full statistical gate for live deployment readiness.

**Total test suite: 680 passed, 2 skipped in ~44 seconds.**

---

## Stories Implemented

| # | Story | Package | Tests |
|---|-------|---------|-------|
| 05 | Signal Engine | `wavelet_research/signal/` | 45 |
| 06 | Backtesting Engine | `wavelet_research/backtest/` | 40 |
| 07 | Research Framework (Orchestrator) | `wavelet_research/orchestrator/` | 32 |
| 08 | Walk-Forward Validation | `wavelet_research/validation/` | 44 |
| 09 | Parameter Optimization Engine | `wavelet_research/optimizer/` | 27 |
| 10 | MT5 Indicator | `wavelet_research/mt5/indicator.py` | 9 |
| 11 | MT5 Expert Advisor | `wavelet_research/mt5/expert_advisor.py` | 21 |
| 12 | Paper Trading Framework | `wavelet_research/paper_trading/` | 17 |
| 13 | Performance Profiling | `wavelet_research/profiling/` | 16 |
| 14 | AI Research Assistant | `wavelet_research/research_assistant/` | 13 |
| 15 | Production Packaging | `wavelet_research/cli_research.py` | 14 |
| FX | Fix Story 01: Repo hygiene + contract + tests | `.gitignore`, `service/contract.py` | 43 |
| 17 | MT5 Thin Indicator + HTTP client | `wavelet_research/mt5_indicator/`, `mql5/` | 24 |
| 18 | Trend Quality Audit | `wavelet_research/trend_quality/` | 22 |
| 19 | Deviation Engine | `wavelet_research/deviation/` | 19 |
| 20 | Historical Deviation Stats | `wavelet_research/deviation_stats/` | 12 |
| 21 | No-Trade Filter Engine | `wavelet_research/filters/` | 20 |
| 22 | Trend-Relative Entry Rules | `wavelet_research/signal/core.py` (extended) | 10 |
| 23 | Trend-Relative Exit Rules | `wavelet_research/backtest/` (extended) | 14 |
| 24 | Calibration CLI | `cli_research.py::cmd_calibrate` | CLI smoke |
| 25 | POST /market-state endpoint | `wavelet_research/service/app.py` | 6 |
| 26 | MT5 Visual Signal Panel | `mql5/WaveletSignalPanel.mq5` | MQL5 (manual) |
| 27 | Paper Trading Acceptance Gate | `paper_trading/acceptance.py` | 6 |
| 28 | EA Safety Controls (Python) | `service/risk.py` | 14 |
| 29 | Final Statistical Gate | `research/final_gate.py` | 8 |

---

## Architecture

```
wavelet_research/
├── engine/              # Story 04: Causal Wavelet Engine
│   ├── config.py        # WaveletEngineConfig
│   ├── core.py          # WaveletEngine (streaming tick processing)
│   ├── buffer.py        # Ring buffer
│   ├── decomposition.py # Wavelet decomposition
│   ├── features.py      # Feature extraction
│   └── models.py        # Tick, WaveletPoint
├── signal/              # Story 05: Signal Engine
│   ├── config.py        # SignalConfig (thresholds, filters)
│   ├── core.py          # SignalEngine (decide, decide_many)
│   ├── models.py        # Signal enum, SignalDecision
│   └── rules.py         # Pure functions: threshold, filters, confidence
├── backtest/            # Story 06: Backtesting Engine
│   ├── config.py        # BacktestConfig, ExitStrategy enum
│   ├── core.py          # BacktestEngine (sequential replay)
│   ├── metrics.py       # compute_report (PF, expectancy, drawdown)
│   ├── models.py        # Trade, TradeJournal, EquityCurve, BacktestReport
│   └── position.py      # OpenPosition, exit checks
├── orchestrator/        # Story 07: Research Framework
│   ├── config.py        # PipelineConfig (wavelet + signal + backtest)
│   ├── core.py          # ExperimentOrchestrator
│   ├── matrix.py        # ParameterMatrix (combinatorial generation)
│   ├── pipeline.py      # run_pipeline (single experiment)
│   └── results.py       # ExperimentReport, ranking, persistence
├── validation/          # Story 08: Walk-Forward Validation
│   ├── core.py          # WalkForwardValidator
│   ├── metrics.py       # ExtendedMetrics (Sharpe, Sortino, stability)
│   ├── models.py        # ValidationReport, RobustnessReport
│   ├── monte_carlo.py   # Trade-order shuffle simulation
│   ├── robustness.py    # RobustnessMetrics (percentiles, P(profit))
│   ├── sensitivity.py   # ParameterSensitivity (CV analysis)
│   └── splits.py        # IS/OOS, walk-forward, rolling window
├── optimizer/           # Story 09: Parameter Optimization
│   ├── config.py        # OptimizerConfig, ObjectiveConfig, ConstraintConfig
│   ├── core.py          # ParameterOptimizer
│   ├── models.py        # ScoredConfig, OptimizationReport
│   ├── scoring.py       # Objective scoring, constraint filtering
│   └── search.py        # Grid search, random search
├── mt5/                 # Stories 10–11: MT5 Integration
│   ├── indicator.py     # MT5Indicator (IndicatorBuffer)
│   ├── expert_advisor.py# MT5ExpertAdvisor (OrderRequest)
│   └── risk.py          # RiskConfig, compute_position_size
├── paper_trading/       # Story 12: Paper Trading
│   ├── core.py          # PaperTrader
│   ├── journal.py       # PaperTradeJournal, PaperTrade
│   └── replay.py        # MarketReplay (tick iterator)
├── profiling/           # Story 13: Performance Profiling
│   └── benchmark.py     # LatencyBenchmark, MemoryBenchmark, BenchmarkSuite
├── research_assistant/  # Story 14: AI Research Assistant
│   ├── analyzer.py      # ExperimentAnalyzer (failure analysis, insights)
│   ├── comparator.py    # ParameterComparator
│   └── models.py        # AnalysisReport, ComparisonReport, Recommendation
├── ingestion/           # Stories 01–02: Data Ingestion
├── research/            # Story 03 + Story 29: Framework + Final Gate
│   └── final_gate.py    # GateMetrics, GateConfig, evaluate_gate
├── trend_quality/       # Story 18: Trend Quality Audit
│   ├── models.py        # TrendQualityReport, TrendQualityState
│   ├── metrics.py       # repaint, smoothness, lag, cross-frequency
│   └── audit.py         # TrendAuditor
├── deviation/           # Story 19: Deviation Engine
│   ├── models.py        # DeviationPoint, DeviationSide
│   ├── normalization.py # relative_distance, volatility_normalized
│   └── core.py          # DeviationEngine
├── deviation_stats/     # Story 20: Historical Deviation Stats
│   ├── models.py        # DeviationEvent, DeviationQueryResult
│   ├── collector.py     # DeviationStatsCollector (offline)
│   └── query.py         # DeviationStatsIndex (bucket lookup)
├── filters/             # Story 21: No-Trade Filter Engine
│   ├── models.py        # FilterReason, FilterResult, FilterConfig
│   ├── rules.py         # Pure filter rule functions
│   └── engine.py        # FilterEngine
├── paper_trading/       # Story 12 + Story 27
│   ├── core.py          # PaperTrader
│   ├── journal.py       # PaperTradeJournal, PaperTrade
│   ├── replay.py        # MarketReplay
│   └── acceptance.py    # evaluate_acceptance, AcceptanceConfig
├── service/             # HTTP Service
│   ├── app.py           # Flask: /health, /wavelet, /market-state
│   ├── risk.py          # Story 28: RiskConfig, evaluate_risk
│   └── contract.py      # Frozen API contract + validation
├── cli_research.py      # Story 15 + 18 + 24 + 29: Production CLI
└── cli.py               # Legacy CLI
```

### MQL5 Files

```
mql5/
├── WaveletThinIndicator.mq5   # Main chart: Trend overlay
├── WaveletOscillator.mq5      # Sub-window: RelDev / ZScore / Energy
└── WaveletSignalPanel.mq5     # Story 26: Visual Signal Panel
```

---

## Key Design Decisions

1. **Immutable dataclasses** — All configs and models are frozen for deterministic behavior
2. **Composition over inheritance** — Engines are composed via DI, not subclassed
3. **Pure functions** — Signal rules, metrics computation, scoring are stateless
4. **No logic duplication** — MT5 panels delegate to Python; no calculations in MQL5
5. **Deterministic by default** — All random operations are seeded, all pipelines are reproducible
6. **Causal only** — No future data ever used in current feature computation
7. **Layered signal gating** — Filter engine → deviation gate → stats gate → threshold → signal
8. **Explicit no-trade** — Every HOLD decision carries a machine-readable reason

---

## Pipeline Flow

### Research / Calibration Pipeline
```
Tick Data → WaveletEngine → WaveletPoint
                              ↓               ↓                 ↓
                       TrendAuditor    DeviationEngine    SignalEngine (decide_with_context)
                              ↓               ↓                 ↓
                       TrendQuality   DeviationPoint     FilterEngine ← DeviationStatsIndex
                              └───────────────┴──────────────────┘
                                              ↓
                                       BacktestEngine → BacktestReport
                                              ↓
                                  calibrate-trend-strategy → best_config.json
```

### Live / Paper Service Pipeline
```
MT5 WaveletSignalPanel
      ↓  POST /market-state
Flask Service → WaveletEngine → DeviationEngine → FilterEngine → SignalEngine
                                                                      ↓
                                                               market_state JSON
                                                              (trend, deviation,
                                                               filter, signal)
```

### Final Gate
```
BacktestReport + PaperJournal + WalkForward
      ↓
evaluate_gate(GateMetrics, GateConfig) → GateResult {PASS|REVIEW|FAIL}
```

### Legacy Research Pipeline
```
ParameterMatrix → PipelineConfig[] → ExperimentOrchestrator → ExperimentReport[]
                                                                      ↓
                             WalkForwardValidator → ValidationReport (IS/OOS/WF/Monte Carlo)
                                                                      ↓
                             ParameterOptimizer → OptimizationReport
```

---

## CLI Usage

```bash
# Audit trend quality
wavelet-research trend-audit --ticks data.csv --wavelet db4 --window 512 --output report.json

# Calibrate thresholds from historical data
wavelet-research calibrate-trend-strategy --ticks data.csv --symbol EURUSD --output config.json

# Final statistical gate
wavelet-research final-gate \
  --trades 100 --profit-factor 1.5 --expectancy 2.0 \
  --max-drawdown-val 50 --gross-profit 500 \
  --mc-survival 0.8 --wf-stability 0.7 --paper-consistency 0.8 \
  --output gate_report.json

# Run research experiments
wavelet-research research --ticks data.csv --wavelets haar,db4 --windows 256,512

# Optimize parameters
wavelet-research optimize --ticks data.csv --random --max-iter 200 --min-trades 10

# Walk-forward validation
wavelet-research validate --ticks data.csv --wavelet haar --window 256 --folds 5

# Paper trading
wavelet-research paper-trade --ticks data.csv --balance 10000 --output journal.csv
```

---

## Metrics Computed

| Category | Metrics |
|----------|---------|
| Performance | Profit Factor, Expectancy, Total PnL, Win Rate |
| Risk | Max Drawdown, Average MAE, Average MFE |
| Risk-Adjusted | Sharpe Ratio, Sortino Ratio, Recovery Factor |
| Stability | Stability Score (R² of equity curve) |
| Validation | OOS Efficiency, Walk-Forward Fold Results |
| Robustness | Monte Carlo percentiles, P(profit), Original vs Median |
| Sensitivity | Coefficient of Variation per parameter |

---

## Quality

- **680 tests** covering all modules (2 skipped — require live MT5 connection)
- **Deterministic** — all tests are repeatable with fixed seeds
- **Isolated** — no test depends on external state
- **Fast** — full suite in ~44 seconds
- **Contract tests** — validate engine interfaces and HTTP API
- **Regression tests** — no data leakage in splits (Story 08)
- **Story-level test files** — `test_story_18_trend_quality.py` … `test_story_24_to_29.py`

---

## Dependencies

```toml
[project]
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.26",
    "pandas>=2.2",
    "PyWavelets>=1.6",
]
```

---

## Not Implemented (by design)

- MT5 live connection (requires MetaTrader5 package)
- AI/ML optimization (Bayesian, genetic)
- Visualization dashboards
- Database persistence (uses CSV/Parquet)
- Multi-threading (parallel-safe design but sequential execution)

---

## Next Steps

1. **Collect real tick data** and build `DeviationStatsIndex` (run `DeviationStatsCollector`)
2. **Run calibration** on collected data: `wavelet-research calibrate-trend-strategy`
3. **Paper trade** with calibrated config and evaluate `AcceptanceGate`
4. **Final gate** once paper run meets acceptance criteria
5. **Compile MQL5 panels** (`WaveletSignalPanel.mq5`, `WaveletOscillator.mq5`) in MetaEditor
6. Connect `RiskConfig` to MT5 EA via service `/risk-check` endpoint
