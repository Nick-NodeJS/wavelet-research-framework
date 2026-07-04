# Wavelet Research Framework — Remaining Implementation Roadmap

Date: 2026-07-04
Focus: Adaptive Trend Engine, not broad ML research.

## Product Goal

Build a practical trading system that:

1. Constructs a correct causal trend in real time.
2. Measures price deviation from that trend.
3. Uses historical behavior around similar deviations to filter noise and bad situations.
4. Produces simple, explainable trading decisions for MT5.

The project should not turn into generic research infrastructure. New work is allowed only if it improves one of:

- trend quality;
- deviation quality;
- trade/no-trade filtering quality;
- live execution safety.

## Current Audit Summary

The uploaded project already contains enough infrastructure:

- causal wavelet engine;
- data ingestion;
- signal engine;
- backtesting;
- walk-forward validation;
- optimizer;
- MT5 indicator/EA modules;
- paper trading;
- profiling;
- research assistant;
- orchestrator;
- tests.

Therefore the next work must focus on compact, practical trading logic around the trend.

## Remaining Stories

| Story | Name | Purpose |
|---|---|---|
| 18 | Trend Quality Audit | Verify that the trend is stable, causal, useful and not visually misleading. |
| 19 | Normalized Deviation Engine | Measure price distance from trend in volatility-aware units. |
| 20 | Historical Deviation Statistics | Build lookup statistics for similar historical deviations. |
| 21 | No-Trade Filter Engine | Detect statistically bad or noisy situations. |
| 22 | Trend-Following Entry Rules | Produce simple entry decisions around trend/deviation. |
| 23 | Exit and Return-to-Trend Rules | Define exits: return to trend, invalidation, time, risk. |
| 24 | Strategy Calibration CLI | Calibrate thresholds from historical data without overfitting. |
| 25 | Real-Time Market State Service | Return trend/deviation/statistics/filter/signal per latest window. |
| 26 | MT5 Signal Panel | Display actionable state in MT5 without duplicating Python logic. |
| 27 | Paper Trading Acceptance Run | Validate live-like behavior before real execution. |
| 28 | Production EA Safety Layer | Safe MT5 execution with strict controls. |
| 29 | Final Statistical Gate | Decide objectively whether strategy is tradable. |

## Non-Goals

Do not add neural networks, complex black-box ML, unrelated indicators, large research dashboards, or generic strategy mining unless a specific story explicitly needs it.
