# Implementation Note for Developer Agent

Before each story implementation:

1. Read `IMPLEMENTATION_REPORT.md` from the main repo.
2. Read `ai_docs/REMAINING_ROADMAP.md`.
3. Read the current story completely.
4. Inspect existing modules before adding new abstractions.
5. Prefer extending existing `signal`, `backtest`, `service`, `paper_trading`, and `validation` modules over creating parallel frameworks.
6. Keep MQL5 thin. Python owns trend, deviation, statistics, filters and signal decisions.
7. Add tests with every implementation.

Main constraint: do not over-engineer. The product is an Adaptive Trend Engine, not a generic strategy-mining platform.
