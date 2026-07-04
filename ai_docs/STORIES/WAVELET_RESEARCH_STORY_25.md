# WAVELET_RESEARCH_STORY_25.md

# Story

Extend the Python service to return full real-time market state, not only wavelet values.

# Goal

MT5 should receive actionable state from Python without duplicating calculations in MQL5.

# Scope

Extend service contract.

# Endpoint

Add or extend:

```text
POST /market-state
```

# Request

```json
{
  "symbol": "EURUSD",
  "ticks": [],
  "config_id": "default"
}
```

# Response

```json
{
  "trend": [],
  "deviation": {
    "normalized": 1.4,
    "side": "above"
  },
  "historical_stats": {
    "sample_size": 1200,
    "return_to_trend_probability": 0.68,
    "median_bars_to_return": 7
  },
  "filter": {
    "can_trade": true,
    "reasons": []
  },
  "signal": {
    "side": "SELL",
    "confidence": 0.71,
    "reason": "above_trend_return_probability"
  }
}
```

# Constraints

- MQL5 remains thin.
- Python owns all logic.
- Service response must be stable and versioned.

# Tests

Add tests for:

- endpoint schema;
- invalid request;
- no-trade response;
- signal response;
- latency smoke test.

# Acceptance Criteria

- MT5 can request current market state from Python.
- Response contains trend, deviation, stats, filters and signal.
- Contract tests cover the response.
