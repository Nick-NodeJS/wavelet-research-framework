# WAVELET_RESEARCH_STORY_26.md

# Story

Implement an MT5 visual signal panel using the Python market-state endpoint.

# Goal

The trader should see what the system thinks now: trend, deviation, probability, and whether trading is allowed.

# Scope

Extend MQL5 indicator/client.

# Display

Show:

```text
Trend direction
Deviation side and normalized value
Historical return probability
Expected bars to return
Signal: BUY / SELL / NO_TRADE / HOLD
No-trade reasons
Service status and latency
```

# Rules

- No trading execution in this story.
- No wavelet or signal calculations in MQL5.
- MQL5 only renders service output.

# Error Handling

Display:

```text
SERVICE_OFFLINE
TIMEOUT
INVALID_RESPONSE
LOW_CONFIDENCE
```

# Tests

Where direct MQL5 tests are limited, add parser/client tests in Python and document manual MT5 checklist.

# Acceptance Criteria

- Indicator panel updates in real time.
- User can visually confirm trend/deviation/signal.
- No business logic is implemented in MQL5.
