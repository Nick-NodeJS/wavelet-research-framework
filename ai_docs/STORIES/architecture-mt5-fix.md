# STORY-018 — MT5 Bridge Architecture Fix

## Type

Architecture / Integration

## Priority

Critical

---

# Problem

During integration testing the following architectural limitation was discovered.

Python Service is reachable from MT5 Expert Advisors:

- ✅ HTTP 200
- ✅ JSON response received

However, `WebRequest()` inside an Indicator always fails with:

```
Error 4014
```

This is an MT5 platform limitation:

> Indicators are not allowed to execute `WebRequest()`.

Therefore the current architecture cannot work regardless of implementation quality.

---

# Goal

Move all HTTP communication from the Indicator into a dedicated MT5 Expert Advisor while keeping the Indicator responsible only for visualization.

---

# Current Architecture

```
Indicator
    │
WebRequest()
    │
Python Service
```

❌ Invalid architecture

---

# Target Architecture

```
                +----------------------+
                |   Python Service     |
                +----------+-----------+
                           ^
                           |
                        HTTP/JSON
                           |
                +----------+-----------+
                |    WaveletBridgeEA   |
                +----------+-----------+
                           |
                    Shared State
                           |
                +----------+-----------+
                | WaveletTrendIndicator|
                +----------------------+
```

---

# Scope

## 1. Create WaveletBridgeEA

Create

```
mql5/WaveletBridgeEA.mq5
```

Responsibilities:

- communicate with Python Service
- own all WebRequest() logic
- perform health checks
- reconnect automatically
- cache latest successful response
- execute requests only on a new bar

---

## 2. Shared State Layer

Create a lightweight communication layer between EA and Indicator.

The implementation may use:

- Global Variables
- Files
- Custom Buffers
- Chart Objects

Implementation choice is left to the developer.

The communication layer must expose:

```
trend[]
relativeDeviation[]
confidence
lastUpdate
status
```

---

## 3. Refactor Indicator

Remove completely:

- HTTP
- JSON
- WebRequest
- timeout handling
- reconnect logic

Indicator responsibilities become only:

- read shared state
- render trend
- render deviation
- render status

No networking logic must remain inside the Indicator.

---

## 4. Bridge Polling

EA should request data

- only on a new completed candle
- not on every tick

Pseudo flow

```
New Bar

↓

HTTP Request

↓

Python

↓

Parse JSON

↓

Update Shared State

↓

Indicator redraw
```

---

## 5. Connection Monitoring

EA must expose

```
Connected

Disconnected

Last Update

Latency

Last HTTP Code
```

Indicator should display current connection state.

---

# Acceptance Criteria

## AC-1

Python Service Health endpoint

```
GET /health
```

returns

```
HTTP 200
```

---

## AC-2

EA successfully calls

```
POST /wavelet
```

and receives valid JSON.

---

## AC-3

Indicator contains zero usages of

```
WebRequest()
```

---

## AC-4

Trend is displayed correctly using data provided by EA.

---

## AC-5

Stopping Python Service

- does not freeze MT5
- indicator keeps last trend
- connection status becomes Disconnected

---

## AC-6

Restarting Python Service restores communication automatically.

---

## AC-7

One HTTP request maximum per completed bar.

---

# Non Functional Requirements

- No HTTP inside Indicator
- No blocking operations inside Indicator
- Automatic reconnect
- Minimal CPU usage
- Reuse existing parsing logic where possible
- Backward compatible Python API

---

# Out of Scope

- Trading signals
- Strategy logic
- Entry/Exit rules
- Order execution
- Wavelet algorithm changes

---

# Definition of Done

- WaveletBridgeEA implemented
- Indicator refactored into renderer only
- Successful communication with Python
- Trend rendered on MT5 chart
- Connection survives Python restart
- Existing tests remain green

---

# Notes

This Story fixes an architectural issue discovered during real integration testing.

The limitation is imposed by the MT5 platform itself:

```
Indicators cannot execute WebRequest().
```

After this Story the architecture becomes:

```
Python Service
        │
        ▼
WaveletBridgeEA
        │
        ▼
WaveletTrendIndicator
        │
        ▼
MT5 Chart
```

This separation also improves maintainability:

- Python → computation
- EA → integration
- Indicator → visualization