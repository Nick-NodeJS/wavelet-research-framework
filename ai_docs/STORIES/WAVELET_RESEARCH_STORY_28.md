# WAVELET_RESEARCH_STORY_28.md

# Story

Add production safety controls to the MT5 Expert Advisor.

# Goal

Real execution must be protected from service errors, bad market states, excessive risk and repeated orders.

# Scope

Extend MT5 EA and Python-side contract if needed.

# Safety Controls

Implement:

```text
max risk per trade
max daily loss
max open positions
symbol whitelist
spread limit
time/session filter
cooldown after loss
cooldown after service error
service heartbeat
duplicate order guard
manual kill switch
paper/live mode flag
```

# Execution Rule

EA may execute only if:

```text
service online
signal confidence >= threshold
filter.can_trade == true
risk checks pass
no duplicate position
```

# Logging

Log every blocked action with reason.

# Tests

Add Python-side tests for risk contract and documented MT5 manual test checklist.

# Acceptance Criteria

- EA cannot trade when service is offline.
- EA cannot exceed risk limits.
- EA logs every skipped trade reason.
- Live mode must be explicitly enabled.
