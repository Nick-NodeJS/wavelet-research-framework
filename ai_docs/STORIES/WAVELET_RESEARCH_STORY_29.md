# WAVELET_RESEARCH_STORY_29.md

# Story

Create a final objective statistical gate for deciding whether the strategy is tradable.

# Goal

Avoid subjective “looks good” decisions. The strategy must pass explicit statistical and operational criteria.

# Scope

Add command:

```bash
wavelet-research final-gate --config config/calibrated.json --ticks data.csv --paper-journal journal.csv --output reports/final_gate.json
```

# Gate Criteria

At minimum:

```text
sufficient sample size
positive expectancy
profit factor threshold
max drawdown threshold
walk-forward stability
Monte Carlo survival probability
reasonable average holding time
no overdependence on one market period
paper trading consistency
```

# Output

```json
{
  "decision": "PASS|FAIL|REVIEW",
  "reasons": [],
  "metrics": {},
  "next_action": "paper_trade_more|allow_small_live|reject_config"
}
```

# Tests

Add tests for:

- pass case;
- fail by drawdown;
- fail by low sample size;
- review case;
- output schema.

# Acceptance Criteria

- Final gate produces one clear decision.
- Real trading is not allowed unless final gate passes or user explicitly overrides.
- Failure reasons are actionable.
