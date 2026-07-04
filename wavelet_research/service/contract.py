"""Frozen HTTP API contract for the Wavelet Service.

Defines the canonical JSON schema for the /wavelet endpoint request and
response. Used for contract validation in tests and integration checks.

The contract is intentionally kept as plain dicts so it requires no
external JSON-schema library — stdlib only.
"""

from __future__ import annotations

_VERSION = "1.0"

#: Canonical field names expected in a tick object
TICK_REQUIRED_FIELDS: frozenset[str] = frozenset({"bid", "ask"})
TICK_OPTIONAL_FIELDS: frozenset[str] = frozenset({"time", "mid"})
TICK_ALL_FIELDS: frozenset[str] = TICK_REQUIRED_FIELDS | TICK_OPTIONAL_FIELDS

#: Canonical field names expected in the /wavelet response
WAVELET_RESPONSE_FIELDS: frozenset[str] = frozenset({
    "trend",
    "relative_deviation",
    "z_score",
    "energy",
    "noise",
})

#: Canonical field names expected in the /health response
HEALTH_RESPONSE_FIELDS: frozenset[str] = frozenset({"status", "wavelet", "version"})

#: The only supported wavelet for the service
SUPPORTED_WAVELET: str = "db4"

#: Contract version — bump when the API changes
CONTRACT_VERSION: str = _VERSION


def validate_request_contract(body: object) -> list[str]:
    """Validate a parsed request body against the contract.

    Does NOT check business logic (min ticks, price validity) — that
    is handled by :mod:`wavelet_research.service.validation`.

    Parameters
    ----------
    body : object
        Parsed JSON body.

    Returns
    -------
    list[str]
        List of violation messages. Empty list means contract-compliant.
    """
    violations: list[str] = []

    if not isinstance(body, dict):
        return ["Request body must be a JSON object"]

    if "ticks" not in body:
        violations.append("Missing required field 'ticks'")
        return violations

    if not isinstance(body["ticks"], list):
        violations.append("'ticks' must be an array")
        return violations

    for i, tick in enumerate(body["ticks"]):
        if not isinstance(tick, dict):
            violations.append(f"ticks[{i}] must be an object")
            continue
        for field in TICK_REQUIRED_FIELDS:
            if field not in tick:
                violations.append(f"ticks[{i}] missing required field '{field}'")

    return violations


def validate_response_contract(body: object) -> list[str]:
    """Validate a parsed response body against the contract.

    Parameters
    ----------
    body : object
        Parsed JSON response.

    Returns
    -------
    list[str]
        List of violation messages. Empty list means contract-compliant.
    """
    violations: list[str] = []

    if not isinstance(body, dict):
        return ["Response body must be a JSON object"]

    missing = WAVELET_RESPONSE_FIELDS - set(body.keys())
    if missing:
        violations.append(f"Response missing fields: {sorted(missing)}")

    for field in WAVELET_RESPONSE_FIELDS:
        if field not in body:
            continue
        if not isinstance(body[field], list):
            violations.append(f"Response field '{field}' must be an array")

    if not violations:
        lengths = {field: len(body[field]) for field in WAVELET_RESPONSE_FIELDS}  # type: ignore[arg-type]
        unique = set(lengths.values())
        if len(unique) != 1:
            violations.append(f"Response arrays have unequal lengths: {lengths}")

    return violations


def validate_health_contract(body: object) -> list[str]:
    """Validate a parsed /health response against the contract.

    Parameters
    ----------
    body : object
        Parsed JSON health response.

    Returns
    -------
    list[str]
        List of violation messages.
    """
    violations: list[str] = []

    if not isinstance(body, dict):
        return ["Health response must be a JSON object"]

    missing = HEALTH_RESPONSE_FIELDS - set(body.keys())
    if missing:
        violations.append(f"Health response missing fields: {sorted(missing)}")
        return violations

    if body.get("status") != "ok":
        violations.append(f"Expected status 'ok', got {body.get('status')!r}")

    if body.get("wavelet") != SUPPORTED_WAVELET:
        violations.append(
            f"Expected wavelet '{SUPPORTED_WAVELET}', got {body.get('wavelet')!r}"
        )

    return violations
