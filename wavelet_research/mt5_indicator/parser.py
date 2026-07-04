"""JSON response parser and validator for the MT5 Indicator client.

Pure functions — no I/O, no state.
"""

from __future__ import annotations

import json

from wavelet_research.mt5_indicator.models import ParsedWaveletResponse

_REQUIRED_FIELDS = frozenset(
    {"trend", "relative_deviation", "z_score", "energy", "noise"}
)


class ResponseParseError(ValueError):
    """Raised when a service response cannot be parsed or validated.

    Parameters
    ----------
    message : str
        Description of the parse failure.
    """


def parse_json_bytes(raw: bytes) -> dict[str, object]:
    """Decode and parse raw HTTP response bytes as JSON.

    Parameters
    ----------
    raw : bytes
        Raw HTTP response body.

    Returns
    -------
    dict[str, object]
        Parsed JSON object.

    Raises
    ------
    ResponseParseError
        If the bytes cannot be decoded or are not a JSON object.
    """
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ResponseParseError(f"Response is not valid UTF-8: {exc}") from exc

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ResponseParseError(f"Response is not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ResponseParseError(
            f"Expected JSON object, got {type(data).__name__}"
        )

    return data  # type: ignore[return-value]


def validate_wavelet_response(
    data: dict[str, object],
    expected_length: int | None = None,
) -> ParsedWaveletResponse:
    """Validate and convert a parsed JSON dictionary into a typed response.

    Parameters
    ----------
    data : dict[str, object]
        Parsed JSON from the /wavelet endpoint.
    expected_length : int | None
        If provided, each array must have this exact length.

    Returns
    -------
    ParsedWaveletResponse
        Validated and typed response.

    Raises
    ------
    ResponseParseError
        If any required field is missing, not a list, contains non-numeric
        values, or arrays have wrong length.
    """
    missing = _REQUIRED_FIELDS - set(data.keys())
    if missing:
        raise ResponseParseError(
            f"Response missing required fields: {sorted(missing)}"
        )

    arrays: dict[str, tuple[float, ...]] = {}
    for field in _REQUIRED_FIELDS:
        raw_array = data[field]
        if not isinstance(raw_array, list):
            raise ResponseParseError(
                f"Field '{field}' must be an array, got {type(raw_array).__name__}"
            )
        try:
            converted = tuple(float(v) for v in raw_array)
        except (TypeError, ValueError) as exc:
            raise ResponseParseError(
                f"Field '{field}' contains non-numeric value: {exc}"
            ) from exc
        arrays[field] = converted

    lengths = {name: len(arr) for name, arr in arrays.items()}
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        raise ResponseParseError(
            f"All arrays must have equal length, got: {lengths}"
        )

    actual_length = next(iter(unique_lengths))
    if expected_length is not None and actual_length != expected_length:
        raise ResponseParseError(
            f"Expected array length {expected_length}, got {actual_length}"
        )

    return ParsedWaveletResponse(
        trend=arrays["trend"],
        relative_deviation=arrays["relative_deviation"],
        z_score=arrays["z_score"],
        energy=arrays["energy"],
        noise=arrays["noise"],
    )
