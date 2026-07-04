"""Request validation for the Wavelet Service."""

from __future__ import annotations

from wavelet_research.engine.config import (
    SUPPORTED_LEVELS,
    SUPPORTED_WAVELETS,
    SUPPORTED_WINDOWS,
)
from wavelet_research.engine.decomposition import SUPPORTED_TREND_MODES
from wavelet_research.service.models import TickRequest, WaveletRequest


class RequestValidationError(ValueError):
    """Raised when an incoming request fails validation.

    Parameters
    ----------
    message : str
        Human-readable description of the failure.
    http_status : int
        Suggested HTTP status code to return (400 or 422).
    """

    def __init__(self, message: str, http_status: int = 400) -> None:
        super().__init__(message)
        self.http_status = http_status


def parse_tick(raw: object, index: int) -> TickRequest:
    """Parse and validate a single tick from raw JSON.

    Parameters
    ----------
    raw : object
        Raw JSON object (expected dict).
    index : int
        Position in the ticks array (for error messages).

    Returns
    -------
    TickRequest
        Validated tick.

    Raises
    ------
    RequestValidationError
        If the tick is malformed or has invalid prices.
    """
    if not isinstance(raw, dict):
        raise RequestValidationError(
            f"ticks[{index}] must be an object, got {type(raw).__name__}"
        )

    for field in ("bid", "ask"):
        if field not in raw:
            raise RequestValidationError(
                f"ticks[{index}] missing required field '{field}'"
            )

    try:
        bid = float(raw["bid"])
        ask = float(raw["ask"])
    except (TypeError, ValueError) as exc:
        raise RequestValidationError(
            f"ticks[{index}] has non-numeric price: {exc}"
        ) from exc

    if bid <= 0 or ask <= 0:
        raise RequestValidationError(
            f"ticks[{index}] has invalid prices: bid={bid}, ask={ask}"
        )
    if ask < bid:
        raise RequestValidationError(
            f"ticks[{index}] ask ({ask}) < bid ({bid})"
        )

    mid_raw = raw.get("mid")
    mid = float(mid_raw) if mid_raw is not None else (bid + ask) / 2.0
    time_val = str(raw.get("time", ""))

    return TickRequest(time=time_val, bid=bid, ask=ask, mid=mid)


def _parse_optional_window(body: dict) -> int | None:
    """Parse and validate the optional 'window' override field.

    Parameters
    ----------
    body : dict
        Parsed request body.

    Returns
    -------
    int | None
        Validated window size, or None if not provided.

    Raises
    ------
    RequestValidationError
        If provided but invalid.
    """
    raw = body.get("window")
    if raw is None:
        return None
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise RequestValidationError(
            f"'window' must be an integer, got {type(raw).__name__}"
        )
    if raw not in SUPPORTED_WINDOWS:
        raise RequestValidationError(
            f"Invalid window {raw}. Supported values: {sorted(SUPPORTED_WINDOWS)}"
        )
    return raw


def _parse_optional_wavelet(body: dict) -> str | None:
    """Parse and validate the optional 'wavelet' override field.

    Parameters
    ----------
    body : dict
        Parsed request body.

    Returns
    -------
    str | None
        Validated wavelet name, or None if not provided.

    Raises
    ------
    RequestValidationError
        If provided but invalid.
    """
    raw = body.get("wavelet")
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise RequestValidationError(
            f"'wavelet' must be a string, got {type(raw).__name__}"
        )
    wavelet = raw.lower()
    if wavelet not in SUPPORTED_WAVELETS:
        raise RequestValidationError(
            f"Invalid wavelet {raw!r}. Supported values: {sorted(SUPPORTED_WAVELETS)}"
        )
    return wavelet


def _parse_optional_level(body: dict) -> int | None:
    """Parse and validate the optional 'level' override field.

    Parameters
    ----------
    body : dict
        Parsed request body.

    Returns
    -------
    int | None
        Validated decomposition level, or None if not provided.

    Raises
    ------
    RequestValidationError
        If provided but invalid.
    """
    raw = body.get("level")
    if raw is None:
        return None
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise RequestValidationError(
            f"'level' must be an integer, got {type(raw).__name__}"
        )
    if raw not in SUPPORTED_LEVELS:
        raise RequestValidationError(
            f"Invalid level {raw}. Supported values: {sorted(SUPPORTED_LEVELS)}"
        )
    return raw


def parse_wavelet_request(body: object, min_ticks: int) -> WaveletRequest:
    """Parse and validate the full /wavelet request body.

    Per-request 'window', 'wavelet', 'level', and 'trend_mode' fields are
    optional.  When provided they override the service startup defaults.
    The tick count is checked against the effective window (per-request
    window if provided, otherwise ``min_ticks``).

    Parameters
    ----------
    body : object
        Parsed JSON body (expected dict).
    min_ticks : int
        Minimum required ticks based on service default window.

    Returns
    -------
    WaveletRequest
        Validated request.

    Raises
    ------
    RequestValidationError
        If the request is malformed or has insufficient history.
    """
    if not isinstance(body, dict):
        raise RequestValidationError("Request body must be a JSON object")

    if "ticks" not in body:
        raise RequestValidationError("Missing required field 'ticks'")

    raw_ticks = body["ticks"]
    if not isinstance(raw_ticks, list):
        raise RequestValidationError("'ticks' must be an array")

    if len(raw_ticks) == 0:
        raise RequestValidationError("'ticks' array is empty")

    # Validate individual ticks first so price errors always return 400
    ticks = tuple(parse_tick(t, i) for i, t in enumerate(raw_ticks))

    # Optional per-request engine overrides — validate before the tick count
    # check so that a bad window value gets a clear 400 rather than a 422
    req_window = _parse_optional_window(body)
    req_wavelet = _parse_optional_wavelet(body)
    req_level = _parse_optional_level(body)

    # Effective minimum = per-request window if provided, else service default
    effective_min = req_window if req_window is not None else min_ticks
    if len(ticks) < effective_min:
        raise RequestValidationError(
            f"Insufficient history: {len(ticks)} ticks provided, "
            f"minimum required is {effective_min}",
            http_status=422,
        )

    raw_mode = body.get("trend_mode", "A2")
    if not isinstance(raw_mode, str):
        raise RequestValidationError(
            "'trend_mode' must be a string, e.g. 'A1', 'A2', 'A3', or 'A4'"
        )
    trend_mode = raw_mode.upper()
    if trend_mode not in SUPPORTED_TREND_MODES:
        raise RequestValidationError(
            f"Invalid trend_mode {raw_mode!r}. "
            f"Supported values: {sorted(SUPPORTED_TREND_MODES)}"
        )

    return WaveletRequest(
        ticks=ticks,
        trend_mode=trend_mode,
        window=req_window,
        wavelet=req_wavelet,
        level=req_level,
    )
