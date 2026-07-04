"""Request validation for the Wavelet Service."""

from __future__ import annotations

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


def parse_wavelet_request(body: object, min_ticks: int) -> WaveletRequest:
    """Parse and validate the full /wavelet request body.

    Parameters
    ----------
    body : object
        Parsed JSON body (expected dict).
    min_ticks : int
        Minimum required number of ticks (equal to engine window size).

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

    if len(ticks) < min_ticks:
        raise RequestValidationError(
            f"Insufficient history: {len(ticks)} ticks provided, "
            f"minimum required is {min_ticks}",
            http_status=422,
        )

    return WaveletRequest(ticks=ticks)
