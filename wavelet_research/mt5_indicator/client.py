"""HTTP client for the MT5 Thin Indicator.

Handles the full request-response cycle:
  1. Serialise ticks to JSON
  2. POST to /wavelet
  3. Parse and validate the response
  4. Map errors to ConnectionStatus

This is the only module that performs I/O.
All parsing and validation is delegated to parser.py.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request

from wavelet_research.mt5_indicator.config import IndicatorConfig
from wavelet_research.mt5_indicator.models import (
    ConnectionStatus,
    IndicatorResponse,
    TickPayload,
)
from wavelet_research.mt5_indicator.parser import (
    ResponseParseError,
    parse_json_bytes,
    validate_wavelet_response,
)

logger = logging.getLogger(__name__)

_CONTENT_TYPE = "application/json"


class WaveletServiceClient:
    """HTTP client that sends ticks to the Wavelet Service and returns parsed data.

    Uses only the Python standard-library ``urllib`` to avoid adding HTTP
    library dependencies — MT5 environments may have constrained package sets.

    Parameters
    ----------
    config : IndicatorConfig
        Indicator configuration (URL, timeout, window).
    """

    def __init__(self, config: IndicatorConfig) -> None:
        self._config = config

    @property
    def config(self) -> IndicatorConfig:
        """Client configuration.

        Returns
        -------
        IndicatorConfig
            Indicator configuration.
        """
        return self._config

    def fetch(self, ticks: list[TickPayload]) -> IndicatorResponse:
        """Send ticks to the service and return the parsed indicator response.

        Parameters
        ----------
        ticks : list[TickPayload]
            Latest ticks to send. Length should equal config.tick_window.

        Returns
        -------
        IndicatorResponse
            Result with status, data, and elapsed time.
            Never raises — all errors are captured in the status field.
        """
        start_ns = time.perf_counter_ns()

        try:
            raw_response = self._post(ticks)
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000

            data_dict = parse_json_bytes(raw_response)
            parsed = validate_wavelet_response(data_dict, expected_length=len(ticks))

            logger.info(
                "POST /wavelet ticks=%d elapsed_ms=%.2f response_bytes=%d",
                len(ticks), elapsed_ms, len(raw_response),
            )
            return IndicatorResponse(
                status=ConnectionStatus.CONNECTED,
                data=parsed,
                elapsed_ms=elapsed_ms,
            )

        except TimeoutError:
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
            logger.warning("POST /wavelet timeout after %.1f ms", elapsed_ms)
            return IndicatorResponse(
                status=ConnectionStatus.TIMEOUT,
                data=None,
                elapsed_ms=elapsed_ms,
            )

        except urllib.error.URLError as exc:
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
            logger.error("POST /wavelet service offline: %s", exc)
            return IndicatorResponse(
                status=ConnectionStatus.SERVICE_OFFLINE,
                data=None,
                elapsed_ms=elapsed_ms,
            )

        except ResponseParseError as exc:
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
            logger.error("POST /wavelet invalid response: %s", exc)
            return IndicatorResponse(
                status=ConnectionStatus.INVALID_RESPONSE,
                data=None,
                elapsed_ms=elapsed_ms,
            )

    def check_health(self) -> ConnectionStatus:
        """Probe the /health endpoint to confirm the service is alive.

        Returns
        -------
        ConnectionStatus
            CONNECTED if the service responds, SERVICE_OFFLINE otherwise.
        """
        timeout = self._config.request_timeout_seconds
        try:
            req = urllib.request.Request(self._config.health_endpoint, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                _ = resp.read()
            return ConnectionStatus.CONNECTED
        except (urllib.error.URLError, TimeoutError, OSError):
            return ConnectionStatus.SERVICE_OFFLINE

    def _post(self, ticks: list[TickPayload]) -> bytes:
        """Perform the raw HTTP POST.

        Parameters
        ----------
        ticks : list[TickPayload]
            Ticks to send.

        Returns
        -------
        bytes
            Raw response body.

        Raises
        ------
        TimeoutError
            On socket/read timeout.
        urllib.error.URLError
            On connection failure.
        """
        payload = json.dumps({"ticks": [t.to_dict() for t in ticks]}).encode("utf-8")
        req = urllib.request.Request(
            self._config.wavelet_endpoint,
            data=payload,
            headers={"Content-Type": _CONTENT_TYPE},
            method="POST",
        )
        timeout = self._config.request_timeout_seconds
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except TimeoutError:
            raise
        except urllib.error.URLError:
            raise
