"""Flask application factory for the Wavelet Service."""

from __future__ import annotations

import logging
import time

from flask import Flask, Response, jsonify, request

from wavelet_research.service.config import ServiceConfig
from wavelet_research.service.models import HealthResponse
from wavelet_research.service.processor import process_ticks
from wavelet_research.service.validation import (
    RequestValidationError,
    parse_wavelet_request,
)

logger = logging.getLogger(__name__)


def create_app(config: ServiceConfig | None = None) -> Flask:
    """Create and configure the Flask application.

    Parameters
    ----------
    config : ServiceConfig | None
        Service configuration. If None, loads from environment.

    Returns
    -------
    Flask
        Configured Flask application.
    """
    if config is None:
        config = ServiceConfig.from_env()

    app = Flask(__name__)
    app.config["WAVELET_SERVICE_CONFIG"] = config

    engine_config = config.to_engine_config()

    _register_health(app, config)
    _register_wavelet(app, config, engine_config)

    logger.info(
        "Wavelet Service created: wavelet=%s window=%d level=%d",
        config.wavelet, config.window, config.level,
    )

    return app


def _register_health(app: Flask, config: ServiceConfig) -> None:
    """Register the /health endpoint."""

    @app.get("/health")
    def health() -> Response:
        """Health check endpoint.

        Returns
        -------
        Response
            JSON with status, wavelet, and version.
        """
        body = HealthResponse(
            status="ok",
            wavelet=config.wavelet,
            version=config.version,
        )
        return jsonify(body.to_dict())


def _register_wavelet(app: Flask, config: ServiceConfig, engine_config) -> None:
    """Register the /wavelet endpoint."""

    @app.post("/wavelet")
    def wavelet() -> tuple[Response, int] | Response:
        """Compute wavelet features for a batch of ticks.

        Expects JSON body:
            { "ticks": [{ "time": "...", "bid": 1.1, "ask": 1.1, "mid": 1.1 }, ...] }

        Returns
        -------
        Response
            JSON with trend, relative_deviation, z_score, energy, noise arrays.
        """
        start = time.perf_counter_ns()

        body = request.get_json(silent=True)
        if body is None:
            logger.warning("Rejected request: invalid JSON")
            return jsonify({"error": "Invalid JSON"}), 400

        try:
            wavelet_request = parse_wavelet_request(body, min_ticks=config.window)
        except RequestValidationError as exc:
            logger.warning("Rejected request: %s", exc)
            return jsonify({"error": str(exc)}), exc.http_status

        result = process_ticks(wavelet_request.ticks, engine_config)

        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        logger.info(
            "POST /wavelet ticks=%d elapsed_ms=%.2f",
            len(wavelet_request.ticks), elapsed_ms,
        )

        return jsonify(result.to_dict())
