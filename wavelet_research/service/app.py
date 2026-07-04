"""Flask application factory for the Wavelet Service."""

from __future__ import annotations

import logging
import time

from flask import Flask, Response, jsonify, request

from wavelet_research.deviation.core import DeviationEngine
from wavelet_research.deviation_stats.models import DeviationQueryResult
from wavelet_research.engine.core import WaveletEngine
from wavelet_research.engine.decomposition import TrendMode
from wavelet_research.engine.models import Tick
from wavelet_research.filters.engine import FilterEngine
from wavelet_research.filters.models import FilterConfig
from wavelet_research.service.config import ServiceConfig
from wavelet_research.service.models import HealthResponse
from wavelet_research.service.processor import _parse_timestamp, process_ticks
from wavelet_research.service.validation import (
    RequestValidationError,
    parse_wavelet_request,
)
from wavelet_research.signal.config import SignalConfig
from wavelet_research.signal.core import SignalEngine
from wavelet_research.trend_quality.audit import TrendAuditor
from wavelet_research.trend_quality.models import TrendQualityState

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
    _register_market_state(app, config, engine_config)

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

        trend_mode = TrendMode(wavelet_request.trend_mode)
        result = process_ticks(
            wavelet_request.ticks,
            engine_config,
            trend_mode,
            wavelet_override=wavelet_request.wavelet,
            window_override=wavelet_request.window,
            level_override=wavelet_request.level,
        )

        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        effective_window = wavelet_request.window or engine_config.window
        effective_wavelet = wavelet_request.wavelet or engine_config.wavelet
        effective_level = wavelet_request.level or engine_config.level
        logger.info(
            "POST /wavelet window=%d wavelet=%s level=%d mode=%s elapsed_ms=%.2f",
            effective_window, effective_wavelet, effective_level,
            trend_mode.value, elapsed_ms,
        )

        return jsonify(result.to_dict())


def _register_market_state(app: Flask, config: ServiceConfig, engine_config) -> None:
    """Register the /market-state endpoint (Story 25)."""

    deviation_engine = DeviationEngine()
    signal_engine = SignalEngine(SignalConfig())
    filter_engine = FilterEngine(FilterConfig())
    auditor = TrendAuditor(engine_config)

    @app.post("/market-state")
    def market_state() -> tuple[Response, int] | Response:
        """Return full real-time market state: trend, deviation, stats, filter, signal.

        Request body:
            { "symbol": "EURUSD", "ticks": [...], "config_id": "default" }

        Returns
        -------
        Response
            JSON with trend array, deviation, historical_stats, filter, and signal.
        """
        start = time.perf_counter_ns()

        body = request.get_json(silent=True)
        if body is None:
            return jsonify({"error": "Invalid JSON"}), 400

        try:
            wavelet_request = parse_wavelet_request(body, min_ticks=config.window)
        except RequestValidationError as exc:
            return jsonify({"error": str(exc)}), exc.http_status

        ticks_list = list(wavelet_request.ticks)
        wavelet_result = process_ticks(wavelet_request.ticks, engine_config)

        last_tr = ticks_list[-1]
        last_tick = Tick(
            time=_parse_timestamp(last_tr.time),
            bid=last_tr.bid,
            ask=last_tr.ask,
            mid=last_tr.mid,
            spread=last_tr.ask - last_tr.bid,
        )

        from wavelet_research.engine.models import WaveletPoint
        last_wp = WaveletPoint(
            trend=wavelet_result.trend[-1],
            deviation=wavelet_result.relative_deviation[-1],
            z_score=wavelet_result.z_score[-1],
            slope=wavelet_result.trend[-1] - wavelet_result.trend[-2] if len(wavelet_result.trend) > 1 else 0.0,
            energy=wavelet_result.energy[-1],
            noise=wavelet_result.noise[-1],
        )

        dp = deviation_engine.compute(last_wp, last_tick)

        empty_stats = DeviationQueryResult(
            sample_size=0,
            return_to_trend_probability=0.0,
            median_bars_to_return=0.0,
            expected_return=0.0,
            expected_adverse_excursion=0.0,
            confidence_level="insufficient",
        )

        recent_trends = list(wavelet_result.trend[-10:])
        tq_state = auditor.assess_current(recent_trends)

        filter_result = filter_engine.evaluate(dp, tq_state, empty_stats)
        decision = signal_engine.decide_with_context(last_wp, dp, empty_stats, filter_result)

        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        logger.info("POST /market-state elapsed_ms=%.2f", elapsed_ms)

        return jsonify({
            "trend": wavelet_result.trend,
            "deviation": {
                "normalized": dp.z_score,
                "side": dp.side.value,
            },
            "historical_stats": {
                "sample_size": empty_stats.sample_size,
                "return_to_trend_probability": empty_stats.return_to_trend_probability,
                "median_bars_to_return": empty_stats.median_bars_to_return,
                "confidence_level": empty_stats.confidence_level,
            },
            "filter": filter_result.to_dict(),
            "signal": {
                "side": decision.signal.value,
                "confidence": decision.confidence,
                "reason": decision.reason,
            },
            "latency_ms": elapsed_ms,
        })
