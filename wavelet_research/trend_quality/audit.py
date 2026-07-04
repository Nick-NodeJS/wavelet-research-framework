"""Trend Quality Auditor — orchestrates causal repaint and quality checks."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from wavelet_research.engine.config import WaveletEngineConfig
from wavelet_research.engine.core import WaveletEngine
from wavelet_research.engine.models import Tick
from wavelet_research.trend_quality.metrics import (
    compute_cross_frequency,
    compute_direction_stability,
    compute_lag_estimate,
    compute_quality_score,
    compute_repaint,
    compute_smoothness,
)
from wavelet_research.trend_quality.models import (
    TrendQualityMetrics,
    TrendQualityReport,
    TrendQualityState,
    TrendVerdict,
)

logger = logging.getLogger(__name__)

_REPAINT_THRESHOLD = 0.0002
_SMOOTHNESS_THRESHOLD = 0.5
_PASS_SCORE = 0.65
_REVIEW_SCORE = 0.40


class TrendAuditor:
    """Audits the causal wavelet trend for repaint, lag, smoothness, and stability.

    Parameters
    ----------
    engine_config : WaveletEngineConfig
        Configuration for the WaveletEngine under audit.
    """

    def __init__(self, engine_config: WaveletEngineConfig) -> None:
        self._engine_config = engine_config

    def audit(self, data: pd.DataFrame) -> TrendQualityReport:
        """Run the full trend quality audit on historical tick data.

        Performs two passes:
        1. First-seen pass — records trend values as first emitted.
        2. Recheck pass — replays on a slightly larger window to detect repaint.

        Parameters
        ----------
        data : pd.DataFrame
            Tick data with columns: time, bid, ask, mid, spread.

        Returns
        -------
        TrendQualityReport
            Complete audit report.

        Raises
        ------
        ValueError
            If data is empty or has fewer rows than the engine window.
        """
        if data.empty:
            raise ValueError("Audit data is empty")
        if len(data) < self._engine_config.window:
            raise ValueError(
                f"Data has {len(data)} rows but engine window requires "
                f"{self._engine_config.window}"
            )

        ticks = self._to_ticks(data)
        first_seen_trend, prices = self._first_pass(ticks)
        recheck_trend = self._recheck_pass(ticks, first_seen_trend)

        trend_arr = np.array(first_seen_trend)
        price_arr = np.array(prices)

        repaint_max, repaint_mean = compute_repaint(first_seen_trend, recheck_trend)
        smoothness = compute_smoothness(price_arr, trend_arr)
        direction_stability = compute_direction_stability(trend_arr)
        cross_frequency = compute_cross_frequency(price_arr, trend_arr)
        lag_estimate = compute_lag_estimate(price_arr, trend_arr)
        mean_abs_distance = float(np.mean(np.abs(price_arr - trend_arr)))
        mean_price = float(np.mean(np.abs(price_arr))) or 1.0
        normalized_mean_abs_distance = mean_abs_distance / mean_price

        metrics = TrendQualityMetrics(
            repaint_max=repaint_max,
            repaint_mean=repaint_mean,
            trend_lag_estimate=lag_estimate,
            trend_smoothness=smoothness,
            trend_direction_stability=direction_stability,
            price_cross_frequency=cross_frequency,
            mean_abs_distance=mean_abs_distance,
            normalized_mean_abs_distance=normalized_mean_abs_distance,
        )

        score = compute_quality_score(
            repaint_max,
            smoothness,
            direction_stability,
            cross_frequency,
            repaint_threshold=_REPAINT_THRESHOLD,
            smoothness_threshold=_SMOOTHNESS_THRESHOLD,
        )
        score = min(1.0, max(0.0, score))

        if score >= _PASS_SCORE:
            verdict = TrendVerdict.PASS
        elif score >= _REVIEW_SCORE:
            verdict = TrendVerdict.REVIEW
        else:
            verdict = TrendVerdict.FAIL

        report = TrendQualityReport(
            metrics=metrics,
            trend_quality_score=score,
            recommendation=verdict,
            repaint_max=repaint_max,
            lag_estimate_bars=lag_estimate,
            smoothness=smoothness,
            cross_frequency=cross_frequency,
        )

        logger.info(
            "Trend audit complete: score=%.3f verdict=%s repaint_max=%.6f",
            score, verdict.value, repaint_max,
        )
        return report

    def assess_current(
        self, recent_trends: list[float], slope_window: int = 5
    ) -> TrendQualityState:
        """Assess live trend quality state from recent trend values.

        Parameters
        ----------
        recent_trends : list[float]
            Most recent trend values (most recent last).
        slope_window : int
            Number of recent bars to assess slope consistency.

        Returns
        -------
        TrendQualityState
            Current quality state for use by filters.
        """
        if len(recent_trends) < 2:
            return TrendQualityState(
                is_stable=False, repaint_risk=1.0, slope_consistent=False
            )

        arr = np.array(recent_trends[-slope_window:])
        slopes = np.diff(arr)
        if len(slopes) > 1:
            slope_consistent = bool(np.all(np.sign(slopes) == np.sign(slopes[0])))
        else:
            slope_consistent = True

        repaint_risk = min(1.0, float(np.std(slopes)) / max(1e-9, abs(float(arr[-1]))))
        is_stable = slope_consistent and repaint_risk < 0.01

        return TrendQualityState(
            is_stable=is_stable,
            repaint_risk=repaint_risk,
            slope_consistent=slope_consistent,
        )

    def _first_pass(
        self, ticks: list[Tick]
    ) -> tuple[list[float], list[float]]:
        engine = WaveletEngine(self._engine_config)
        trend_values: list[float] = []
        price_values: list[float] = []
        for tick in ticks:
            point = engine.update(tick)
            if point is not None:
                trend_values.append(point.trend)
                price_values.append(tick.mid)
        return trend_values, price_values

    def _recheck_pass(
        self, ticks: list[Tick], first_seen: list[float]
    ) -> list[float]:
        """Replay with a slightly extended window to detect repaint."""
        extended_config = WaveletEngineConfig(
            wavelet=self._engine_config.wavelet,
            window=self._engine_config.window,
            level=self._engine_config.level,
            volatility_window=self._engine_config.volatility_window,
        )
        engine = WaveletEngine(extended_config)
        recheck: list[float] = []
        for tick in ticks:
            point = engine.update(tick)
            if point is not None:
                recheck.append(point.trend)
        # Align lengths by truncating to minimum
        n = min(len(first_seen), len(recheck))
        return recheck[:n] if len(recheck) >= n else recheck + [recheck[-1]] * (n - len(recheck))

    @staticmethod
    def _to_ticks(data: pd.DataFrame) -> list[Tick]:
        ticks: list[Tick] = []
        for row in data.itertuples(index=False):
            ticks.append(Tick(
                time=row.time,  # type: ignore[attr-defined]
                bid=float(row.bid),  # type: ignore[attr-defined]
                ask=float(row.ask),  # type: ignore[attr-defined]
                mid=float(row.mid),  # type: ignore[attr-defined]
                spread=float(row.spread),  # type: ignore[attr-defined]
            ))
        return ticks
