"""Single experiment pipeline execution.

Runs one PipelineConfig through the full WaveletEngine → SignalEngine
→ BacktestEngine pipeline.
"""

from __future__ import annotations

import logging

import pandas as pd

from wavelet_research.backtest.core import BacktestEngine
from wavelet_research.backtest.models import BacktestReport
from wavelet_research.engine.core import WaveletEngine
from wavelet_research.orchestrator.config import PipelineConfig
from wavelet_research.orchestrator.results import ExperimentReport
from wavelet_research.signal.core import SignalEngine

logger = logging.getLogger(__name__)


def run_pipeline(
    config: PipelineConfig,
    data: pd.DataFrame,
) -> ExperimentReport:
    """Execute a single experiment pipeline.

    Creates fresh engine instances and runs the full pipeline:
    WaveletEngine → SignalEngine → BacktestEngine.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration.
    data : pd.DataFrame
        Normalized tick data.

    Returns
    -------
    ExperimentReport
        Complete experiment report.
    """
    wavelet_engine = WaveletEngine(config.wavelet_config)
    signal_engine = SignalEngine(config.signal_config)
    backtest_engine = BacktestEngine(config.backtest_config)

    report = backtest_engine.run(data, wavelet_engine, signal_engine)

    return ExperimentReport(config=config, report=report)
