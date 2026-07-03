"""Wavelet Research Framework.

Provides a reusable research runner for evaluating experiment configurations
on historical tick data.
"""

from wavelet_research.research.config import ExperimentConfig
from wavelet_research.research.experiment import ExperimentResult
from wavelet_research.research.metrics import MetricsCollector, TradeMetrics
from wavelet_research.research.ranking import rank_results
from wavelet_research.research.runner import ResearchRunner
from wavelet_research.research.storage import ResultStorage

__all__ = [
    "ExperimentConfig",
    "ExperimentResult",
    "MetricsCollector",
    "TradeMetrics",
    "ResearchRunner",
    "ResultStorage",
    "rank_results",
]
