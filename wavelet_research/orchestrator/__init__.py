"""Experiment Orchestrator: end-to-end research pipeline execution."""

from wavelet_research.orchestrator.config import PipelineConfig
from wavelet_research.orchestrator.core import ExperimentOrchestrator
from wavelet_research.orchestrator.matrix import ParameterMatrix
from wavelet_research.orchestrator.results import ExperimentReport

__all__ = [
    "ExperimentOrchestrator",
    "ExperimentReport",
    "ParameterMatrix",
    "PipelineConfig",
]
