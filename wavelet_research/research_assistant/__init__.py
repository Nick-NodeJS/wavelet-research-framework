"""AI Research Assistant for experiment analysis."""

from wavelet_research.research_assistant.analyzer import ExperimentAnalyzer
from wavelet_research.research_assistant.comparator import ParameterComparator
from wavelet_research.research_assistant.models import (
    AnalysisReport,
    ComparisonReport,
    FailureExplanation,
    Recommendation,
)

__all__ = [
    "AnalysisReport",
    "ComparisonReport",
    "ExperimentAnalyzer",
    "FailureExplanation",
    "ParameterComparator",
    "Recommendation",
]
