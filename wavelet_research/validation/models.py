"""Domain models for the validation framework."""

from __future__ import annotations

from dataclasses import dataclass, field

from wavelet_research.validation.metrics import ExtendedMetrics
from wavelet_research.validation.robustness import RobustnessMetrics
from wavelet_research.validation.sensitivity import ParameterSensitivity


@dataclass(frozen=True)
class ValidationSplit:
    """Result of a single validation split (IS or OOS).

    Parameters
    ----------
    split_name : str
        Name or index of the split.
    metrics : ExtendedMetrics
        Extended performance metrics for this split.
    """

    split_name: str
    metrics: ExtendedMetrics


@dataclass(frozen=True)
class WalkForwardResult:
    """Result of one walk-forward fold.

    Parameters
    ----------
    fold_index : int
        Fold number (0-indexed).
    in_sample : ValidationSplit
        In-sample results.
    out_of_sample : ValidationSplit
        Out-of-sample results.
    oos_efficiency : float
        OOS PnL / IS PnL. Closer to 1.0 = more robust.
    """

    fold_index: int
    in_sample: ValidationSplit
    out_of_sample: ValidationSplit
    oos_efficiency: float


@dataclass(frozen=True)
class ValidationReport:
    """Complete validation report.

    Parameters
    ----------
    in_sample_metrics : ExtendedMetrics
        Full IS metrics.
    out_of_sample_metrics : ExtendedMetrics
        Full OOS metrics.
    walk_forward_results : tuple[WalkForwardResult, ...]
        Per-fold walk-forward results.
    mean_oos_efficiency : float
        Average OOS efficiency across folds.
    """

    in_sample_metrics: ExtendedMetrics
    out_of_sample_metrics: ExtendedMetrics
    walk_forward_results: tuple[WalkForwardResult, ...]
    mean_oos_efficiency: float


@dataclass(frozen=True)
class RobustnessReport:
    """Robustness report from Monte Carlo analysis.

    Parameters
    ----------
    robustness : RobustnessMetrics
        Monte Carlo robustness metrics.
    n_simulations : int
        Number of Monte Carlo simulations run.
    """

    robustness: RobustnessMetrics
    n_simulations: int


@dataclass(frozen=True)
class ParameterSensitivityReport:
    """Parameter sensitivity report.

    Parameters
    ----------
    sensitivities : tuple[ParameterSensitivity, ...]
        Sensitivity analysis per parameter.
    most_sensitive_parameter : str
        Parameter with highest CV (least stable).
    most_stable_parameter : str
        Parameter with lowest CV (most stable).
    """

    sensitivities: tuple[ParameterSensitivity, ...]
    most_sensitive_parameter: str
    most_stable_parameter: str
