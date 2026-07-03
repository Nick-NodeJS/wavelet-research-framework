"""Parameter stability / sensitivity analysis.

Evaluates how sensitive strategy performance is to parameter changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wavelet_research.orchestrator.results import ExperimentReport


@dataclass(frozen=True)
class ParameterSensitivity:
    """Sensitivity of a single parameter.

    Parameters
    ----------
    parameter_name : str
        Name of the parameter.
    values : tuple[object, ...]
        Distinct values tested.
    pnl_by_value : tuple[float, ...]
        Mean PnL for each value.
    profit_factor_by_value : tuple[float, ...]
        Mean profit factor for each value.
    coefficient_of_variation : float
        CV of PnL across values. Lower = more stable.
    """

    parameter_name: str
    values: tuple[object, ...]
    pnl_by_value: tuple[float, ...]
    profit_factor_by_value: tuple[float, ...]
    coefficient_of_variation: float


def _extract_parameter(
    report: ExperimentReport, param_name: str
) -> object:
    """Extract a named parameter from an ExperimentReport.

    Parameters
    ----------
    report : ExperimentReport
        The experiment report.
    param_name : str
        Parameter name (prefixed with config source).

    Returns
    -------
    object
        Parameter value.
    """
    wc = report.config.wavelet_config
    sc = report.config.signal_config
    bc = report.config.backtest_config

    param_map = {
        "wavelet": wc.wavelet,
        "window": wc.window,
        "level": wc.level,
        "volatility_window": wc.volatility_window,
        "buy_z_threshold": sc.buy_z_threshold,
        "sell_z_threshold": sc.sell_z_threshold,
        "slope_filter": sc.slope_filter_enabled,
        "exit_strategy": bc.exit_strategy.value,
        "max_hold_ticks": bc.max_hold_ticks,
    }

    if param_name not in param_map:
        raise ValueError(f"Unknown parameter: {param_name}")
    return param_map[param_name]


def analyze_parameter_sensitivity(
    reports: list[ExperimentReport],
    parameter_name: str,
) -> ParameterSensitivity:
    """Analyze sensitivity of results to a single parameter.

    Groups reports by parameter value and computes mean PnL
    and profit factor for each group.

    Parameters
    ----------
    reports : list[ExperimentReport]
        All experiment reports.
    parameter_name : str
        Parameter to analyze.

    Returns
    -------
    ParameterSensitivity
        Sensitivity analysis for the parameter.

    Raises
    ------
    ValueError
        If reports is empty or parameter unknown.
    """
    if not reports:
        raise ValueError("Reports must not be empty.")

    groups: dict[object, list[ExperimentReport]] = {}
    for r in reports:
        value = _extract_parameter(r, parameter_name)
        groups.setdefault(value, []).append(r)

    values: list[object] = []
    pnl_means: list[float] = []
    pf_means: list[float] = []

    for value in sorted(groups.keys(), key=str):
        group = groups[value]
        mean_pnl = float(np.mean([r.report.total_pnl for r in group]))
        pf_vals = [r.report.profit_factor for r in group]
        finite_pf = [v for v in pf_vals if np.isfinite(v)]
        mean_pf = float(np.mean(finite_pf)) if finite_pf else 0.0

        values.append(value)
        pnl_means.append(mean_pnl)
        pf_means.append(mean_pf)

    pnl_arr = np.asarray(pnl_means, dtype=np.float64)
    mean = float(np.mean(pnl_arr))
    std = float(np.std(pnl_arr))
    cv = std / abs(mean) if mean != 0 else float("inf")

    return ParameterSensitivity(
        parameter_name=parameter_name,
        values=tuple(values),
        pnl_by_value=tuple(pnl_means),
        profit_factor_by_value=tuple(pf_means),
        coefficient_of_variation=cv,
    )
