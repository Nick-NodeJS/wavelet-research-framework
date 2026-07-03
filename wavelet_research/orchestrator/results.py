"""Result models and aggregation for the orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Sequence

import pandas as pd

from wavelet_research.backtest.models import BacktestReport
from wavelet_research.orchestrator.config import PipelineConfig


@dataclass(frozen=True)
class ExperimentReport:
    """Result of a single orchestrated experiment.

    Parameters
    ----------
    config : PipelineConfig
        Configuration used for this experiment.
    report : BacktestReport
        Backtest report with metrics, journal, and equity curve.
    """

    config: PipelineConfig
    report: BacktestReport


def report_to_row(experiment: ExperimentReport) -> dict:
    """Convert an ExperimentReport to a flat dictionary for storage.

    Parameters
    ----------
    experiment : ExperimentReport
        Experiment report to convert.

    Returns
    -------
    dict
        Flat dictionary suitable for DataFrame construction.
    """
    wc = experiment.config.wavelet_config
    sc = experiment.config.signal_config
    bc = experiment.config.backtest_config
    r = experiment.report

    return {
        "wavelet": wc.wavelet,
        "window": wc.window,
        "level": wc.level,
        "volatility_window": wc.volatility_window,
        "buy_z_threshold": sc.buy_z_threshold,
        "sell_z_threshold": sc.sell_z_threshold,
        "slope_filter": sc.slope_filter_enabled,
        "exit_strategy": bc.exit_strategy.value,
        "max_hold_ticks": bc.max_hold_ticks,
        "commission_pips": bc.commission_pips,
        "slippage_pips": bc.slippage_pips,
        "trades": r.trades,
        "win_rate": r.win_rate,
        "profit_factor": r.profit_factor,
        "expectancy": r.expectancy,
        "total_pnl": r.total_pnl,
        "max_drawdown": r.max_drawdown,
        "average_mae": r.average_mae,
        "average_mfe": r.average_mfe,
        "average_trade": r.average_trade,
        "average_holding_time": r.average_holding_time,
    }


def rank_experiments(
    experiments: Sequence[ExperimentReport],
) -> list[ExperimentReport]:
    """Rank experiment reports by profit factor, expectancy, drawdown, PnL.

    Parameters
    ----------
    experiments : Sequence[ExperimentReport]
        Experiments to rank.

    Returns
    -------
    list[ExperimentReport]
        Sorted experiments (best first).
    """
    if not experiments:
        return []

    def sort_key(exp: ExperimentReport) -> tuple[float, ...]:
        r = exp.report
        return (
            -r.profit_factor,
            -r.expectancy,
            r.max_drawdown,
            -r.total_pnl,
        )

    return sorted(experiments, key=sort_key)


def save_results(
    experiments: Sequence[ExperimentReport],
    output_path: Path,
    filename: str = "results",
) -> Path:
    """Save experiment results to CSV.

    Parameters
    ----------
    experiments : Sequence[ExperimentReport]
        Experiments to persist.
    output_path : Path
        Output directory.
    filename : str
        Base filename without extension.

    Returns
    -------
    Path
        Path to saved file.

    Raises
    ------
    ValueError
        If experiments is empty.
    """
    if not experiments:
        raise ValueError("No experiments to save.")

    output_path.mkdir(parents=True, exist_ok=True)
    rows = [report_to_row(e) for e in experiments]
    df = pd.DataFrame(rows)
    filepath = output_path / f"{filename}.csv"
    df.to_csv(filepath, index=False)
    return filepath


def load_results(output_path: Path, filename: str = "results") -> pd.DataFrame:
    """Load previously saved results.

    Parameters
    ----------
    output_path : Path
        Output directory.
    filename : str
        Base filename without extension.

    Returns
    -------
    pd.DataFrame
        Loaded results.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    filepath = output_path / f"{filename}.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"Results file not found: {filepath}")
    return pd.read_csv(filepath)
