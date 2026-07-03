"""Data splitting for validation: IS/OOS, walk-forward, rolling windows.

All splits are index-based and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for data splitting.

    Parameters
    ----------
    in_sample_ratio : float
        Fraction of data for in-sample (0.0, 1.0).
    n_folds : int
        Number of walk-forward folds. Must be >= 1.

    Raises
    ------
    ValueError
        If parameters are invalid.
    """

    in_sample_ratio: float = 0.7
    n_folds: int = 3

    def __post_init__(self) -> None:
        if not (0.0 < self.in_sample_ratio < 1.0):
            raise ValueError(
                f"in_sample_ratio must be in (0.0, 1.0), got {self.in_sample_ratio}"
            )
        if self.n_folds < 1:
            raise ValueError(f"n_folds must be >= 1, got {self.n_folds}")


def split_in_out_of_sample(
    data: pd.DataFrame, ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into in-sample and out-of-sample sets.

    Parameters
    ----------
    data : pd.DataFrame
        Full dataset.
    ratio : float
        Fraction for in-sample.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (in_sample, out_of_sample) DataFrames.

    Raises
    ------
    ValueError
        If data is empty or ratio invalid.
    """
    if data.empty:
        raise ValueError("Data is empty.")
    if not (0.0 < ratio < 1.0):
        raise ValueError(f"ratio must be in (0.0, 1.0), got {ratio}")

    split_idx = int(len(data) * ratio)
    split_idx = max(1, min(split_idx, len(data) - 1))
    return data.iloc[:split_idx].reset_index(drop=True), data.iloc[split_idx:].reset_index(drop=True)


def generate_walk_forward_splits(
    data: pd.DataFrame, config: SplitConfig
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Generate walk-forward validation splits.

    Each fold uses an expanding in-sample window followed by
    a fixed out-of-sample window.

    Parameters
    ----------
    data : pd.DataFrame
        Full dataset.
    config : SplitConfig
        Split configuration.

    Returns
    -------
    list[tuple[pd.DataFrame, pd.DataFrame]]
        List of (in_sample, out_of_sample) pairs.

    Raises
    ------
    ValueError
        If data is too small for the requested folds.
    """
    if data.empty:
        raise ValueError("Data is empty.")

    n = len(data)
    oos_size = int(n * (1 - config.in_sample_ratio) / config.n_folds)
    oos_size = max(1, oos_size)

    splits: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for fold in range(config.n_folds):
        oos_end = n - (config.n_folds - fold - 1) * oos_size
        oos_start = oos_end - oos_size
        is_end = oos_start

        if is_end < 1 or oos_end > n:
            break

        is_data = data.iloc[:is_end].reset_index(drop=True)
        oos_data = data.iloc[oos_start:oos_end].reset_index(drop=True)
        splits.append((is_data, oos_data))

    if not splits:
        raise ValueError(
            f"Data too small ({n} rows) for {config.n_folds} folds."
        )

    return splits


def generate_rolling_splits(
    data: pd.DataFrame, window_size: int, step_size: int
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Generate rolling window splits.

    Each split uses a fixed-size training window followed by
    a step-sized test window.

    Parameters
    ----------
    data : pd.DataFrame
        Full dataset.
    window_size : int
        Size of the rolling training window.
    step_size : int
        Size of the test window (step between rolls).

    Returns
    -------
    list[tuple[pd.DataFrame, pd.DataFrame]]
        List of (train, test) pairs.

    Raises
    ------
    ValueError
        If parameters are invalid.
    """
    if data.empty:
        raise ValueError("Data is empty.")
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")
    if step_size < 1:
        raise ValueError(f"step_size must be >= 1, got {step_size}")

    n = len(data)
    splits: list[tuple[pd.DataFrame, pd.DataFrame]] = []

    start = 0
    while start + window_size + step_size <= n:
        train = data.iloc[start:start + window_size].reset_index(drop=True)
        test = data.iloc[start + window_size:start + window_size + step_size].reset_index(drop=True)
        splits.append((train, test))
        start += step_size

    if not splits:
        raise ValueError(
            f"Data too small ({n} rows) for window={window_size}, step={step_size}."
        )

    return splits
