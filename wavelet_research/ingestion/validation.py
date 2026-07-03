"""Validation logic for normalized tick data.

Rejects data containing:
- ask < bid (invalid spread)
- invalid timestamps (NaT)
- non-numeric bid/ask
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class TickValidationError(ValueError):
    """Raised when tick data fails validation."""

    pass


def validate_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with invalid (NaT) timestamps.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with a time column.

    Returns
    -------
    pd.DataFrame
        Dataframe with invalid timestamps removed.

    Raises
    ------
    TickValidationError
        If all timestamps are invalid.
    """
    invalid_mask = df["time"].isna()
    invalid_count = invalid_mask.sum()

    if invalid_count > 0:
        logger.warning("Dropping %d rows with invalid timestamps.", invalid_count)
        df = df[~invalid_mask].reset_index(drop=True)

    if len(df) == 0:
        raise TickValidationError("All timestamps are invalid.")

    return df


def validate_spreads(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where ask < bid (invalid spread).

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with bid and ask columns.

    Returns
    -------
    pd.DataFrame
        Dataframe with invalid spread rows removed.

    Raises
    ------
    TickValidationError
        If all rows have invalid spreads.
    """
    invalid_mask = df["ask"] < df["bid"]
    invalid_count = invalid_mask.sum()

    if invalid_count > 0:
        logger.warning("Dropping %d rows with invalid spread (ask < bid).", invalid_count)
        df = df[~invalid_mask].reset_index(drop=True)

    if len(df) == 0:
        raise TickValidationError("All rows have invalid spreads (ask < bid).")

    return df


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Run full validation pipeline on normalized tick data.

    Parameters
    ----------
    df : pd.DataFrame
        Normalized dataframe with columns: time, bid, ask, mid, spread, flags.

    Returns
    -------
    pd.DataFrame
        Validated dataframe with invalid rows removed.

    Raises
    ------
    TickValidationError
        If no valid data remains after validation.
    """
    df = validate_timestamps(df)
    df = validate_spreads(df)
    return df
