"""Normalization logic for parsed tick data.

Handles forward-filling of bid/ask and computation of derived fields.
"""

from __future__ import annotations

import pandas as pd


def forward_fill_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill bid and ask prices independently.

    MT5 exports frequently contain partial updates where only bid or ask changes.
    This function forward-fills each independently to preserve all market updates.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with bid and ask columns (may contain NaN).

    Returns
    -------
    pd.DataFrame
        Dataframe with bid and ask forward-filled.
    """
    result = df.copy()
    result["bid"] = result["bid"].ffill()
    result["ask"] = result["ask"].ffill()
    return result


def drop_incomplete_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where bid or ask is still NaN after forward-filling.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe after forward-fill.

    Returns
    -------
    pd.DataFrame
        Dataframe with no NaN in bid or ask.
    """
    return df.dropna(subset=["bid", "ask"]).reset_index(drop=True)


def compute_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mid and spread from bid and ask.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with valid bid and ask columns.

    Returns
    -------
    pd.DataFrame
        Dataframe with mid and spread columns added.
    """
    result = df.copy()
    result["mid"] = (result["bid"] + result["ask"]) / 2.0
    result["spread"] = result["ask"] - result["bid"]
    return result


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Apply full normalization pipeline to parsed tick data.

    Steps:
    1. Forward-fill bid and ask independently.
    2. Drop rows still missing bid or ask.
    3. Compute derived fields (mid, spread).

    Parameters
    ----------
    df : pd.DataFrame
        Raw parsed dataframe with columns: time, bid, ask, flags.

    Returns
    -------
    pd.DataFrame
        Normalized dataframe with columns: time, bid, ask, mid, spread, flags.
    """
    df = forward_fill_prices(df)
    df = drop_incomplete_rows(df)
    df = compute_derived_fields(df)
    return df[["time", "bid", "ask", "mid", "spread", "flags"]]
