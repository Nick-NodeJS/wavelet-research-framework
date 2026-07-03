"""Parsing logic for tick data files.

Supports two formats:
- Format A: Normalized CSV with columns time,bid,ask
- Format B: Raw MT5 tick export (tab-separated with <DATE>, <TIME>, <BID>, <ASK>, <LAST>, <VOLUME>, <FLAGS>)

Format detection is automatic based on header inspection.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import pandas as pd


class TickFormat(Enum):
    """Detected tick file format."""

    NORMALIZED_CSV = "normalized_csv"
    MT5_EXPORT = "mt5_export"


def detect_format(path: Path) -> TickFormat:
    """Detect the format of a tick data file by inspecting the header line.

    Parameters
    ----------
    path : Path
        Path to the tick data file.

    Returns
    -------
    TickFormat
        Detected file format.

    Raises
    ------
    ValueError
        If the file is empty or the format is unrecognized.
    """
    with open(path, "r", encoding="utf-8") as fh:
        header = fh.readline().strip()

    if not header:
        raise ValueError(f"Empty file: {path}")

    if "<DATE>" in header and "<TIME>" in header:
        return TickFormat.MT5_EXPORT

    lower_header = header.lower()
    if "time" in lower_header and "bid" in lower_header and "ask" in lower_header:
        return TickFormat.NORMALIZED_CSV

    raise ValueError(f"Unrecognized tick file format. Header: {header!r}")


def parse_mt5_export(path: Path) -> pd.DataFrame:
    """Parse a raw MT5 tick export file.

    Parameters
    ----------
    path : Path
        Path to the MT5 export file.

    Returns
    -------
    pd.DataFrame
        Raw parsed dataframe with columns: time, bid, ask, flags.
        Bid/Ask may contain NaN for partial updates.
    """
    df = pd.read_csv(
        path,
        sep="\t",
        header=0,
        names=["date", "time_col", "bid", "ask", "last", "volume", "flags"],
        dtype={"date": str, "time_col": str, "flags": "Int64"},
        na_values=[""],
        low_memory=True,
    )

    # Combine date and time into a single timestamp
    datetime_str = df["date"].str.strip() + " " + df["time_col"].str.strip()
    df["time"] = pd.to_datetime(datetime_str, format="%Y.%m.%d %H:%M:%S.%f", errors="coerce")

    # Convert bid/ask to numeric
    df["bid"] = pd.to_numeric(df["bid"], errors="coerce")
    df["ask"] = pd.to_numeric(df["ask"], errors="coerce")

    # Keep only relevant columns
    result = df[["time", "bid", "ask", "flags"]].copy()
    return result


def parse_normalized_csv(path: Path) -> pd.DataFrame:
    """Parse a normalized CSV file with time,bid,ask columns.

    Parameters
    ----------
    path : Path
        Path to the normalized CSV file.

    Returns
    -------
    pd.DataFrame
        Raw parsed dataframe with columns: time, bid, ask, flags.
    """
    df = pd.read_csv(path)

    required = {"time", "bid", "ask"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in normalized CSV: {sorted(missing)}")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["bid"] = pd.to_numeric(df["bid"], errors="coerce")
    df["ask"] = pd.to_numeric(df["ask"], errors="coerce")

    if "flags" not in df.columns:
        df["flags"] = pd.array([pd.NA] * len(df), dtype="Int64")
    else:
        df["flags"] = df["flags"].astype("Int64")

    result = df[["time", "bid", "ask", "flags"]].copy()
    return result


def parse_ticks(path: Path) -> pd.DataFrame:
    """Auto-detect format and parse a tick data file.

    Parameters
    ----------
    path : Path
        Path to the tick data file.

    Returns
    -------
    pd.DataFrame
        Raw parsed dataframe with columns: time, bid, ask, flags.

    Raises
    ------
    ValueError
        If the file format is unrecognized or required columns are missing.
    FileNotFoundError
        If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Tick data file not found: {path}")

    fmt = detect_format(path)

    if fmt == TickFormat.MT5_EXPORT:
        return parse_mt5_export(path)
    else:
        return parse_normalized_csv(path)
