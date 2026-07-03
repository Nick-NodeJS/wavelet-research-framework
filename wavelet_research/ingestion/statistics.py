"""Statistics computation for tick data inspection.

Collects metrics at each stage of the ingestion pipeline without
duplicating any parsing, normalization, or validation logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import pandas as pd

from wavelet_research.ingestion.models import TickFrame
from wavelet_research.ingestion.normalization import (
    compute_derived_fields,
    drop_incomplete_rows,
    forward_fill_prices,
)
from wavelet_research.ingestion.parser import parse_ticks
from wavelet_research.ingestion.validation import validate


@dataclass(frozen=True)
class TickStatistics:
    """Dataset statistics collected during tick ingestion.

    Parameters
    ----------
    raw_rows : int
        Number of rows in the raw parsed file.
    normalized_rows : int
        Number of rows after normalization and validation.
    first_timestamp : pd.Timestamp
        Earliest timestamp in the dataset.
    last_timestamp : pd.Timestamp
        Latest timestamp in the dataset.
    bid_missing_before_ffill : int
        Count of NaN bid values before forward-fill.
    ask_missing_before_ffill : int
        Count of NaN ask values before forward-fill.
    bid_missing_after_ffill : int
        Count of NaN bid values after forward-fill (before drop).
    ask_missing_after_ffill : int
        Count of NaN ask values after forward-fill (before drop).
    invalid_spread_count : int
        Count of rows where ask < bid after normalization.
    duplicate_timestamp_count : int
        Count of timestamps that appear more than once.
    min_spread : float
        Minimum spread in the final dataset.
    avg_spread : float
        Mean spread in the final dataset.
    max_spread : float
        Maximum spread in the final dataset.
    min_bid : float
        Minimum bid in the final dataset.
    max_bid : float
        Maximum bid in the final dataset.
    min_ask : float
        Minimum ask in the final dataset.
    max_ask : float
        Maximum ask in the final dataset.
    time_span : pd.Timedelta
        Duration from first to last timestamp.
    memory_usage_mb : float
        Approximate memory usage of the final dataframe in MB.
    """

    raw_rows: int
    normalized_rows: int
    first_timestamp: pd.Timestamp
    last_timestamp: pd.Timestamp
    bid_missing_before_ffill: int
    ask_missing_before_ffill: int
    bid_missing_after_ffill: int
    ask_missing_after_ffill: int
    invalid_spread_count: int
    duplicate_timestamp_count: int
    min_spread: float
    avg_spread: float
    max_spread: float
    min_bid: float
    max_bid: float
    min_ask: float
    max_ask: float
    time_span: pd.Timedelta
    memory_usage_mb: float
    ticks: TickFrame


def inspect_ticks(path: Union[str, Path]) -> TickStatistics:
    """Load tick data and collect statistics at each pipeline stage.

    Reuses the existing ingestion components without duplicating logic.

    Parameters
    ----------
    path : str or Path
        Path to the tick data file.

    Returns
    -------
    TickStatistics
        Collected statistics and the final TickFrame.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the format is unrecognized.
    TickValidationError
        If no valid data remains after validation.
    """
    path = Path(path)

    # Stage 1: Parse
    raw = parse_ticks(path)
    raw_rows = len(raw)

    # Stage 2: Measure missing before forward-fill
    bid_missing_before_ffill = int(raw["bid"].isna().sum())
    ask_missing_before_ffill = int(raw["ask"].isna().sum())

    # Stage 3: Forward-fill
    filled = forward_fill_prices(raw)
    bid_missing_after_ffill = int(filled["bid"].isna().sum())
    ask_missing_after_ffill = int(filled["ask"].isna().sum())

    # Stage 4: Drop incomplete and compute derived fields
    dropped = drop_incomplete_rows(filled)
    normalized = compute_derived_fields(dropped)

    # Stage 5: Count invalid spreads before validation removes them
    invalid_spread_count = int((normalized["ask"] < normalized["bid"]).sum())

    # Stage 6: Validate
    validated = validate(normalized)
    normalized_rows = len(validated)

    # Stage 7: Compute final statistics
    first_timestamp = validated["time"].iloc[0]
    last_timestamp = validated["time"].iloc[-1]
    time_span = last_timestamp - first_timestamp

    duplicate_timestamp_count = int(validated["time"].duplicated().sum())

    min_spread = float(validated["spread"].min())
    avg_spread = float(validated["spread"].mean())
    max_spread = float(validated["spread"].max())

    min_bid = float(validated["bid"].min())
    max_bid = float(validated["bid"].max())
    min_ask = float(validated["ask"].min())
    max_ask = float(validated["ask"].max())

    memory_usage_mb = validated.memory_usage(deep=True).sum() / (1024 * 1024)

    tick_frame = TickFrame(frame=validated)

    return TickStatistics(
        raw_rows=raw_rows,
        normalized_rows=normalized_rows,
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
        bid_missing_before_ffill=bid_missing_before_ffill,
        ask_missing_before_ffill=ask_missing_before_ffill,
        bid_missing_after_ffill=bid_missing_after_ffill,
        ask_missing_after_ffill=ask_missing_after_ffill,
        invalid_spread_count=invalid_spread_count,
        duplicate_timestamp_count=duplicate_timestamp_count,
        min_spread=min_spread,
        avg_spread=avg_spread,
        max_spread=max_spread,
        min_bid=min_bid,
        max_bid=max_bid,
        min_ask=min_ask,
        max_ask=max_ask,
        time_span=time_span,
        memory_usage_mb=round(memory_usage_mb, 2),
        ticks=tick_frame,
    )


def format_statistics(stats: TickStatistics) -> str:
    """Format tick statistics as a human-readable report.

    Parameters
    ----------
    stats : TickStatistics
        Collected statistics to format.

    Returns
    -------
    str
        Formatted multi-line statistics report.
    """
    lines = [
        f"raw_rows                  {stats.raw_rows}",
        f"normalized_rows           {stats.normalized_rows}",
        "",
        f"first_timestamp           {stats.first_timestamp}",
        f"last_timestamp            {stats.last_timestamp}",
        "",
        f"bid_missing_before_ffill  {stats.bid_missing_before_ffill}",
        f"ask_missing_before_ffill  {stats.ask_missing_before_ffill}",
        "",
        f"bid_missing_after_ffill   {stats.bid_missing_after_ffill}",
        f"ask_missing_after_ffill   {stats.ask_missing_after_ffill}",
        "",
        f"invalid_spread_count      {stats.invalid_spread_count}",
        "",
        f"duplicate_timestamp_count {stats.duplicate_timestamp_count}",
        "",
        f"min_spread                {stats.min_spread:.10f}",
        f"avg_spread                {stats.avg_spread:.10f}",
        f"max_spread                {stats.max_spread:.10f}",
        "",
        f"min_bid                   {stats.min_bid:.5f}",
        f"max_bid                   {stats.max_bid:.5f}",
        "",
        f"min_ask                   {stats.min_ask:.5f}",
        f"max_ask                   {stats.max_ask:.5f}",
        "",
        f"time_span                 {stats.time_span}",
        "",
        f"memory_usage_mb           {stats.memory_usage_mb}",
    ]
    return "\n".join(lines)
