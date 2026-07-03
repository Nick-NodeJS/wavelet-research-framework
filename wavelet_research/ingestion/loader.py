"""Top-level loader orchestrating parsing, normalization, and validation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import pandas as pd

from wavelet_research.ingestion.models import TickFrame
from wavelet_research.ingestion.normalization import normalize
from wavelet_research.ingestion.parser import parse_ticks
from wavelet_research.ingestion.validation import validate

logger = logging.getLogger(__name__)


def load_ticks(path: Union[str, Path]) -> TickFrame:
    """Load tick data from a file, auto-detecting the format.

    Pipeline:
    1. Parse the file (auto-detect MT5 export vs normalized CSV).
    2. Normalize (forward-fill, compute derived fields).
    3. Validate (remove invalid timestamps, invalid spreads).

    Parameters
    ----------
    path : str or Path
        Path to the tick data file.

    Returns
    -------
    TickFrame
        Normalized, validated tick data container.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the format is unrecognized or data is insufficient.
    TickValidationError
        If no valid data remains after validation.
    """
    path = Path(path)
    logger.info("Loading ticks from %s", path)

    # Parse
    raw = parse_ticks(path)
    logger.info("Parsed %d raw rows.", len(raw))

    # Normalize
    normalized = normalize(raw)
    logger.info("Normalized to %d rows.", len(normalized))

    # Validate
    validated = validate(normalized)
    logger.info("Validated: %d rows remain.", len(validated))

    return TickFrame(frame=validated)
