"""Result persistence for the research framework.

Supports CSV and Parquet formats for persisting experiment results.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Sequence

import pandas as pd

from wavelet_research.research.config import ExperimentConfig
from wavelet_research.research.experiment import ExperimentResult
from wavelet_research.research.metrics import TradeMetrics

logger = logging.getLogger(__name__)


class StorageFormat(Enum):
    """Supported storage formats."""

    CSV = "csv"
    PARQUET = "parquet"


def _result_to_row(result: ExperimentResult) -> dict:
    """Convert an ExperimentResult to a flat dictionary for storage.

    Parameters
    ----------
    result : ExperimentResult
        Experiment result to convert.

    Returns
    -------
    dict
        Flat dictionary with config and metrics fields.
    """
    row = asdict(result.config)
    row.update(asdict(result.metrics))
    return row


class ResultStorage:
    """Persists experiment results to disk.

    Parameters
    ----------
    output_path : Path or str
        Directory to store result files.
    storage_format : StorageFormat
        Output format (CSV or Parquet).
    """

    def __init__(
        self,
        output_path: Path | str,
        storage_format: StorageFormat = StorageFormat.CSV,
    ) -> None:
        self._output_path = Path(output_path)
        self._format = storage_format

    @property
    def output_path(self) -> Path:
        """Storage output directory."""
        return self._output_path

    @property
    def storage_format(self) -> StorageFormat:
        """Configured storage format."""
        return self._format

    def save(
        self, results: Sequence[ExperimentResult], filename: str = "results"
    ) -> Path:
        """Save experiment results to disk.

        Parameters
        ----------
        results : Sequence[ExperimentResult]
            Results to persist.
        filename : str
            Base filename (without extension).

        Returns
        -------
        Path
            Path to the saved file.

        Raises
        ------
        ValueError
            If no results are provided.
        """
        if not results:
            raise ValueError("No results to save.")

        self._output_path.mkdir(parents=True, exist_ok=True)

        rows = [_result_to_row(r) for r in results]
        df = pd.DataFrame(rows)

        if self._format == StorageFormat.CSV:
            filepath = self._output_path / f"{filename}.csv"
            df.to_csv(filepath, index=False)
        elif self._format == StorageFormat.PARQUET:
            filepath = self._output_path / f"{filename}.parquet"
            df.to_parquet(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {self._format}")

        logger.info("Saved %d results to %s", len(results), filepath)
        return filepath

    def load(self, filename: str = "results") -> pd.DataFrame:
        """Load previously saved results.

        Parameters
        ----------
        filename : str
            Base filename (without extension).

        Returns
        -------
        pd.DataFrame
            Loaded results as a dataframe.

        Raises
        ------
        FileNotFoundError
            If the results file does not exist.
        """
        if self._format == StorageFormat.CSV:
            filepath = self._output_path / f"{filename}.csv"
        else:
            filepath = self._output_path / f"{filename}.parquet"

        if not filepath.exists():
            raise FileNotFoundError(f"Results file not found: {filepath}")

        if self._format == StorageFormat.CSV:
            return pd.read_csv(filepath)
        else:
            return pd.read_parquet(filepath)
