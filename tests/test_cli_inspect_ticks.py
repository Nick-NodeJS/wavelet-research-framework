"""Tests for the CLI tick inspection tool (Story 1a)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from wavelet_research.cli_inspect_ticks import main
from wavelet_research.ingestion.statistics import (
    TickStatistics,
    format_statistics,
    inspect_ticks,
)
from wavelet_research.ingestion.validation import TickValidationError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_mt5_file(tmp_path: Path) -> Path:
    """Create a valid MT5 export file with enough rows."""
    content = (
        "<DATE>\t<TIME>\t<BID>\t<ASK>\t<LAST>\t<VOLUME>\t<FLAGS>\n"
        "2026.06.01\t00:00:00.100\t1.10000\t1.10010\t\t\t6\n"
        "2026.06.01\t00:00:00.200\t1.10001\t1.10011\t\t\t6\n"
        "2026.06.01\t00:00:00.300\t\t1.10012\t\t\t4\n"
        "2026.06.01\t00:00:00.400\t1.10003\t1.10013\t\t\t6\n"
        "2026.06.01\t00:00:00.500\t1.10004\t1.10014\t\t\t6\n"
    )
    p = tmp_path / "valid.csv"
    p.write_text(content)
    return p


@pytest.fixture
def valid_csv_file(tmp_path: Path) -> Path:
    """Create a valid normalized CSV file."""
    content = (
        "time,bid,ask\n"
        "2026-06-01 00:00:00.100,1.10000,1.10010\n"
        "2026-06-01 00:00:00.200,1.10001,1.10011\n"
        "2026-06-01 00:00:00.300,1.10002,1.10012\n"
    )
    p = tmp_path / "valid.csv"
    p.write_text(content)
    return p


@pytest.fixture
def invalid_spread_file(tmp_path: Path) -> Path:
    """Create a file where ALL rows have ask < bid."""
    content = (
        "time,bid,ask\n"
        "2026-06-01 00:00:00.100,1.10010,1.10000\n"
        "2026-06-01 00:00:00.200,1.10011,1.10001\n"
    )
    p = tmp_path / "bad_spread.csv"
    p.write_text(content)
    return p


@pytest.fixture
def partial_invalid_spread_file(tmp_path: Path) -> Path:
    """Create a file where SOME rows have ask < bid."""
    content = (
        "time,bid,ask\n"
        "2026-06-01 00:00:00.100,1.10000,1.10010\n"
        "2026-06-01 00:00:00.200,1.10020,1.10005\n"
        "2026-06-01 00:00:00.300,1.10002,1.10012\n"
    )
    p = tmp_path / "partial_bad_spread.csv"
    p.write_text(content)
    return p


@pytest.fixture
def invalid_timestamp_file(tmp_path: Path) -> Path:
    """Create a file where all timestamps are invalid."""
    content = (
        "time,bid,ask\n"
        "not-a-date,1.10000,1.10010\n"
        "also-bad,1.10001,1.10011\n"
    )
    p = tmp_path / "bad_timestamp.csv"
    p.write_text(content)
    return p


@pytest.fixture
def empty_data_file(tmp_path: Path) -> Path:
    """Create a file with header only (no data rows)."""
    content = "time,bid,ask\n"
    p = tmp_path / "empty.csv"
    p.write_text(content)
    return p


@pytest.fixture
def duplicate_timestamp_file(tmp_path: Path) -> Path:
    """Create a file with duplicate timestamps."""
    content = (
        "time,bid,ask\n"
        "2026-06-01 00:00:00.100,1.10000,1.10010\n"
        "2026-06-01 00:00:00.100,1.10001,1.10011\n"
        "2026-06-01 00:00:00.200,1.10002,1.10012\n"
        "2026-06-01 00:00:00.200,1.10003,1.10013\n"
        "2026-06-01 00:00:00.300,1.10004,1.10014\n"
    )
    p = tmp_path / "duplicates.csv"
    p.write_text(content)
    return p


@pytest.fixture
def real_data_path() -> Path:
    """Path to the real EURUSD data file."""
    return Path(
        "/home/nick/PrivateProjects/TradeAutomation/wavelet-research-v0"
        "/data/EURUSD_202606010002_202606292358.csv"
    )


# ---------------------------------------------------------------------------
# inspect_ticks Tests
# ---------------------------------------------------------------------------


class TestInspectTicks:
    def test_valid_dataset(self, valid_mt5_file: Path) -> None:
        stats = inspect_ticks(valid_mt5_file)
        assert stats.raw_rows == 5
        assert stats.normalized_rows == 5
        assert stats.bid_missing_before_ffill == 1
        assert stats.ask_missing_before_ffill == 0
        assert stats.bid_missing_after_ffill == 0
        assert stats.ask_missing_after_ffill == 0
        assert stats.invalid_spread_count == 0
        assert stats.min_spread >= 0
        assert stats.avg_spread >= 0
        assert stats.max_spread >= 0
        assert stats.min_bid > 0
        assert stats.max_bid > 0
        assert stats.min_ask > 0
        assert stats.max_ask > 0
        assert isinstance(stats.time_span, pd.Timedelta)
        assert stats.memory_usage_mb >= 0
        assert isinstance(stats.ticks, object)
        assert len(stats.ticks) == 5

    def test_valid_csv_dataset(self, valid_csv_file: Path) -> None:
        stats = inspect_ticks(valid_csv_file)
        assert stats.raw_rows == 3
        assert stats.normalized_rows == 3
        assert stats.bid_missing_before_ffill == 0
        assert stats.ask_missing_before_ffill == 0

    def test_invalid_spread_all_rows(self, invalid_spread_file: Path) -> None:
        with pytest.raises(TickValidationError, match="invalid spreads"):
            inspect_ticks(invalid_spread_file)

    def test_partial_invalid_spread(self, partial_invalid_spread_file: Path) -> None:
        stats = inspect_ticks(partial_invalid_spread_file)
        assert stats.invalid_spread_count == 1
        assert stats.normalized_rows == 2

    def test_invalid_timestamp_all_rows(self, invalid_timestamp_file: Path) -> None:
        with pytest.raises(TickValidationError, match="All timestamps are invalid"):
            inspect_ticks(invalid_timestamp_file)

    def test_empty_dataset(self, empty_data_file: Path) -> None:
        with pytest.raises(TickValidationError, match="All timestamps are invalid"):
            inspect_ticks(empty_data_file)

    def test_duplicate_timestamps(self, duplicate_timestamp_file: Path) -> None:
        stats = inspect_ticks(duplicate_timestamp_file)
        assert stats.duplicate_timestamp_count == 2
        assert stats.normalized_rows == 5

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            inspect_ticks("/nonexistent/path.csv")


# ---------------------------------------------------------------------------
# format_statistics Tests
# ---------------------------------------------------------------------------


class TestFormatStatistics:
    def test_format_contains_all_fields(self, valid_mt5_file: Path) -> None:
        stats = inspect_ticks(valid_mt5_file)
        output = format_statistics(stats)
        expected_labels = [
            "raw_rows",
            "normalized_rows",
            "first_timestamp",
            "last_timestamp",
            "bid_missing_before_ffill",
            "ask_missing_before_ffill",
            "bid_missing_after_ffill",
            "ask_missing_after_ffill",
            "invalid_spread_count",
            "duplicate_timestamp_count",
            "min_spread",
            "avg_spread",
            "max_spread",
            "min_bid",
            "max_bid",
            "min_ask",
            "max_ask",
            "time_span",
            "memory_usage_mb",
        ]
        for label in expected_labels:
            assert label in output, f"Missing label: {label}"


# ---------------------------------------------------------------------------
# CLI Tests
# ---------------------------------------------------------------------------


class TestCLI:
    def test_successful_execution(self, valid_mt5_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
        exit_code = main(["--input", str(valid_mt5_file)])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "raw_rows" in captured.out
        assert "normalized_rows" in captured.out
        assert "min_spread" in captured.out

    def test_validation_failure_exit_code(
        self, invalid_spread_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        exit_code = main(["--input", str(invalid_spread_file)])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "ERROR" in captured.err

    def test_file_not_found_exit_code(self, capsys: pytest.CaptureFixture[str]) -> None:
        exit_code = main(["--input", "/nonexistent/file.csv"])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "ERROR" in captured.err

    def test_invalid_timestamp_exit_code(
        self, invalid_timestamp_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        exit_code = main(["--input", str(invalid_timestamp_file)])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "ERROR" in captured.err

    def test_empty_dataset_exit_code(
        self, empty_data_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        exit_code = main(["--input", str(empty_data_file)])
        assert exit_code == 1

    def test_real_data_execution(self, real_data_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Smoke test with real EURUSD data."""
        if not real_data_path.exists():
            pytest.skip("Real data file not available.")
        exit_code = main(["--input", str(real_data_path)])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "raw_rows" in captured.out
        assert "memory_usage_mb" in captured.out
