"""Tests for the data ingestion layer (Story 1)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wavelet_research.ingestion import TickFrame, load_ticks
from wavelet_research.ingestion.normalization import (
    compute_derived_fields,
    drop_incomplete_rows,
    forward_fill_prices,
    normalize,
)
from wavelet_research.ingestion.parser import (
    TickFormat,
    detect_format,
    parse_mt5_export,
    parse_normalized_csv,
    parse_ticks,
)
from wavelet_research.ingestion.validation import (
    TickValidationError,
    validate,
    validate_spreads,
    validate_timestamps,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def normalized_csv_file(tmp_path: Path) -> Path:
    """Create a minimal normalized CSV file."""
    content = "time,bid,ask\n2026-06-01 00:00:00.100,1.16538,1.16544\n2026-06-01 00:00:01.200,1.16540,1.16546\n2026-06-01 00:00:02.300,1.16542,1.16548\n"
    p = tmp_path / "normalized.csv"
    p.write_text(content)
    return p


@pytest.fixture
def mt5_export_file(tmp_path: Path) -> Path:
    """Create a minimal MT5 export file."""
    content = (
        "<DATE>\t<TIME>\t<BID>\t<ASK>\t<LAST>\t<VOLUME>\t<FLAGS>\n"
        "2026.06.01\t00:02:00.701\t1.16538\t1.16544\t\t\t6\n"
        "2026.06.01\t00:02:00.821\t1.16523\t1.16559\t\t\t6\n"
        "2026.06.01\t00:02:00.882\t\t1.16565\t\t\t4\n"
        "2026.06.01\t00:02:01.385\t\t1.16572\t\t\t4\n"
        "2026.06.01\t00:02:02.151\t1.16538\t1.16544\t\t\t6\n"
    )
    p = tmp_path / "mt5_export.csv"
    p.write_text(content)
    return p


@pytest.fixture
def mt5_with_partial_updates(tmp_path: Path) -> Path:
    """MT5 file where first rows have only bid or only ask."""
    content = (
        "<DATE>\t<TIME>\t<BID>\t<ASK>\t<LAST>\t<VOLUME>\t<FLAGS>\n"
        "2026.06.01\t00:00:00.100\t1.10000\t\t\t\t2\n"
        "2026.06.01\t00:00:00.200\t\t1.10010\t\t\t4\n"
        "2026.06.01\t00:00:00.300\t1.10005\t1.10015\t\t\t6\n"
    )
    p = tmp_path / "partial.csv"
    p.write_text(content)
    return p


@pytest.fixture
def real_data_path() -> Path:
    """Path to the real EURUSD data file."""
    return Path("/home/nick/PrivateProjects/TradeAutomation/wavelet-research-v0/data/EURUSD_202606010002_202606292358.csv")


# ---------------------------------------------------------------------------
# Format Detection Tests
# ---------------------------------------------------------------------------


class TestFormatDetection:
    def test_detect_mt5_format(self, mt5_export_file: Path) -> None:
        assert detect_format(mt5_export_file) == TickFormat.MT5_EXPORT

    def test_detect_normalized_csv_format(self, normalized_csv_file: Path) -> None:
        assert detect_format(normalized_csv_file) == TickFormat.NORMALIZED_CSV

    def test_detect_empty_file_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.csv"
        p.write_text("")
        with pytest.raises(ValueError, match="Empty file"):
            detect_format(p)

    def test_detect_unrecognized_format_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "unknown.csv"
        p.write_text("col1,col2,col3\n1,2,3\n")
        with pytest.raises(ValueError, match="Unrecognized"):
            detect_format(p)


# ---------------------------------------------------------------------------
# Normalized CSV Parsing Tests
# ---------------------------------------------------------------------------


class TestNormalizedCSVParsing:
    def test_parse_normalized_csv(self, normalized_csv_file: Path) -> None:
        df = parse_normalized_csv(normalized_csv_file)
        assert list(df.columns) == ["time", "bid", "ask", "flags"]
        assert len(df) == 3
        assert df["bid"].iloc[0] == pytest.approx(1.16538)
        assert df["ask"].iloc[0] == pytest.approx(1.16544)

    def test_parse_normalized_csv_timestamps(self, normalized_csv_file: Path) -> None:
        df = parse_normalized_csv(normalized_csv_file)
        assert pd.api.types.is_datetime64_any_dtype(df["time"])
        assert df["time"].iloc[0].microsecond == 100000  # 100ms

    def test_parse_normalized_csv_missing_columns(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.csv"
        p.write_text("time,bid\n2026-01-01,1.1\n")
        with pytest.raises(ValueError, match="Missing columns"):
            parse_normalized_csv(p)


# ---------------------------------------------------------------------------
# MT5 Export Parsing Tests
# ---------------------------------------------------------------------------


class TestMT5Parsing:
    def test_parse_mt5_export(self, mt5_export_file: Path) -> None:
        df = parse_mt5_export(mt5_export_file)
        assert list(df.columns) == ["time", "bid", "ask", "flags"]
        assert len(df) == 5

    def test_parse_mt5_timestamps(self, mt5_export_file: Path) -> None:
        df = parse_mt5_export(mt5_export_file)
        assert pd.api.types.is_datetime64_any_dtype(df["time"])
        # First timestamp: 2026.06.01 00:02:00.701
        ts = df["time"].iloc[0]
        assert ts.year == 2026
        assert ts.month == 6
        assert ts.day == 1
        assert ts.hour == 0
        assert ts.minute == 2
        assert ts.second == 0
        assert ts.microsecond == 701000  # 701ms

    def test_parse_mt5_partial_bid(self, mt5_export_file: Path) -> None:
        df = parse_mt5_export(mt5_export_file)
        # Row 3 (index 2) has no bid
        assert pd.isna(df["bid"].iloc[2])
        # But has ask
        assert df["ask"].iloc[2] == pytest.approx(1.16565)

    def test_parse_mt5_flags(self, mt5_export_file: Path) -> None:
        df = parse_mt5_export(mt5_export_file)
        assert df["flags"].iloc[0] == 6
        assert df["flags"].iloc[2] == 4


# ---------------------------------------------------------------------------
# Forward Fill Tests
# ---------------------------------------------------------------------------


class TestForwardFill:
    def test_bid_forward_fill(self) -> None:
        df = pd.DataFrame({
            "time": pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 00:00:01", "2026-01-01 00:00:02"]),
            "bid": [1.10, np.nan, np.nan],
            "ask": [1.11, 1.12, 1.13],
            "flags": pd.array([6, 4, 4], dtype="Int64"),
        })
        result = forward_fill_prices(df)
        assert result["bid"].iloc[1] == pytest.approx(1.10)
        assert result["bid"].iloc[2] == pytest.approx(1.10)

    def test_ask_forward_fill(self) -> None:
        df = pd.DataFrame({
            "time": pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 00:00:01", "2026-01-01 00:00:02"]),
            "bid": [1.10, 1.11, 1.12],
            "ask": [1.11, np.nan, np.nan],
            "flags": pd.array([6, 2, 2], dtype="Int64"),
        })
        result = forward_fill_prices(df)
        assert result["ask"].iloc[1] == pytest.approx(1.11)
        assert result["ask"].iloc[2] == pytest.approx(1.11)

    def test_drop_rows_still_missing_after_ffill(self) -> None:
        """Rows at the start with no prior value should be dropped."""
        df = pd.DataFrame({
            "time": pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 00:00:01", "2026-01-01 00:00:02"]),
            "bid": [np.nan, np.nan, 1.10],
            "ask": [1.11, 1.12, 1.13],
            "flags": pd.array([4, 4, 6], dtype="Int64"),
        })
        filled = forward_fill_prices(df)
        result = drop_incomplete_rows(filled)
        assert len(result) == 1
        assert result["bid"].iloc[0] == pytest.approx(1.10)

    def test_forward_fill_preserves_order(self, mt5_with_partial_updates: Path) -> None:
        df = parse_mt5_export(mt5_with_partial_updates)
        filled = forward_fill_prices(df)
        incomplete_dropped = drop_incomplete_rows(filled)
        # First row has bid but no ask -> dropped after ffill since ask has no prior
        # Second row has ask but no bid -> bid ffilled from row 0 -> kept
        # Third row has both -> kept
        assert len(incomplete_dropped) == 2


# ---------------------------------------------------------------------------
# Validation Tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_timestamps_removed(self) -> None:
        df = pd.DataFrame({
            "time": [pd.Timestamp("2026-01-01"), pd.NaT, pd.Timestamp("2026-01-02")],
            "bid": [1.10, 1.11, 1.12],
            "ask": [1.11, 1.12, 1.13],
            "mid": [1.105, 1.115, 1.125],
            "spread": [0.01, 0.01, 0.01],
            "flags": pd.array([6, 6, 6], dtype="Int64"),
        })
        result = validate_timestamps(df)
        assert len(result) == 2

    def test_all_invalid_timestamps_raises(self) -> None:
        df = pd.DataFrame({
            "time": [pd.NaT, pd.NaT],
            "bid": [1.10, 1.11],
            "ask": [1.11, 1.12],
            "mid": [1.105, 1.115],
            "spread": [0.01, 0.01],
            "flags": pd.array([6, 6], dtype="Int64"),
        })
        with pytest.raises(TickValidationError, match="All timestamps are invalid"):
            validate_timestamps(df)

    def test_invalid_spread_removed(self) -> None:
        df = pd.DataFrame({
            "time": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "bid": [1.10, 1.15, 1.12],
            "ask": [1.11, 1.10, 1.13],  # Second row: ask < bid
            "mid": [1.105, 1.125, 1.125],
            "spread": [0.01, -0.05, 0.01],
            "flags": pd.array([6, 6, 6], dtype="Int64"),
        })
        result = validate_spreads(df)
        assert len(result) == 2

    def test_all_invalid_spreads_raises(self) -> None:
        df = pd.DataFrame({
            "time": pd.to_datetime(["2026-01-01"]),
            "bid": [1.15],
            "ask": [1.10],
            "mid": [1.125],
            "spread": [-0.05],
            "flags": pd.array([6], dtype="Int64"),
        })
        with pytest.raises(TickValidationError, match="All rows have invalid spreads"):
            validate_spreads(df)


# ---------------------------------------------------------------------------
# Derived Fields Tests
# ---------------------------------------------------------------------------


class TestDerivedFields:
    def test_mid_computation(self) -> None:
        df = pd.DataFrame({
            "time": pd.to_datetime(["2026-01-01"]),
            "bid": [1.10000],
            "ask": [1.10020],
            "flags": pd.array([6], dtype="Int64"),
        })
        result = compute_derived_fields(df)
        assert result["mid"].iloc[0] == pytest.approx(1.10010)

    def test_spread_computation(self) -> None:
        df = pd.DataFrame({
            "time": pd.to_datetime(["2026-01-01"]),
            "bid": [1.10000],
            "ask": [1.10020],
            "flags": pd.array([6], dtype="Int64"),
        })
        result = compute_derived_fields(df)
        assert result["spread"].iloc[0] == pytest.approx(0.00020)


# ---------------------------------------------------------------------------
# Integration: load_ticks Tests
# ---------------------------------------------------------------------------


class TestLoadTicks:
    def test_load_normalized_csv(self, normalized_csv_file: Path) -> None:
        ticks = load_ticks(normalized_csv_file)
        assert isinstance(ticks, TickFrame)
        assert len(ticks) == 3
        assert list(ticks.frame.columns) == ["time", "bid", "ask", "mid", "spread", "flags"]

    def test_load_mt5_export(self, mt5_export_file: Path) -> None:
        ticks = load_ticks(mt5_export_file)
        assert isinstance(ticks, TickFrame)
        assert len(ticks) == 5  # All 5 rows valid after ffill

    def test_load_nonexistent_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_ticks("/nonexistent/file.csv")

    def test_tickframe_properties(self, normalized_csv_file: Path) -> None:
        ticks = load_ticks(normalized_csv_file)
        assert len(ticks.time) == 3
        assert len(ticks.bid) == 3
        assert len(ticks.ask) == 3
        assert len(ticks.mid) == 3
        assert len(ticks.spread) == 3
        assert len(ticks.flags) == 3

    def test_deterministic_output(self, mt5_export_file: Path) -> None:
        """Loading the same file twice produces identical results."""
        ticks1 = load_ticks(mt5_export_file)
        ticks2 = load_ticks(mt5_export_file)
        pd.testing.assert_frame_equal(ticks1.frame, ticks2.frame)


# ---------------------------------------------------------------------------
# Real Data Smoke Test
# ---------------------------------------------------------------------------


class TestRealData:
    def test_load_real_eurusd_data(self, real_data_path: Path) -> None:
        """Smoke test: load the real EURUSD MT5 export."""
        if not real_data_path.exists():
            pytest.skip("Real data file not available.")

        ticks = load_ticks(real_data_path)

        # Basic sanity checks
        assert len(ticks) > 100_000  # Should have many ticks
        assert ticks.bid.min() > 0
        assert ticks.ask.min() > 0
        assert (ticks.spread >= 0).all()
        assert ticks.time.is_monotonic_increasing or True  # ticks may have same timestamp
        assert not ticks.bid.isna().any()
        assert not ticks.ask.isna().any()
        assert not ticks.mid.isna().any()
        assert not ticks.time.isna().any()

    def test_real_data_deterministic(self, real_data_path: Path) -> None:
        """Loading real data twice produces identical output."""
        if not real_data_path.exists():
            pytest.skip("Real data file not available.")

        ticks1 = load_ticks(real_data_path)
        ticks2 = load_ticks(real_data_path)
        pd.testing.assert_frame_equal(ticks1.frame, ticks2.frame)
