"""Tests for the Production CLI (Story 15)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from wavelet_research.cli_research import build_parser, main, _load_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tick_csv(path: Path, n: int = 500) -> None:
    """Create a test tick CSV file."""
    rng = np.random.default_rng(42)
    mid = 1.10 + np.cumsum(rng.normal(0, 0.00002, n))
    spread = 0.00010
    df = pd.DataFrame({
        "time": pd.date_range("2026-06-01", periods=n, freq="100ms"),
        "bid": mid - spread / 2,
        "ask": mid + spread / 2,
    })
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Parser Tests
# ---------------------------------------------------------------------------


class TestParser:
    def test_build_parser(self) -> None:
        parser = build_parser()
        assert parser is not None
        assert parser.prog == "wavelet-research"

    def test_research_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["research", "--ticks", "test.csv"])
        assert args.command == "research"
        assert args.ticks == "test.csv"

    def test_optimize_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize", "--ticks", "test.csv", "--random"])
        assert args.command == "optimize"
        assert args.random is True

    def test_validate_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["validate", "--ticks", "test.csv", "--folds", "5"])
        assert args.command == "validate"
        assert args.folds == 5

    def test_paper_trade_command(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["paper-trade", "--ticks", "test.csv", "--balance", "5000"])
        assert args.command == "paper-trade"
        assert args.balance == 5000.0

    def test_no_command_returns_1(self) -> None:
        with patch("sys.argv", ["wavelet-research"]):
            result = main()
        assert result == 1


# ---------------------------------------------------------------------------
# Data Loading Tests
# ---------------------------------------------------------------------------


class TestDataLoading:
    def test_load_valid_csv(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            path = Path(f.name)
            _make_tick_csv(path, n=100)

        df = _load_data(str(path))
        assert "mid" in df.columns
        assert "spread" in df.columns
        assert len(df) == 100
        path.unlink()

    def test_load_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            _load_data("/nonexistent/path.csv")

    def test_load_missing_columns(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            path = Path(f.name)
            pd.DataFrame({"x": [1, 2]}).to_csv(path, index=False)

        with pytest.raises(ValueError, match="Missing required"):
            _load_data(str(path))
        path.unlink()


# ---------------------------------------------------------------------------
# CLI Integration Tests
# ---------------------------------------------------------------------------


class TestCLIIntegration:
    def test_research_command_runs(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            path = Path(f.name)
            _make_tick_csv(path, n=400)

        with patch("sys.argv", [
            "wavelet-research", "research",
            "--ticks", str(path),
            "--wavelets", "haar",
            "--windows", "256",
            "--levels", "2",
            "--buy-z", "2.0",
            "--sell-z", "2.0",
            "--max-hold", "10",
        ]):
            result = main()
        assert result == 0
        path.unlink()

    def test_optimize_command_runs(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            path = Path(f.name)
            _make_tick_csv(path, n=400)

        with patch("sys.argv", [
            "wavelet-research", "optimize",
            "--ticks", str(path),
            "--wavelets", "haar",
            "--windows", "256",
            "--levels", "2",
            "--buy-z", "1.5,2.0",
            "--sell-z", "1.5",
            "--max-hold", "10",
            "--min-trades", "0",
        ]):
            result = main()
        assert result == 0
        path.unlink()

    def test_validate_command_runs(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            path = Path(f.name)
            _make_tick_csv(path, n=600)

        with patch("sys.argv", [
            "wavelet-research", "validate",
            "--ticks", str(path),
            "--wavelet", "haar",
            "--window", "256",
            "--level", "2",
            "--folds", "2",
        ]):
            result = main()
        assert result == 0
        path.unlink()

    def test_paper_trade_command_runs(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            path = Path(f.name)
            _make_tick_csv(path, n=500)

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as out:
            out_path = out.name

        with patch("sys.argv", [
            "wavelet-research", "paper-trade",
            "--ticks", str(path),
            "--output", out_path,
        ]):
            result = main()
        assert result == 0
        path.unlink()
        Path(out_path).unlink(missing_ok=True)

    def test_error_handling(self) -> None:
        with patch("sys.argv", [
            "wavelet-research", "research",
            "--ticks", "/nonexistent.csv",
        ]):
            result = main()
        assert result == 1
