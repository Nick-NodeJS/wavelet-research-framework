"""CLI for inspecting tick data files.

Usage:
    python3 -m wavelet_research.cli_inspect_ticks --input data/EURUSD.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback

from wavelet_research.ingestion.statistics import format_statistics, inspect_ticks
from wavelet_research.ingestion.validation import TickValidationError


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="cli_inspect_ticks",
        description="Inspect and validate MT5 tick data files.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the tick data file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the tick inspection CLI.

    Parameters
    ----------
    argv : list[str] or None
        Command-line arguments. Uses sys.argv if None.

    Returns
    -------
    int
        Exit code: 0 on success, non-zero on failure.
    """
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    try:
        stats = inspect_ticks(args.input)
        print(format_statistics(stats))
        return 0
    except (TickValidationError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
