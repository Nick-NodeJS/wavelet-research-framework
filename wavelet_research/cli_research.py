"""Production CLI for the Wavelet Research Framework.

Provides commands for running research experiments, optimization,
validation, and paper trading sessions.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from wavelet_research.backtest.config import BacktestConfig, ExitStrategy
from wavelet_research.engine.config import WaveletEngineConfig
from wavelet_research.optimizer.config import (
    ConstraintConfig,
    ObjectiveConfig,
    OptimizerConfig,
    SearchMethod,
)
from wavelet_research.optimizer.core import ParameterOptimizer
from wavelet_research.orchestrator.config import PipelineConfig
from wavelet_research.orchestrator.core import ExperimentOrchestrator
from wavelet_research.orchestrator.matrix import ParameterMatrix
from wavelet_research.paper_trading.core import PaperTrader
from wavelet_research.mt5.expert_advisor import EAConfig
from wavelet_research.mt5.risk import RiskConfig
from wavelet_research.signal.config import SignalConfig
from wavelet_research.validation.core import WalkForwardValidator
from wavelet_research.validation.splits import SplitConfig

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _load_data(path: str) -> pd.DataFrame:
    """Load and normalize tick data.

    Parameters
    ----------
    path : str
        Path to CSV file.

    Returns
    -------
    pd.DataFrame
        Normalized DataFrame.

    Raises
    ------
    FileNotFoundError
        If file doesn't exist.
    ValueError
        If required columns missing.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(filepath)

    required = {"bid", "ask"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Missing required columns: {required - set(df.columns)}")

    if "mid" not in df.columns:
        df["mid"] = (df["bid"] + df["ask"]) / 2
    if "spread" not in df.columns:
        df["spread"] = df["ask"] - df["bid"]
    if "time" not in df.columns:
        df["time"] = pd.date_range("2026-01-01", periods=len(df), freq="100ms")
    else:
        df["time"] = pd.to_datetime(df["time"])

    return df


def cmd_research(args: argparse.Namespace) -> int:
    """Run research experiments."""
    _setup_logging(args.verbose)
    data = _load_data(args.ticks)

    matrix = ParameterMatrix(
        wavelets=tuple(args.wavelets.split(",")),
        windows=tuple(int(w) for w in args.windows.split(",")),
        levels=tuple(int(l) for l in args.levels.split(",")),
        buy_z_thresholds=tuple(float(z) for z in args.buy_z.split(",")),
        sell_z_thresholds=tuple(float(z) for z in args.sell_z.split(",")),
        exit_strategies=tuple(ExitStrategy(e) for e in args.exits.split(",")),
        max_hold_ticks_list=tuple(int(h) for h in args.max_hold.split(",")),
    )

    configs = matrix.generate()
    logger.info("Generated %d configurations", len(configs))

    output_path = Path(args.output) if args.output else None
    orchestrator = ExperimentOrchestrator(configs, output_path=str(output_path) if output_path else None)
    results = orchestrator.run(data)

    print(f"Completed {len(results)} experiments")
    if results:
        best = results[0]
        print(f"Best: {best.config.identifier} (PnL={best.report.total_pnl:.2f})")

    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    """Run parameter optimization."""
    _setup_logging(args.verbose)
    data = _load_data(args.ticks)

    method = SearchMethod.RANDOM if args.random else SearchMethod.GRID
    cfg = OptimizerConfig(
        search_method=method,
        wavelets=tuple(args.wavelets.split(",")),
        windows=tuple(int(w) for w in args.windows.split(",")),
        levels=tuple(int(l) for l in args.levels.split(",")),
        buy_z_thresholds=tuple(float(z) for z in args.buy_z.split(",")),
        sell_z_thresholds=tuple(float(z) for z in args.sell_z.split(",")),
        max_hold_ticks_list=tuple(int(h) for h in args.max_hold.split(",")),
        max_iterations=args.max_iter,
        seed=args.seed,
        constraints=ConstraintConfig(
            min_trades=args.min_trades,
            max_drawdown=args.max_drawdown,
        ),
    )

    optimizer = ParameterOptimizer(cfg)
    report = optimizer.optimize(data)

    print(f"Evaluated: {report.total_evaluated}")
    print(f"Passed: {report.total_passed}")
    if report.best_configs:
        best = report.best_configs[0]
        print(f"Best: {best.config.identifier} (score={best.score:.4f})")

    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Run walk-forward validation."""
    _setup_logging(args.verbose)
    data = _load_data(args.ticks)

    config = PipelineConfig(
        wavelet_config=WaveletEngineConfig(
            wavelet=args.wavelet, window=args.window, level=args.level,
        ),
        signal_config=SignalConfig(
            buy_z_threshold=args.buy_z_val, sell_z_threshold=args.sell_z_val,
            slope_filter_enabled=False,
        ),
        backtest_config=BacktestConfig(
            pip_size=0.00001,
            exit_strategy=ExitStrategy.MAX_HOLD,
            max_hold_ticks=args.max_hold_val,
        ),
    )

    split_cfg = SplitConfig(
        in_sample_ratio=args.is_ratio, n_folds=args.folds,
    )

    validator = WalkForwardValidator(config, split_cfg)
    report = validator.validate(data)

    print(f"IS PnL: {report.in_sample_metrics.total_pnl:.2f}")
    print(f"OOS PnL: {report.out_of_sample_metrics.total_pnl:.2f}")
    print(f"Mean OOS Efficiency: {report.mean_oos_efficiency:.4f}")
    for wf in report.walk_forward_results:
        print(f"  Fold {wf.fold_index}: efficiency={wf.oos_efficiency:.4f}")

    return 0


def cmd_paper_trade(args: argparse.Namespace) -> int:
    """Run paper trading session."""
    _setup_logging(args.verbose)
    data = _load_data(args.ticks)

    ea_config = EAConfig(
        wavelet_config=WaveletEngineConfig(
            wavelet=args.wavelet, window=args.window, level=args.level,
        ),
        signal_config=SignalConfig(
            buy_z_threshold=args.buy_z_val, sell_z_threshold=args.sell_z_val,
            slope_filter_enabled=False,
        ),
        risk_config=RiskConfig(
            max_open_positions=args.max_positions,
            lot_size=args.lot_size,
        ),
        stop_loss_pips=args.stop_loss,
        take_profit_pips=args.take_profit,
    )

    trader = PaperTrader(ea_config, initial_balance=args.balance)
    journal = trader.run(data)

    print(f"Trades: {journal.count}")
    print(f"Total PnL: {journal.total_pnl:.2f}")
    print(f"Win Rate: {journal.win_rate:.2%}")
    print(f"Profit Factor: {journal.profit_factor:.2f}")
    print(f"Final Balance: {trader.balance:.2f}")

    if args.output:
        df = journal.to_dataframe()
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"Journal saved: {args.output}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="wavelet-research",
        description="Wavelet Research Framework CLI",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Research
    research = subparsers.add_parser("research", help="Run research experiments")
    research.add_argument("--ticks", required=True, help="Path to tick CSV")
    research.add_argument("--wavelets", default="haar,db4", help="Wavelets (comma-sep)")
    research.add_argument("--windows", default="256,512", help="Windows (comma-sep)")
    research.add_argument("--levels", default="2", help="Levels (comma-sep)")
    research.add_argument("--buy-z", default="1.5,2.0", help="Buy Z thresholds")
    research.add_argument("--sell-z", default="1.5,2.0", help="Sell Z thresholds")
    research.add_argument("--exits", default="max_hold", help="Exit strategies")
    research.add_argument("--max-hold", default="10,50", help="Max hold ticks")
    research.add_argument("--output", default=None, help="Output CSV path")

    # Optimize
    optimize = subparsers.add_parser("optimize", help="Run parameter optimization")
    optimize.add_argument("--ticks", required=True, help="Path to tick CSV")
    optimize.add_argument("--wavelets", default="haar,db4")
    optimize.add_argument("--windows", default="256,512")
    optimize.add_argument("--levels", default="2")
    optimize.add_argument("--buy-z", default="1.0,1.5,2.0,2.5")
    optimize.add_argument("--sell-z", default="1.0,1.5,2.0,2.5")
    optimize.add_argument("--max-hold", default="10,50")
    optimize.add_argument("--random", action="store_true", help="Use random search")
    optimize.add_argument("--max-iter", type=int, default=100)
    optimize.add_argument("--seed", type=int, default=42)
    optimize.add_argument("--min-trades", type=int, default=5)
    optimize.add_argument("--max-drawdown", type=float, default=100.0)

    # Validate
    validate = subparsers.add_parser("validate", help="Walk-forward validation")
    validate.add_argument("--ticks", required=True, help="Path to tick CSV")
    validate.add_argument("--wavelet", default="haar")
    validate.add_argument("--window", type=int, default=256)
    validate.add_argument("--level", type=int, default=2)
    validate.add_argument("--buy-z-val", type=float, default=1.5)
    validate.add_argument("--sell-z-val", type=float, default=1.5)
    validate.add_argument("--max-hold-val", type=int, default=10)
    validate.add_argument("--is-ratio", type=float, default=0.7)
    validate.add_argument("--folds", type=int, default=3)

    # Paper Trade
    paper = subparsers.add_parser("paper-trade", help="Run paper trading")
    paper.add_argument("--ticks", required=True, help="Path to tick CSV")
    paper.add_argument("--wavelet", default="haar")
    paper.add_argument("--window", type=int, default=256)
    paper.add_argument("--level", type=int, default=2)
    paper.add_argument("--buy-z-val", type=float, default=1.5)
    paper.add_argument("--sell-z-val", type=float, default=1.5)
    paper.add_argument("--stop-loss", type=float, default=20.0)
    paper.add_argument("--take-profit", type=float, default=30.0)
    paper.add_argument("--max-positions", type=int, default=1)
    paper.add_argument("--lot-size", type=float, default=0.01)
    paper.add_argument("--balance", type=float, default=10000.0)
    paper.add_argument("--output", default=None, help="Output journal CSV")

    return parser


def main() -> int:
    """CLI entry point.

    Returns
    -------
    int
        Exit code.
    """
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    commands = {
        "research": cmd_research,
        "optimize": cmd_optimize,
        "validate": cmd_validate,
        "paper-trade": cmd_paper_trade,
    }

    handler = commands.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    try:
        return handler(args)
    except Exception as e:
        logger.error("Error: %s", e)
        if args.verbose:
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())
