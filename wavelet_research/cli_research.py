"""Production CLI for the Wavelet Research Framework.

Provides commands for running research experiments, optimization,
validation, and paper trading sessions.
"""

from __future__ import annotations

import argparse
import json
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
from wavelet_research.research.final_gate import GateConfig, GateMetrics, evaluate_gate
from wavelet_research.trend_quality.audit import TrendAuditor
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


def cmd_trend_audit(args: argparse.Namespace) -> int:
    """Run trend quality audit on historical data."""
    _setup_logging(args.verbose)
    data = _load_data(args.ticks)

    engine_config = WaveletEngineConfig(
        wavelet=args.wavelet, window=args.window, level=args.level,
    )
    auditor = TrendAuditor(engine_config)
    report = auditor.audit(data)

    result = report.to_dict()
    print(json.dumps(result, indent=2))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"Report saved: {args.output}")

    return 0 if report.recommendation.value != "fail" else 1


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Derive conservative thresholds from historical data."""
    _setup_logging(args.verbose)
    data = _load_data(args.ticks)

    # Deterministic grid search over key thresholds
    from wavelet_research.backtest.core import BacktestEngine
    from wavelet_research.engine.core import WaveletEngine
    from wavelet_research.signal.core import SignalEngine

    best_score = -1.0
    best_cfg: dict = {}

    for min_dev in (0.8, 1.0, 1.2, 1.5):
        for max_hold in (10, 20, 40):
            eng_cfg = WaveletEngineConfig(wavelet=args.wavelet, window=args.window, level=2)
            sig_cfg = SignalConfig(
                buy_z_threshold=min_dev,
                sell_z_threshold=min_dev,
                slope_filter_enabled=False,
                min_normalized_deviation=min_dev,
                min_return_probability=0.55,
                min_stats_sample_size=30,
            )
            bt_cfg = BacktestConfig(
                pip_size=0.00001,
                exit_strategy=ExitStrategy.RETURN_TO_TREND,
                max_hold_ticks=max_hold,
            )
            try:
                engine = WaveletEngine(eng_cfg)
                signal_engine = SignalEngine(sig_cfg)
                bt_engine = BacktestEngine(bt_cfg)
                report = bt_engine.run(data, engine, signal_engine)
                if report.trades < 5:
                    continue
                score = report.profit_factor * (1.0 - abs(report.max_drawdown) / max(abs(report.total_pnl), 1.0))
                if score > best_score:
                    best_score = score
                    best_cfg = {
                        "symbol": args.symbol,
                        "wavelet": args.wavelet,
                        "window": args.window,
                        "entry": {
                            "min_normalized_deviation": min_dev,
                            "min_return_probability": 0.55,
                            "min_stats_sample_size": 30,
                        },
                        "exit": {
                            "exit_strategy": "return_to_trend",
                            "max_holding_bars": max_hold,
                            "max_adverse_normalized_deviation": 2.5,
                        },
                        "filters": {
                            "max_spread": 0.0003,
                            "cooldown_bars": 5,
                        },
                        "validation_summary": {
                            "trades": report.trades,
                            "profit_factor": report.profit_factor,
                            "total_pnl": report.total_pnl,
                            "score": score,
                        },
                    }
            except Exception as exc:
                logger.debug("Config failed: %s", exc)

    if not best_cfg:
        print("No valid configuration found")
        return 1

    print(json.dumps(best_cfg, indent=2))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(best_cfg, indent=2))
        print(f"Config saved: {args.output}")
    return 0


def cmd_final_gate(args: argparse.Namespace) -> int:
    """Run the final statistical gate."""
    _setup_logging(args.verbose)

    metrics = GateMetrics(
        total_trades=args.trades,
        profit_factor=args.profit_factor,
        expectancy=args.expectancy,
        max_drawdown=args.max_drawdown_val,
        gross_profit=max(args.gross_profit, 0.01),
        avg_holding_bars=args.avg_holding,
        monte_carlo_survival=args.mc_survival,
        walk_forward_stability=args.wf_stability,
        paper_consistency=args.paper_consistency,
    )
    config = GateConfig(
        min_trades=args.min_trades_gate,
        min_profit_factor=args.min_pf,
        min_expectancy=args.min_exp,
        max_drawdown_pct=args.max_dd_pct,
        min_monte_carlo_survival=args.min_mc,
        min_walk_forward_stability=args.min_wf,
        min_paper_consistency=args.min_paper,
    )
    result = evaluate_gate(metrics, config)
    output = result.to_dict()
    print(json.dumps(output, indent=2))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(output, indent=2))
        print(f"Gate report saved: {args.output}")

    return 0 if result.decision.value == "PASS" else 1


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

    # Trend Audit
    trend_audit = subparsers.add_parser("trend-audit", help="Audit trend quality")
    trend_audit.add_argument("--ticks", required=True, help="Path to tick CSV")
    trend_audit.add_argument("--wavelet", default="db4")
    trend_audit.add_argument("--window", type=int, default=512)
    trend_audit.add_argument("--level", type=int, default=2)
    trend_audit.add_argument("--output", default=None, help="Output JSON path")

    # Calibrate
    calibrate = subparsers.add_parser("calibrate-trend-strategy", help="Calibrate thresholds")
    calibrate.add_argument("--ticks", required=True, help="Path to tick CSV")
    calibrate.add_argument("--wavelet", default="db4")
    calibrate.add_argument("--window", type=int, default=512)
    calibrate.add_argument("--symbol", default="UNKNOWN")
    calibrate.add_argument("--output", default=None, help="Output JSON path")

    # Final Gate
    gate = subparsers.add_parser("final-gate", help="Run final statistical gate")
    gate.add_argument("--trades", type=int, required=True)
    gate.add_argument("--profit-factor", type=float, required=True)
    gate.add_argument("--expectancy", type=float, required=True)
    gate.add_argument("--max-drawdown-val", type=float, required=True)
    gate.add_argument("--gross-profit", type=float, required=True)
    gate.add_argument("--avg-holding", type=float, default=15.0)
    gate.add_argument("--mc-survival", type=float, default=0.75)
    gate.add_argument("--wf-stability", type=float, default=0.65)
    gate.add_argument("--paper-consistency", type=float, default=0.75)
    gate.add_argument("--min-trades-gate", type=int, default=50)
    gate.add_argument("--min-pf", type=float, default=1.3)
    gate.add_argument("--min-exp", type=float, default=0.0)
    gate.add_argument("--max-dd-pct", type=float, default=0.30)
    gate.add_argument("--min-mc", type=float, default=0.70)
    gate.add_argument("--min-wf", type=float, default=0.60)
    gate.add_argument("--min-paper", type=float, default=0.70)
    gate.add_argument("--output", default=None, help="Output JSON path")

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
        "trend-audit": cmd_trend_audit,
        "calibrate-trend-strategy": cmd_calibrate,
        "final-gate": cmd_final_gate,
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
