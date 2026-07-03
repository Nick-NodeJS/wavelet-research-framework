import argparse
from pathlib import Path
import pandas as pd

from .data import load_ticks
from .wavelet_engine import WaveletConfig
from .backtester import run_backtest, result_to_row


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ticks", required=True, help="CSV with columns: time,bid,ask")
    p.add_argument("--symbol", default="EURUSD")
    p.add_argument("--pip-size", type=float, default=0.0001)
    p.add_argument("--max-hold", type=int, default=500)
    p.add_argument("--output", default="reports/report.csv")
    p.add_argument("--trades-output", default="reports/trades.csv")
    return p.parse_args()


def main():
    args = parse_args()
    data = load_ticks(args.ticks)

    wavelets = ["haar", "db4", "db6", "sym4", "coif3"]
    windows = [256, 512, 1024, 2048, 4096]
    levels = [1, 2, 3, 4]
    thresholds = [2.0, 2.5, 3.0, 3.5]

    rows = []
    all_trades = []

    for wavelet in wavelets:
        for window in windows:
            for level in levels:
                for threshold in thresholds:
                    if len(data.frame) <= window + args.max_hold + 10:
                        continue

                    cfg = WaveletConfig(
                        wavelet=wavelet,
                        window=window,
                        level=level,
                        threshold=threshold,
                        vol_window=min(256, window // 2),
                    )

                    try:
                        result, trades = run_backtest(
                            ticks=data.frame,
                            cfg=cfg,
                            pip_size=args.pip_size,
                            max_hold=args.max_hold,
                        )
                    except Exception as e:
                        print(f"skip {cfg}: {e}")
                        continue

                    row = result_to_row(result)
                    row["symbol"] = args.symbol
                    rows.append(row)

                    for t in trades:
                        d = t.__dict__.copy()
                        d.update({
                            "symbol": args.symbol,
                            "wavelet": wavelet,
                            "window": window,
                            "level": level,
                            "threshold": threshold,
                        })
                        all_trades.append(d)

    report = pd.DataFrame(rows)
    if not report.empty:
        report = report.sort_values(
            by=["profit_factor", "expectancy", "total_pips"],
            ascending=[False, False, False],
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False)

    trades_df = pd.DataFrame(all_trades)
    Path(args.trades_output).parent.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(args.trades_output, index=False)

    print(f"Report written: {args.output}")
    print(f"Trades written: {args.trades_output}")

    if not report.empty:
        print(report.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
