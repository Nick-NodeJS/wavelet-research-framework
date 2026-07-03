from dataclasses import dataclass, asdict
from typing import List
import numpy as np
import pandas as pd

from .wavelet_engine import WaveletConfig, causal_wavelet_point
from .signals import Signal, decide


@dataclass
class Trade:
    entry_index: int
    exit_index: int
    side: str
    entry_price: float
    exit_price: float
    pnl_pips: float
    mae_pips: float
    mfe_pips: float
    reason: str


@dataclass
class BacktestResult:
    config: WaveletConfig
    trades: int
    win_rate: float
    profit_factor: float
    expectancy: float
    max_drawdown_pips: float
    avg_win: float
    avg_loss: float
    total_pips: float


def _max_drawdown(equity):
    peak = -1e18
    max_dd = 0.0
    for x in equity:
        peak = max(peak, x)
        max_dd = min(max_dd, x - peak)
    return abs(max_dd)


def run_backtest(
    ticks: pd.DataFrame,
    cfg: WaveletConfig,
    pip_size: float,
    max_hold: int,
) -> tuple[BacktestResult, List[Trade]]:
    mid = ((ticks["bid"] + ticks["ask"]) / 2.0).to_numpy()
    bid = ticks["bid"].to_numpy()
    ask = ticks["ask"].to_numpy()

    trades: List[Trade] = []
    i = cfg.window

    while i < len(ticks) - max_hold - 1:
        point = causal_wavelet_point(mid[: i + 1], cfg)
        decision = decide(point, cfg.threshold)

        if decision.signal == Signal.HOLD:
            i += 1
            continue

        side = decision.signal.value
        entry_index = i

        if side == "BUY":
            entry_price = ask[i]
        else:
            entry_price = bid[i]

        # v0 exit: hold fixed max_hold ticks.
        # Later: exit on return to trend, partial close, opposite signal.
        exit_index = i + max_hold

        if side == "BUY":
            exit_price = bid[exit_index]
            path = bid[i:exit_index + 1] - entry_price
        else:
            exit_price = ask[exit_index]
            path = entry_price - ask[i:exit_index + 1]

        pnl_pips = float((path[-1]) / pip_size)
        mae_pips = float(np.min(path) / pip_size)
        mfe_pips = float(np.max(path) / pip_size)

        trades.append(
            Trade(
                entry_index=entry_index,
                exit_index=exit_index,
                side=side,
                entry_price=float(entry_price),
                exit_price=float(exit_price),
                pnl_pips=pnl_pips,
                mae_pips=mae_pips,
                mfe_pips=mfe_pips,
                reason=decision.reason,
            )
        )

        # Avoid immediate duplicate entries.
        i = exit_index + 1

    pnls = np.array([t.pnl_pips for t in trades], dtype=float)

    if len(pnls) == 0:
        result = BacktestResult(cfg, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return result, trades

    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    gross_profit = float(np.sum(wins)) if len(wins) else 0.0
    gross_loss = abs(float(np.sum(losses))) if len(losses) else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    equity = np.cumsum(pnls)

    result = BacktestResult(
        config=cfg,
        trades=len(trades),
        win_rate=float(len(wins) / len(pnls)),
        profit_factor=float(profit_factor),
        expectancy=float(np.mean(pnls)),
        max_drawdown_pips=float(_max_drawdown(equity)),
        avg_win=float(np.mean(wins)) if len(wins) else 0.0,
        avg_loss=float(np.mean(losses)) if len(losses) else 0.0,
        total_pips=float(np.sum(pnls)),
    )

    return result, trades


def result_to_row(result: BacktestResult) -> dict:
    row = asdict(result.config)
    row.update(
        {
            "trades": result.trades,
            "win_rate": result.win_rate,
            "profit_factor": result.profit_factor,
            "expectancy": result.expectancy,
            "max_drawdown_pips": result.max_drawdown_pips,
            "avg_win": result.avg_win,
            "avg_loss": result.avg_loss,
            "total_pips": result.total_pips,
        }
    )
    return row
