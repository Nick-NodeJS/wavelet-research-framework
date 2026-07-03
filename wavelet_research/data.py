from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class TickData:
    frame: pd.DataFrame

    @property
    def mid(self):
        return (self.frame["bid"] + self.frame["ask"]) / 2.0


def load_ticks(path: str) -> TickData:
    df = pd.read_csv(path)

    required = {"time", "bid", "ask"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df["bid"] = pd.to_numeric(df["bid"], errors="coerce")
    df["ask"] = pd.to_numeric(df["ask"], errors="coerce")

    df = df.dropna(subset=["time", "bid", "ask"])
    df = df.sort_values("time").reset_index(drop=True)

    if len(df) < 100:
        raise ValueError("Too few ticks for research backtest")

    if (df["ask"] < df["bid"]).any():
        raise ValueError("Invalid data: ask < bid exists")

    return TickData(df)
