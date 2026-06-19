"""Databento trades -> 1-minute OHLCV loader.

The OHLCV-1m loader (`data_loader.py`) handles pre-aggregated bars. This
module handles the raw *trades* schema (tick-level, with an aggressor `side`
column we don't use yet but which enables a future true-delta upgrade), and
aggregates it into the same `list[Bar]` the engine consumes:

    open  = first trade price in the minute
    high  = max trade price
    low   = min trade price
    close = last trade price
    volume= sum of trade sizes

A trades archive can span many contracts (e.g. NQZ5, NQH6, NQM6 as the
front month rolls). Each daily file is one UTC day, so minutes never cross
files and per-file aggregation can simply be concatenated.

Used by the walk-forward harness to build independent per-contract samples.
"""
from __future__ import annotations

import glob
import io
import os
from collections import defaultdict

import pandas as pd
import zstandard as zstd

from strategy.bars import Bar

_TRADES_COLS = ["ts_event", "action", "side", "price", "size", "symbol"]


def _read_trades_csv(path: str) -> pd.DataFrame:
    if path.endswith(".zst"):
        with open(path, "rb") as f:
            data = zstd.ZstdDecompressor().stream_reader(f).read()
        return pd.read_csv(io.BytesIO(data), usecols=_TRADES_COLS)
    return pd.read_csv(path, usecols=_TRADES_COLS)


def _agg_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Group a (already symbol-filtered, trades-only) frame to 1m OHLCV+delta.

    Expects a `ts` column floored to the minute and a signed `signed_size`
    column (+size for buy aggressor, -size for sell). Returns columns
    ts/open/high/low/close/volume/delta, sorted by ts.
    """
    g = (
        df.groupby("ts")
        .agg(
            open=("price", "first"),
            high=("price", "max"),
            low=("price", "min"),
            close=("price", "last"),
            volume=("size", "sum"),
            delta=("signed_size", "sum"),
        )
        .reset_index()
        .sort_values("ts")
    )
    return g


def _add_signed_size(df: pd.DataFrame) -> pd.DataFrame:
    """Signed aggressor size: +size for buy ('B'), -size for sell ('A'), 0 ('N')."""
    sign = df["side"].map({"B": 1, "A": -1}).fillna(0)
    df = df.copy()
    df["signed_size"] = df["size"] * sign
    return df


def _frame_to_bars(g: pd.DataFrame) -> list[Bar]:
    return [
        Bar(ts=r.ts, open=float(r.open), high=float(r.high), low=float(r.low),
            close=float(r.close), volume=float(r.volume), delta=float(r.delta))
        for r in g.itertuples(index=False)
    ]


def aggregate_trades_to_1m_bars(df: pd.DataFrame, symbol: str) -> list[Bar]:
    """Aggregate a raw trades DataFrame to 1m bars for one symbol.

    df must have columns ts_event/action/price/size/symbol. Keeps only
    action == 'T' rows for `symbol`. Pure (no I/O) — the unit-tested core.
    """
    sub = df[(df["action"] == "T") & (df["symbol"] == symbol)].copy()
    if sub.empty:
        return []
    sub["ts"] = pd.to_datetime(sub["ts_event"], utc=True).dt.floor("min")
    return _frame_to_bars(_agg_frame(_add_signed_size(sub)))


def symbol_volumes(directory: str) -> dict[str, int]:
    """Total traded size per symbol across all daily files in `directory`."""
    vols: dict[str, int] = defaultdict(int)
    for path in sorted(glob.glob(os.path.join(directory, "*.trades.csv.zst"))):
        df = _read_trades_csv(path)
        df = df[df["action"] == "T"]
        for sym, v in df.groupby("symbol")["size"].sum().items():
            vols[str(sym)] += int(v)
    return dict(sorted(vols.items(), key=lambda kv: -kv[1]))


def load_trades_dir_ohlcv(directory: str, symbols: list[str]) -> dict[str, list[Bar]]:
    """Aggregate every daily trades file in `directory` to 1m bars, for each
    requested symbol. Reads each file once. Returns {symbol: list[Bar]}."""
    per_sym: dict[str, list[pd.DataFrame]] = {s: [] for s in symbols}
    for path in sorted(glob.glob(os.path.join(directory, "*.trades.csv.zst"))):
        df = _read_trades_csv(path)
        df = df[df["action"] == "T"]
        df["ts"] = pd.to_datetime(df["ts_event"], utc=True).dt.floor("min")
        df = _add_signed_size(df)
        for sym in symbols:
            sub = df[df["symbol"] == sym]
            if not sub.empty:
                per_sym[sym].append(_agg_frame(sub))
    out: dict[str, list[Bar]] = {}
    for sym, frames in per_sym.items():
        if not frames:
            out[sym] = []
            continue
        combined = pd.concat(frames, ignore_index=True).sort_values("ts")
        out[sym] = _frame_to_bars(combined)
    return out
