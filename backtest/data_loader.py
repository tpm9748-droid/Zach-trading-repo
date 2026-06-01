"""Databento OHLCV-1m loader.

Reads a Databento `ohlcv-1m` CSV (optionally zstd-compressed) and produces
a list[Bar] suitable for run_backtest().

Databento CSV columns (when `map_symbols=true` and `pretty_*=true`):
    ts_event       ISO-8601 UTC, e.g. 2026-05-01T00:00:00.000000000Z
    rtype          int
    publisher_id   int
    instrument_id  int    (CME contract id — changes on roll)
    open, high, low, close   float (already "pretty" decimal)
    volume         int
    symbol         str    (e.g. NQM6 = NQ June 2026)

Roll handling: a single OHLCV file can span multiple contracts if the
date range crosses an expiry. The loader detects multiple symbols and
either:
  - raises if symbol_filter is None and multiple symbols present, OR
  - keeps only bars matching the requested symbol, OR
  - if symbol_filter='auto', keeps the symbol with the highest cumulative
    volume (the dominant contract for the period).
"""
from __future__ import annotations

import io
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import zstandard as zstd

from strategy.bars import Bar


@dataclass(frozen=True)
class DataShape:
    """Quick summary of a loaded dataset — for verification, not analysis."""
    n_bars: int
    symbols: dict[str, int]  # symbol -> bar count
    start_ts: pd.Timestamp
    end_ts: pd.Timestamp
    has_gaps: bool


def _open_csv(path: Path) -> pd.DataFrame:
    """Read a .csv or .csv.zst into a DataFrame."""
    if path.suffix == ".zst":
        with open(path, "rb") as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                buf = io.BytesIO(reader.read())
        return pd.read_csv(buf)
    return pd.read_csv(path)


def load_databento_ohlcv(
    path: str | Path,
    symbol_filter: Optional[str] = None,
) -> list[Bar]:
    """Load Databento OHLCV-1m bars from a CSV.zst file.

    symbol_filter:
      - None:        require a single symbol (raise otherwise)
      - 'auto':      keep only the highest-volume symbol (front-month stitching)
      - 'NQM6' etc.: keep only that symbol
    """
    path = Path(path)
    df = _open_csv(path)

    required = {"ts_event", "open", "high", "low", "close", "volume", "symbol"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Databento CSV missing columns: {sorted(missing)}")

    if symbol_filter is None:
        unique_syms = df["symbol"].unique()
        if len(unique_syms) > 1:
            raise ValueError(
                f"Multiple symbols in {path.name}: {list(unique_syms)}. "
                "Pass symbol_filter='auto' or an explicit symbol."
            )
    elif symbol_filter == "auto":
        # Keep the symbol with the highest cumulative volume.
        vol_by_sym = df.groupby("symbol")["volume"].sum().sort_values(ascending=False)
        chosen = vol_by_sym.index[0]
        df = df[df["symbol"] == chosen].copy()
    else:
        df = df[df["symbol"] == symbol_filter].copy()
        if df.empty:
            raise ValueError(f"No bars found for symbol={symbol_filter!r} in {path.name}")

    df["ts"] = pd.to_datetime(df["ts_event"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)

    bars: list[Bar] = []
    for row in df.itertuples(index=False):
        bars.append(Bar(
            ts=row.ts,
            open=float(row.open), high=float(row.high),
            low=float(row.low), close=float(row.close),
            volume=float(row.volume),
        ))
    return bars


def shape_of(bars: Iterable[Bar], expected_step: pd.Timedelta = pd.Timedelta(minutes=1)) -> DataShape:
    """Summarize a bar series — useful for sanity-checking a load."""
    bars_list = list(bars)
    if not bars_list:
        raise ValueError("empty bar series")
    timestamps = [b.ts for b in bars_list]
    has_gaps = False
    for prev, nxt in zip(timestamps, timestamps[1:]):
        if (nxt - prev) > expected_step:
            has_gaps = True
            break
    # We don't have the symbol on Bar, so the caller usually wants to compute
    # per-symbol counts from the original DataFrame. Provide an empty dict
    # when called on bars alone.
    return DataShape(
        n_bars=len(bars_list),
        symbols={},
        start_ts=timestamps[0],
        end_ts=timestamps[-1],
        has_gaps=has_gaps,
    )


def inspect_databento_file(path: str | Path) -> DataShape:
    """Cheap parse-and-summarize without building Bar objects.

    Useful when you just want to know what's in a file before committing
    to the full load.
    """
    path = Path(path)
    df = _open_csv(path)
    df["ts"] = pd.to_datetime(df["ts_event"], utc=True)
    df = df.sort_values("ts")
    sym_counts = df["symbol"].value_counts().to_dict()
    diffs = df["ts"].diff().dropna()
    has_gaps = bool((diffs > pd.Timedelta(minutes=1)).any())
    return DataShape(
        n_bars=len(df),
        symbols={str(k): int(v) for k, v in sym_counts.items()},
        start_ts=df["ts"].iloc[0],
        end_ts=df["ts"].iloc[-1],
        has_gaps=has_gaps,
    )
