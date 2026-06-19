"""Bar data structures.

Bar is a single OHLCV bar with a UTC timestamp. BarSeries wraps an ordered
list of bars with an explicit cursor — the engine advances the cursor and
strategy code may only look at bars up to and including the cursor. This
makes lookahead bugs structurally impossible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd


@dataclass(frozen=True)
class Bar:
    ts: pd.Timestamp  # tz-aware, UTC
    open: float
    high: float
    low: float
    close: float
    volume: float
    # Signed aggressor delta (buy size - sell size) when built from tick
    # trades; 0.0 for pre-aggregated OHLCV bars that lack order-flow data.
    delta: float = 0.0

    def __post_init__(self) -> None:
        if self.ts.tzinfo is None:
            raise ValueError(f"Bar timestamp must be tz-aware, got naive: {self.ts}")
        if self.high < self.low:
            raise ValueError(f"Bar high {self.high} < low {self.low}")
        if not (self.low <= self.open <= self.high):
            raise ValueError(f"Bar open {self.open} outside [low={self.low}, high={self.high}]")
        if not (self.low <= self.close <= self.high):
            raise ValueError(f"Bar close {self.close} outside [low={self.low}, high={self.high}]")

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open


class BarSeries:
    """Cursor-bounded view over a sequence of bars.

    The engine calls `advance()` to move forward one bar. Strategy code reads
    via `current`, `at(i)`, `slice(start, end)`, `window(n)` — none of which
    may return data past the cursor.
    """

    def __init__(self, bars: Sequence[Bar]) -> None:
        if not bars:
            raise ValueError("BarSeries needs at least one bar")
        for prev, nxt in zip(bars, bars[1:]):
            if nxt.ts <= prev.ts:
                raise ValueError(f"Bars not strictly time-ordered: {prev.ts} -> {nxt.ts}")
        self._bars: list[Bar] = list(bars)
        self._cursor: int = -1  # advance() to 0 to start

    def __len__(self) -> int:
        return len(self._bars)

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def has_more(self) -> bool:
        return self._cursor + 1 < len(self._bars)

    def advance(self) -> Bar:
        if not self.has_more:
            raise IndexError("BarSeries exhausted")
        self._cursor += 1
        return self._bars[self._cursor]

    @property
    def current(self) -> Bar:
        if self._cursor < 0:
            raise IndexError("BarSeries not advanced yet")
        return self._bars[self._cursor]

    def at(self, i: int) -> Bar:
        """Index into past bars. Negative i counts back from cursor (-1 = current)."""
        if self._cursor < 0:
            raise IndexError("BarSeries not advanced yet")
        if i < 0:
            i = self._cursor + 1 + i  # -1 -> cursor
        if i < 0 or i > self._cursor:
            raise IndexError(f"Index {i} out of bounds [0, {self._cursor}]")
        return self._bars[i]

    def window(self, n: int) -> list[Bar]:
        """Last n bars up to and including current."""
        if self._cursor < 0:
            raise IndexError("BarSeries not advanced yet")
        if n <= 0:
            raise ValueError("window size must be positive")
        start = max(0, self._cursor + 1 - n)
        return self._bars[start : self._cursor + 1]

    def slice(self, start: int, end: int) -> list[Bar]:
        """Half-open [start, end). end must be <= cursor + 1."""
        if end > self._cursor + 1:
            raise IndexError(f"slice end {end} past cursor+1 {self._cursor + 1}")
        if start < 0 or start > end:
            raise IndexError(f"bad slice [{start}, {end})")
        return self._bars[start:end]


def bars_from_dataframe(df: pd.DataFrame, tz: str = "UTC") -> list[Bar]:
    """Build a list[Bar] from a DataFrame with columns ts/open/high/low/close/volume.

    ts may be a column or the index. If naive, it is localized to `tz`.
    """
    if "ts" in df.columns:
        ts = pd.to_datetime(df["ts"])
    else:
        ts = pd.to_datetime(df.index)
    if ts.dt.tz is None if hasattr(ts, "dt") else ts.tz is None:
        ts = ts.dt.tz_localize(tz) if hasattr(ts, "dt") else ts.tz_localize(tz)
    out: list[Bar] = []
    for i, row in enumerate(df.itertuples(index=False)):
        out.append(
            Bar(
                ts=ts.iloc[i],
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
            )
        )
    return out
