"""Shared test fixtures and helpers.

The build_bars helper lets tests construct hand-crafted bar sequences
without verbose Bar() boilerplate.
"""
from __future__ import annotations

from datetime import datetime
from typing import Iterable

import pandas as pd
import pytest

from strategy.bars import Bar, BarSeries


def make_bar(
    ts: str | pd.Timestamp,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float = 100.0,
    tz: str = "US/Eastern",
) -> Bar:
    """Build one bar. `ts` may be 'YYYY-MM-DD HH:MM' (in tz) or a Timestamp."""
    if isinstance(ts, str):
        t = pd.Timestamp(ts).tz_localize(tz).tz_convert("UTC")
    else:
        t = ts if ts.tzinfo else ts.tz_localize(tz).tz_convert("UTC")
    return Bar(ts=t, open=open_, high=high, low=low, close=close, volume=volume)


def build_series(rows: Iterable[tuple]) -> BarSeries:
    """rows: iterable of (ts, o, h, l, c[, v])."""
    bars = []
    for row in rows:
        if len(row) == 5:
            ts, o, h, l, c = row
            v = 100.0
        else:
            ts, o, h, l, c, v = row
        bars.append(make_bar(ts, o, h, l, c, v))
    return BarSeries(bars)


@pytest.fixture
def build():
    return build_series


@pytest.fixture
def mk():
    return make_bar
