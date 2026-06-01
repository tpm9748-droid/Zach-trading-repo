"""Engine smoke tests.

These verify the engine wires its sub-modules together correctly. Full
strategy behavior on real-shaped data is the end-to-end smoke test in
test_smoke.py (module 8).
"""
from __future__ import annotations

import pandas as pd
import pytest

from backtest.engine import run_backtest
from strategy.bars import Bar
from strategy.params import DEFAULT_PARAMS


def bar(minute: int, o: float, h: float, l: float, c: float, v: float = 100.0) -> Bar:
    base = pd.Timestamp("2026-06-01 18:00").tz_localize("US/Eastern").tz_convert("UTC")
    return Bar(ts=base + pd.Timedelta(minutes=minute), open=o, high=h, low=l, close=c, volume=v)


def test_empty_bar_list_returns_empty_result():
    result = run_backtest([], DEFAULT_PARAMS)
    assert result.trades == []
    assert result.bar_count == 0
    assert result.metrics.overall.n_trades == 0


def test_engine_advances_through_all_bars_without_signals():
    """Flat bars produce no sweeps and no trades."""
    bars = [bar(i, 100, 100.5, 99.5, 100, 100) for i in range(50)]
    result = run_backtest(bars, DEFAULT_PARAMS)
    assert result.bar_count == 50
    assert result.trades == []


def test_engine_returns_sweep_events_when_collected():
    """collect_sweep_events flag retains intermediate sweep events even when
    no trade fires."""
    bars = [bar(i, 100, 100.5, 99.5, 100, 100) for i in range(50)]
    result = run_backtest(bars, DEFAULT_PARAMS, collect_sweep_events=True)
    # No sweeps happen on flat bars
    assert result.sweep_events == []


def test_engine_accepts_bar_series_or_iterable():
    """run_backtest works with a list or iterator."""
    bars = [bar(i, 100, 100.5, 99.5, 100, 100) for i in range(10)]
    r1 = run_backtest(bars, DEFAULT_PARAMS)
    r2 = run_backtest(iter(bars), DEFAULT_PARAMS)
    assert r1.bar_count == r2.bar_count == 10
