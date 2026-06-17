"""Trades -> 1m OHLCV aggregation tests."""
from __future__ import annotations

import pandas as pd

from backtest.trades_loader import aggregate_trades_to_1m_bars


def _df(rows):
    return pd.DataFrame(rows, columns=["ts_event", "action", "price", "size", "symbol"])


def test_aggregates_ohlcv_within_a_minute():
    df = _df([
        ("2025-12-01T00:00:01.000000000Z", "T", 100.0, 1, "NQH6"),
        ("2025-12-01T00:00:30.000000000Z", "T", 102.0, 2, "NQH6"),  # high
        ("2025-12-01T00:00:45.000000000Z", "T",  99.0, 3, "NQH6"),  # low
        ("2025-12-01T00:00:59.000000000Z", "T", 101.0, 4, "NQH6"),  # close
    ])
    bars = aggregate_trades_to_1m_bars(df, "NQH6")
    assert len(bars) == 1
    b = bars[0]
    assert (b.open, b.high, b.low, b.close) == (100.0, 102.0, 99.0, 101.0)
    assert b.volume == 10  # 1+2+3+4
    assert str(b.ts) == "2025-12-01 00:00:00+00:00"  # floored to the minute


def test_splits_into_separate_minute_bars():
    df = _df([
        ("2025-12-01T00:00:10.000000000Z", "T", 100.0, 1, "NQH6"),
        ("2025-12-01T00:01:10.000000000Z", "T", 105.0, 1, "NQH6"),
    ])
    bars = aggregate_trades_to_1m_bars(df, "NQH6")
    assert len(bars) == 2
    assert bars[0].close == 100.0
    assert bars[1].open == 105.0


def test_filters_by_symbol_and_action():
    df = _df([
        ("2025-12-01T00:00:10.000000000Z", "T", 100.0, 5, "NQH6"),
        ("2025-12-01T00:00:20.000000000Z", "T", 200.0, 9, "NQM6"),  # other contract
        ("2025-12-01T00:00:30.000000000Z", "A", 100.0, 7, "NQH6"),  # not a trade
    ])
    bars = aggregate_trades_to_1m_bars(df, "NQH6")
    assert len(bars) == 1
    assert bars[0].volume == 5  # only the single NQH6 trade
    assert bars[0].high == 100.0


def test_empty_when_symbol_absent():
    df = _df([("2025-12-01T00:00:10.000000000Z", "T", 100.0, 1, "NQH6")])
    assert aggregate_trades_to_1m_bars(df, "NQM6") == []
