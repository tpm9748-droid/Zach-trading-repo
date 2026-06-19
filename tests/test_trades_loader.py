"""Trades -> 1m OHLCV(+delta) aggregation tests."""
from __future__ import annotations

import pandas as pd

from backtest.trades_loader import aggregate_trades_to_1m_bars


def _df(rows):
    # rows: (ts_event, action, side, price, size, symbol)
    return pd.DataFrame(rows, columns=["ts_event", "action", "side", "price", "size", "symbol"])


def test_aggregates_ohlcv_within_a_minute():
    df = _df([
        ("2025-12-01T00:00:01.000000000Z", "T", "B", 100.0, 1, "NQH6"),
        ("2025-12-01T00:00:30.000000000Z", "T", "B", 102.0, 2, "NQH6"),  # high
        ("2025-12-01T00:00:45.000000000Z", "T", "A",  99.0, 3, "NQH6"),  # low
        ("2025-12-01T00:00:59.000000000Z", "T", "B", 101.0, 4, "NQH6"),  # close
    ])
    bars = aggregate_trades_to_1m_bars(df, "NQH6")
    assert len(bars) == 1
    b = bars[0]
    assert (b.open, b.high, b.low, b.close) == (100.0, 102.0, 99.0, 101.0)
    assert b.volume == 10  # 1+2+3+4
    assert str(b.ts) == "2025-12-01 00:00:00+00:00"


def test_delta_is_signed_by_aggressor_side():
    # Buys +size, sells -size, none 0.
    df = _df([
        ("2025-12-01T00:00:01.000000000Z", "T", "B", 100.0, 5, "NQH6"),
        ("2025-12-01T00:00:02.000000000Z", "T", "A", 100.0, 2, "NQH6"),
        ("2025-12-01T00:00:03.000000000Z", "T", "N", 100.0, 7, "NQH6"),  # no side
    ])
    b = aggregate_trades_to_1m_bars(df, "NQH6")[0]
    assert b.volume == 14      # 5+2+7
    assert b.delta == 3        # +5 -2 +0


def test_splits_into_separate_minute_bars():
    df = _df([
        ("2025-12-01T00:00:10.000000000Z", "T", "B", 100.0, 1, "NQH6"),
        ("2025-12-01T00:01:10.000000000Z", "T", "A", 105.0, 1, "NQH6"),
    ])
    bars = aggregate_trades_to_1m_bars(df, "NQH6")
    assert len(bars) == 2
    assert bars[0].close == 100.0 and bars[0].delta == 1
    assert bars[1].open == 105.0 and bars[1].delta == -1


def test_filters_by_symbol_and_action():
    df = _df([
        ("2025-12-01T00:00:10.000000000Z", "T", "B", 100.0, 5, "NQH6"),
        ("2025-12-01T00:00:20.000000000Z", "T", "B", 200.0, 9, "NQM6"),  # other contract
        ("2025-12-01T00:00:30.000000000Z", "A", "B", 100.0, 7, "NQH6"),  # not a trade
    ])
    bars = aggregate_trades_to_1m_bars(df, "NQH6")
    assert len(bars) == 1
    assert bars[0].volume == 5
    assert bars[0].high == 100.0


def test_empty_when_symbol_absent():
    df = _df([("2025-12-01T00:00:10.000000000Z", "T", "B", 100.0, 1, "NQH6")])
    assert aggregate_trades_to_1m_bars(df, "NQM6") == []
