from __future__ import annotations

import pandas as pd
import pytest

from strategy.bars import Bar, BarSeries


def test_bar_requires_tz(mk):
    with pytest.raises(ValueError, match="tz-aware"):
        Bar(ts=pd.Timestamp("2026-01-01 09:30"), open=1, high=2, low=0, close=1, volume=1)


def test_bar_rejects_invalid_ohlc():
    ts = pd.Timestamp("2026-01-01 09:30", tz="UTC")
    with pytest.raises(ValueError, match="high.*low"):
        Bar(ts=ts, open=1, high=0, low=2, close=1, volume=1)
    with pytest.raises(ValueError, match="open"):
        Bar(ts=ts, open=10, high=2, low=0, close=1, volume=1)
    with pytest.raises(ValueError, match="close"):
        Bar(ts=ts, open=1, high=2, low=0, close=10, volume=1)


def test_bar_polarity(mk):
    bull = mk("2026-01-01 09:30", 100, 110, 99, 108)
    bear = mk("2026-01-01 09:31", 108, 109, 100, 101)
    doji = mk("2026-01-01 09:32", 101, 105, 99, 101)
    assert bull.is_bullish and not bull.is_bearish
    assert bear.is_bearish and not bear.is_bullish
    assert not doji.is_bullish and not doji.is_bearish


def test_barseries_requires_strict_ordering(mk):
    a = mk("2026-01-01 09:30", 1, 2, 0, 1)
    b = mk("2026-01-01 09:30", 1, 2, 0, 1)  # duplicate ts
    with pytest.raises(ValueError, match="time-ordered"):
        BarSeries([a, b])


def test_barseries_cursor_lifecycle(build):
    s = build([
        ("2026-01-01 09:30", 1, 2, 0, 1),
        ("2026-01-01 09:31", 1, 2, 0, 1),
        ("2026-01-01 09:32", 1, 2, 0, 1),
    ])
    assert s.cursor == -1
    assert s.has_more

    with pytest.raises(IndexError):
        _ = s.current

    s.advance()
    assert s.cursor == 0
    s.advance()
    s.advance()
    assert s.cursor == 2
    assert not s.has_more
    with pytest.raises(IndexError):
        s.advance()


def test_barseries_lookahead_protection(build):
    s = build([
        ("2026-01-01 09:30", 1, 2, 0, 1),
        ("2026-01-01 09:31", 3, 4, 2, 3),
        ("2026-01-01 09:32", 5, 6, 4, 5),
    ])
    s.advance()  # cursor=0
    with pytest.raises(IndexError):
        s.at(1)  # future bar
    with pytest.raises(IndexError):
        s.slice(0, 2)  # ends past cursor+1

    s.advance()  # cursor=1
    assert s.at(0).open == 1
    assert s.at(1).open == 3
    assert s.at(-1).open == 3  # negative = relative to cursor
    assert s.at(-2).open == 1


def test_barseries_window(build):
    s = build([
        ("2026-01-01 09:30", 1, 2, 0, 1),
        ("2026-01-01 09:31", 2, 3, 1, 2),
        ("2026-01-01 09:32", 3, 4, 2, 3),
        ("2026-01-01 09:33", 4, 5, 3, 4),
    ])
    s.advance(); s.advance(); s.advance()  # cursor=2
    win = s.window(2)
    assert [b.open for b in win] == [2, 3]
    # window larger than available returns what's there
    win5 = s.window(5)
    assert [b.open for b in win5] == [1, 2, 3]
