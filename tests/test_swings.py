"""Swing detector + CHoCH break tests.

Each test builds a hand-crafted bar sequence with a known swing structure
and verifies the detector finds (or correctly doesn't find) it.
"""
from __future__ import annotations

import pandas as pd
import pytest

from strategy.bars import Bar
from strategy.swings import (
    BreakMode,
    SwingDetector,
    SwingKind,
    SwingPoint,
    broke_swing_down,
    broke_swing_up,
)


def et(ts: str) -> pd.Timestamp:
    return pd.Timestamp(ts).tz_localize("US/Eastern").tz_convert("UTC")


def bar(minute: int, o: float, h: float, l: float, c: float, v: float = 100.0) -> Bar:
    """Bar at 2026-06-01 09:30 + minute minutes."""
    base = pd.Timestamp("2026-06-01 09:30").tz_localize("US/Eastern").tz_convert("UTC")
    return Bar(ts=base + pd.Timedelta(minutes=minute), open=o, high=h, low=l, close=c, volume=v)


def feed(detector: SwingDetector, bars: list[Bar]) -> list:
    """Feed bars and return list of (bar_idx_when_detected, swing_or_None)."""
    out = []
    for i, b in enumerate(bars):
        sp = detector.on_bar(b)
        out.append((i, sp))
    return out


# ---------------------------------------------------------------------------
# SwingDetector basics
# ---------------------------------------------------------------------------

def test_rejects_n_below_1():
    with pytest.raises(ValueError):
        SwingDetector(n=0)


def test_no_swing_until_enough_bars():
    """N=3 requires 2*3+1=7 bars before any swing can be confirmed."""
    d = SwingDetector(n=3)
    bars = [bar(i, 100, 110, 90, 100) for i in range(6)]  # 6 bars: nothing
    results = feed(d, bars)
    assert all(sp is None for _, sp in results)
    assert d.all_swings == []


def test_detects_swing_high_with_n3():
    """7 bars, the middle one has the highest high → swing high confirmed
    when the 7th bar arrives."""
    d = SwingDetector(n=3)
    # bars: highs 100, 101, 102, 110, 103, 102, 101
    #                                ^candidate idx 3
    highs = [100, 101, 102, 110, 103, 102, 101]
    bars = [bar(i, 99, h, 90, h - 0.5) for i, h in enumerate(highs)]
    results = feed(d, bars)
    # Only the 7th bar (index 6) should produce a swing
    confirmations = [(i, sp) for i, sp in results if sp is not None]
    assert len(confirmations) == 1
    i, sp = confirmations[0]
    assert i == 6  # confirmed at bar index 6
    assert sp.bar_idx == 3  # candidate is at index 3
    assert sp.confirmed_at_idx == 6
    assert sp.kind == SwingKind.HIGH
    assert sp.price == 110


def test_detects_swing_low_with_n3():
    d = SwingDetector(n=3)
    lows = [100, 99, 98, 90, 97, 98, 99]
    bars = [bar(i, l + 5, l + 10, l, l + 5) for i, l in enumerate(lows)]
    results = feed(d, bars)
    confirmations = [(i, sp) for i, sp in results if sp is not None]
    assert len(confirmations) == 1
    _, sp = confirmations[0]
    assert sp.kind == SwingKind.LOW
    assert sp.bar_idx == 3
    assert sp.price == 90


def test_strict_greater_rejects_ties():
    """If candidate ties with a neighbor it's NOT a swing under strict rule."""
    d = SwingDetector(n=3)
    # Candidate high == one of the left neighbors → not strictly greater
    highs = [100, 101, 102, 110, 110, 102, 101]  # idx 4 ties with idx 3
    bars = [bar(i, 99, h, 90, h - 0.5) for i, h in enumerate(highs)]
    results = feed(d, bars)
    assert all(sp is None for _, sp in results)


def test_detects_multiple_swings_in_sequence():
    """Build a series with a swing high then a swing low."""
    d = SwingDetector(n=2)
    # N=2 needs 5-bar windows. Construct:
    #   idx:    0  1  2  3  4  5  6  7  8
    #   high: 100 101 110 101 100 99  98  91 90  ← swing high at idx 2, swing low at idx 7?
    # Let's design carefully:
    # Want a swing HIGH at idx 2 (need idx 0,1 < 2.high and idx 3,4 < 2.high)
    # Want a swing LOW at idx 6 (need idx 4,5 > 6.low and idx 7,8 > 6.low)
    highs = [100, 101, 110, 101, 100, 99, 95, 99, 100]
    lows  = [ 90,  91,  92,  91,  90, 89, 80, 89, 90]
    bars = [bar(i, (h + l) / 2, h, l, (h + l) / 2) for i, (h, l) in enumerate(zip(highs, lows))]
    feed(d, bars)
    assert len(d.highs) == 1
    assert d.highs[0].bar_idx == 2
    assert d.highs[0].price == 110
    assert len(d.lows) == 1
    assert d.lows[0].bar_idx == 6
    assert d.lows[0].price == 80


def test_confirmation_idx_matches_n():
    d = SwingDetector(n=3)
    highs = [100, 101, 102, 110, 103, 102, 101]
    bars = [bar(i, 99, h, 90, h - 0.5) for i, h in enumerate(highs)]
    feed(d, bars)
    sp = d.highs[0]
    assert sp.confirmed_at_idx - sp.bar_idx == 3


# ---------------------------------------------------------------------------
# most_recent queries
# ---------------------------------------------------------------------------

def test_most_recent_respects_confirmation_visibility():
    """A swing only becomes visible when its confirmation bar arrives."""
    d = SwingDetector(n=3)
    highs = [100, 101, 102, 110, 103, 102, 101]
    bars = [bar(i, 99, h, 90, h - 0.5) for i, h in enumerate(highs)]
    feed(d, bars)
    sp = d.highs[0]
    # Swing bar is idx 3, confirmed at idx 6.
    # At "engine cursor 5" the swing isn't visible yet:
    assert d.most_recent(SwingKind.HIGH, before_confirmation_idx=5) is None
    # At "engine cursor 6" it is:
    assert d.most_recent(SwingKind.HIGH, before_confirmation_idx=6) == sp


def test_most_recent_returns_latest():
    """When two swings of the same kind exist, return the more recent."""
    d = SwingDetector(n=2)
    # Two swing highs
    highs = [100, 101, 110, 101, 100, 102, 115, 102, 101]
    lows  = [ 90,  91,  92,  91,  90,  91,  92,  91,  90]
    bars = [bar(i, (h + l) / 2, h, l, (h + l) / 2) for i, (h, l) in enumerate(zip(highs, lows))]
    feed(d, bars)
    assert len(d.highs) == 2
    latest = d.most_recent(SwingKind.HIGH)
    assert latest.price == 115


def test_most_recent_before_bar_idx():
    d = SwingDetector(n=2)
    highs = [100, 101, 110, 101, 100, 102, 115, 102, 101]
    lows  = [ 90,  91,  92,  91,  90,  91,  92,  91,  90]
    bars = [bar(i, (h + l) / 2, h, l, (h + l) / 2) for i, (h, l) in enumerate(zip(highs, lows))]
    feed(d, bars)
    # Restrict to swings before bar idx 5 → only the first one (at idx 2)
    sp = d.most_recent(SwingKind.HIGH, before_bar_idx=5)
    assert sp.price == 110


# ---------------------------------------------------------------------------
# Break helpers (CHoCH)
# ---------------------------------------------------------------------------

def _swing(kind: SwingKind, price: float) -> SwingPoint:
    return SwingPoint(
        bar_idx=0, confirmed_at_idx=0,
        ts=et("2026-06-01 09:30"), price=price, kind=kind,
    )


def test_break_swing_down_close_mode():
    sw = _swing(SwingKind.LOW, 100.0)
    # Close below → break
    b = bar(10, 102, 103, 99, 99.5)
    assert broke_swing_down(b, sw, BreakMode.CLOSE)
    # Wick below but close above → no break (close mode)
    b2 = bar(11, 102, 103, 99, 100.5)
    assert not broke_swing_down(b2, sw, BreakMode.CLOSE)


def test_break_swing_down_wick_mode():
    sw = _swing(SwingKind.LOW, 100.0)
    b = bar(10, 102, 103, 99, 100.5)  # wick below, close above
    assert broke_swing_down(b, sw, BreakMode.WICK)
    b2 = bar(11, 102, 103, 100.5, 101)  # wick at 100.5 — not below 100
    assert not broke_swing_down(b2, sw, BreakMode.WICK)


def test_break_swing_up_close_mode():
    sw = _swing(SwingKind.HIGH, 100.0)
    b = bar(10, 99, 102, 98, 101)  # close above → break
    assert broke_swing_up(b, sw, BreakMode.CLOSE)
    b2 = bar(11, 99, 102, 98, 99.5)  # wick above but close below → no break
    assert not broke_swing_up(b2, sw, BreakMode.CLOSE)


def test_break_swing_up_wick_mode():
    sw = _swing(SwingKind.HIGH, 100.0)
    b = bar(10, 99, 102, 98, 99.5)  # wick above, close below
    assert broke_swing_up(b, sw, BreakMode.WICK)


def test_break_helpers_validate_swing_kind():
    high_sw = _swing(SwingKind.HIGH, 100.0)
    low_sw = _swing(SwingKind.LOW, 100.0)
    b = bar(10, 100, 101, 99, 100)
    with pytest.raises(ValueError):
        broke_swing_down(b, high_sw)
    with pytest.raises(ValueError):
        broke_swing_up(b, low_sw)
