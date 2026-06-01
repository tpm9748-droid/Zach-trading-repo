"""FVG + Order Block tests."""
from __future__ import annotations

import pandas as pd
import pytest

from strategy.bars import Bar
from strategy.fvg import (
    FVG,
    FVGDetector,
    FVGKind,
    FVGState,
    OrderBlock,
    find_order_block,
)


def bar(minute: int, o: float, h: float, l: float, c: float, v: float = 100.0) -> Bar:
    base = pd.Timestamp("2026-06-01 09:30").tz_localize("US/Eastern").tz_convert("UTC")
    return Bar(ts=base + pd.Timedelta(minutes=minute), open=o, high=h, low=l, close=c, volume=v)


def feed(d: FVGDetector, bars: list[Bar]):
    return [d.on_bar(b) for b in bars]


# ---------------------------------------------------------------------------
# FVG formation
# ---------------------------------------------------------------------------

def test_no_fvg_with_overlapping_wicks():
    d = FVGDetector()
    # 3 bars with overlapping ranges
    bars = [
        bar(0, 100, 105, 99, 103),
        bar(1, 103, 108, 102, 106),
        bar(2, 106, 110, 104, 108),  # c3.low (104) <= c1.high (105) -> no FVG
    ]
    results = feed(d, bars)
    assert all(r is None for r in results)
    assert d.all_fvgs == []


def test_bullish_fvg_formation():
    """Up-impulse where c3.low > c1.high leaves a bullish gap."""
    d = FVGDetector()
    bars = [
        bar(0, 100, 102, 99, 101),   # c1: high=102
        bar(1, 102, 108, 101, 107),  # c2: the impulse bar
        bar(2, 107, 110, 105, 109),  # c3: low=105 > c1.high=102 -> bullish FVG [102, 105]
    ]
    feed(d, bars)
    assert len(d.all_fvgs) == 1
    f = d.all_fvgs[0]
    assert f.kind == FVGKind.BULLISH
    assert f.lower == 102.0  # c1.high
    assert f.upper == 105.0  # c3.low
    assert f.created_at_idx == 2
    assert f.state == FVGState.ACTIVE


def test_bearish_fvg_formation():
    """Down-impulse where c3.high < c1.low leaves a bearish gap."""
    d = FVGDetector()
    bars = [
        bar(0, 110, 112, 108, 109),  # c1: low=108
        bar(1, 109, 110, 103, 104),  # c2: down impulse
        bar(2, 104, 106, 100, 102),  # c3: high=106 < c1.low=108 -> bearish FVG [106, 108]
    ]
    feed(d, bars)
    assert len(d.all_fvgs) == 1
    f = d.all_fvgs[0]
    assert f.kind == FVGKind.BEARISH
    assert f.lower == 106.0  # c3.high
    assert f.upper == 108.0  # c1.low


def test_fvg_only_fires_on_arrival_of_c3():
    d = FVGDetector()
    r1 = d.on_bar(bar(0, 100, 102, 99, 101))
    assert r1 is None
    r2 = d.on_bar(bar(1, 102, 108, 101, 107))
    assert r2 is None
    r3 = d.on_bar(bar(2, 107, 110, 105, 109))  # arrival of c3 -> new FVG
    assert r3 is not None and r3.kind == FVGKind.BULLISH


# ---------------------------------------------------------------------------
# FVG state transitions
# ---------------------------------------------------------------------------

def test_bullish_fvg_active_to_partial_on_touch():
    d = FVGDetector()
    # Form bullish FVG [102, 105]
    feed(d, [
        bar(0, 100, 102, 99, 101),
        bar(1, 102, 108, 101, 107),
        bar(2, 107, 110, 105, 109),
    ])
    f = d.all_fvgs[0]
    assert f.state == FVGState.ACTIVE

    # Next bar dips low=104 (inside the gap but not below it). Close=108 (above).
    d.on_bar(bar(3, 108, 109, 104, 108))
    assert f.state == FVGState.PARTIAL
    assert f.first_touched_at_idx == 3


def test_bullish_fvg_partial_to_filled_on_close_below():
    d = FVGDetector()
    feed(d, [
        bar(0, 100, 102, 99, 101),
        bar(1, 102, 108, 101, 107),
        bar(2, 107, 110, 105, 109),  # bullish FVG [102, 105]
    ])
    d.on_bar(bar(3, 108, 109, 104, 108))   # PARTIAL
    d.on_bar(bar(4, 108, 109, 100, 101))   # close 101 <= lower (102) -> FILLED
    f = d.all_fvgs[0]
    assert f.state == FVGState.FILLED
    assert f.filled_at_idx == 4


def test_bullish_fvg_can_fill_without_partial():
    """A bar that closes through without a prior touch jumps straight to FILLED."""
    d = FVGDetector()
    feed(d, [
        bar(0, 100, 102, 99, 101),
        bar(1, 102, 108, 101, 107),
        bar(2, 107, 110, 105, 109),  # bullish FVG [102, 105]
    ])
    d.on_bar(bar(3, 108, 109, 100, 101))  # straight close below lower
    f = d.all_fvgs[0]
    assert f.state == FVGState.FILLED


def test_bearish_fvg_state_transitions():
    d = FVGDetector()
    feed(d, [
        bar(0, 110, 112, 108, 109),  # c1.low=108
        bar(1, 109, 110, 103, 104),  # impulse
        bar(2, 104, 106, 100, 102),  # bearish FVG [106, 108]
    ])
    d.on_bar(bar(3, 102, 107, 102, 105))  # high=107 touches the gap
    f = d.all_fvgs[0]
    assert f.state == FVGState.PARTIAL

    d.on_bar(bar(4, 105, 110, 105, 109))  # close 109 >= upper(108) -> FILLED
    assert f.state == FVGState.FILLED


def test_filled_fvg_stays_filled():
    d = FVGDetector()
    feed(d, [
        bar(0, 100, 102, 99, 101),
        bar(1, 102, 108, 101, 107),
        bar(2, 107, 110, 105, 109),
    ])
    d.on_bar(bar(3, 108, 109, 100, 101))  # FILLED
    d.on_bar(bar(4, 101, 110, 101, 109))  # price recovers way above
    assert d.all_fvgs[0].state == FVGState.FILLED


# ---------------------------------------------------------------------------
# Multiple FVGs
# ---------------------------------------------------------------------------

def test_multiple_fvgs_tracked_independently():
    d = FVGDetector()
    # FVG #1 (bullish)
    feed(d, [
        bar(0, 100, 102, 99, 101),
        bar(1, 102, 108, 101, 107),
        bar(2, 107, 110, 105, 109),  # bullish FVG [102, 105]
    ])
    # Some filler bars that don't make a new FVG
    d.on_bar(bar(3, 109, 111, 108, 110))
    d.on_bar(bar(4, 110, 112, 109, 111))
    # FVG #2 (bullish, higher up)
    d.on_bar(bar(5, 111, 113, 110, 112))   # this is c1 of next FVG
    d.on_bar(bar(6, 112, 120, 111, 119))   # c2 impulse
    d.on_bar(bar(7, 119, 122, 116, 121))   # c3.low=116 > c1.high=113 -> FVG [113, 116]
    assert len(d.all_fvgs) == 2

    # First should still be ACTIVE (or PARTIAL if touched). Second is brand new.
    assert d.all_fvgs[0].lower == 102
    assert d.all_fvgs[1].lower == 113
    assert d.all_fvgs[1].state == FVGState.ACTIVE


def test_find_in_range_filters_correctly():
    d = FVGDetector()
    # Bullish FVG at idx 2
    feed(d, [
        bar(0, 100, 102, 99, 101),
        bar(1, 102, 108, 101, 107),
        bar(2, 107, 110, 105, 109),
    ])
    # Some bars
    d.on_bar(bar(3, 109, 110, 108, 109))
    d.on_bar(bar(4, 109, 110, 108, 109))
    # Bearish FVG at idx 7
    d.on_bar(bar(5, 109, 111, 108, 110))
    d.on_bar(bar(6, 110, 111, 102, 103))   # c2 down impulse
    d.on_bar(bar(7, 103, 105, 99, 100))    # c3.high=105 < c1.low=108? c1 here is bar 5 with low=108. Yes.

    in_range = d.find_in_range(0, 4)
    assert len(in_range) == 1
    assert in_range[0].kind == FVGKind.BULLISH

    in_range = d.find_in_range(5, 10, kind=FVGKind.BEARISH)
    assert len(in_range) == 1
    assert in_range[0].created_at_idx == 7


# ---------------------------------------------------------------------------
# Order block
# ---------------------------------------------------------------------------

def test_ob_bullish_displacement_finds_last_bearish_candle():
    """Bullish displacement starts at idx 5. Last bearish candle before that is idx 3."""
    bars = [
        bar(0, 100, 101, 99, 100.5),   # bullish
        bar(1, 100.5, 102, 100, 101.5),  # bullish
        bar(2, 101.5, 103, 101, 102.5),  # bullish
        bar(3, 102.5, 103, 100, 101),    # BEARISH <- this should be the OB
        bar(4, 101, 102, 100.5, 101.5),  # bullish (small)
        bar(5, 101.5, 106, 101, 105.5),  # bullish IMPULSE (displacement start)
    ]
    ob = find_order_block(bars, displacement_start_idx=5, displacement_direction=FVGKind.BULLISH)
    assert ob is not None
    assert ob.bar_idx == 3
    assert ob.upper == 103
    assert ob.lower == 100


def test_ob_bearish_displacement_finds_last_bullish_candle():
    bars = [
        bar(0, 100, 102, 99, 101),       # bullish
        bar(1, 101, 101.5, 99, 100),     # bearish
        bar(2, 100, 100.5, 98, 99),      # bearish
        bar(3, 99, 102, 99, 101),        # BULLISH <- the OB
        bar(4, 101, 101.5, 100, 100.5),  # bearish (small)
        bar(5, 100.5, 100.5, 95, 95.5),  # bearish IMPULSE
    ]
    ob = find_order_block(bars, displacement_start_idx=5, displacement_direction=FVGKind.BEARISH)
    assert ob is not None
    assert ob.bar_idx == 3
    assert ob.upper == 102
    assert ob.lower == 99


def test_ob_skips_doji_bars():
    """Bars with open == close are neither bullish nor bearish; OB search skips them."""
    bars = [
        bar(0, 100, 102, 99, 100),       # doji
        bar(1, 100, 101, 99, 99),        # BEARISH
        bar(2, 99, 100, 98, 99),         # doji
        bar(3, 99, 105, 98, 104),        # bullish impulse
    ]
    ob = find_order_block(bars, displacement_start_idx=3, displacement_direction=FVGKind.BULLISH)
    assert ob is not None
    assert ob.bar_idx == 1


def test_ob_returns_none_if_no_opposing_candle_found():
    """All preceding bars same polarity as displacement -> no OB."""
    bars = [
        bar(0, 100, 101, 99.5, 100.5),
        bar(1, 100.5, 102, 100, 101.5),
        bar(2, 101.5, 103, 101, 102.5),
        bar(3, 102.5, 108, 102, 107),  # displacement
    ]
    ob = find_order_block(bars, displacement_start_idx=3, displacement_direction=FVGKind.BULLISH)
    assert ob is None


def test_ob_respects_max_lookback():
    """Even if a valid OB exists further back, max_lookback caps the search."""
    bars = [
        bar(0, 101, 102, 99, 100),  # bearish — too far back
    ] + [bar(i, 100, 101, 100, 100.5) for i in range(1, 20)]  # 19 bullish/dojis
    bars.append(bar(20, 100.5, 106, 100, 105.5))  # bullish displacement
    ob = find_order_block(
        bars, displacement_start_idx=20,
        displacement_direction=FVGKind.BULLISH,
        max_lookback=5,
    )
    assert ob is None
