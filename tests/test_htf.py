"""HTF aggregation and trend tests."""
from __future__ import annotations

import pandas as pd
import pytest

from strategy.bars import Bar
from strategy.htf import (
    HTFTrend,
    TimeframeAggregator,
    TrendDirection,
    daily_bucket_start,
    four_hour_bucket_start,
)


def bar(ts: str, o: float, h: float, l: float, c: float, v: float = 100.0) -> Bar:
    t = pd.Timestamp(ts).tz_localize("US/Eastern").tz_convert("UTC")
    return Bar(ts=t, open=o, high=h, low=l, close=c, volume=v)


# ---------------------------------------------------------------------------
# Bucket functions
# ---------------------------------------------------------------------------


def test_daily_bucket_anchors_at_session_18et():
    """All bars on the same session day map to the same 18:00 ET start."""
    b1 = pd.Timestamp("2026-06-01 19:00").tz_localize("US/Eastern").tz_convert("UTC")
    b2 = pd.Timestamp("2026-06-02 09:30").tz_localize("US/Eastern").tz_convert("UTC")
    b3 = pd.Timestamp("2026-06-02 16:00").tz_localize("US/Eastern").tz_convert("UTC")
    assert daily_bucket_start(b1) == daily_bucket_start(b2) == daily_bucket_start(b3)
    # New session day starts at the next 18:00 ET
    b4 = pd.Timestamp("2026-06-02 18:00").tz_localize("US/Eastern").tz_convert("UTC")
    assert daily_bucket_start(b4) != daily_bucket_start(b1)


def test_four_hour_buckets_align_to_18et():
    """4h buckets per session start at 18, 22, 02, 06, 10, 14 ET."""
    starts = []
    for hm in ["18:30", "20:00", "22:00", "01:30", "02:00", "05:00", "06:00", "09:30", "10:00", "13:00", "14:00", "16:30"]:
        # Crossing midnight: hours before 18:00 ET belong to the prior session day
        day = "2026-06-01" if hm >= "18:00" else "2026-06-02"
        ts = pd.Timestamp(f"{day} {hm}").tz_localize("US/Eastern").tz_convert("UTC")
        starts.append(four_hour_bucket_start(ts))
    # We expect 6 distinct buckets (one per 4h window). Group consecutive equal starts.
    unique_in_order = [starts[0]]
    for s in starts[1:]:
        if s != unique_in_order[-1]:
            unique_in_order.append(s)
    assert len(unique_in_order) == 6


# ---------------------------------------------------------------------------
# TimeframeAggregator
# ---------------------------------------------------------------------------


def test_aggregator_emits_nothing_for_first_bar():
    agg = TimeframeAggregator(daily_bucket_start)
    assert agg.on_bar(bar("2026-06-01 19:00", 100, 105, 99, 102)) is None


def test_aggregator_closes_bucket_on_boundary_cross():
    agg = TimeframeAggregator(daily_bucket_start)
    agg.on_bar(bar("2026-06-01 19:00", 100, 105, 99, 102))
    agg.on_bar(bar("2026-06-02 10:00", 102, 110, 101, 108))
    # Cross into a new session day
    closed = agg.on_bar(bar("2026-06-02 18:00", 108, 115, 107, 112))
    assert closed is not None
    assert closed.open == 100   # first bar's open
    assert closed.high == 110   # max across 2 bars
    assert closed.low == 99
    assert closed.close == 108  # last bar's close in the closed bucket
    assert closed.volume == 200  # sum


def test_aggregator_flush_finalizes_open_bucket():
    agg = TimeframeAggregator(daily_bucket_start)
    agg.on_bar(bar("2026-06-01 19:00", 100, 105, 99, 102))
    agg.on_bar(bar("2026-06-02 10:00", 102, 110, 101, 108))
    closed = agg.flush()
    assert closed is not None
    assert closed.open == 100
    assert closed.high == 110
    # Second flush returns None
    assert agg.flush() is None


# ---------------------------------------------------------------------------
# HTFTrend
# ---------------------------------------------------------------------------


def test_htf_unknown_period_raises():
    with pytest.raises(ValueError):
        HTFTrend(period="weekly")


def test_htf_trend_undefined_with_too_few_swings():
    """Need at least 2 highs + 2 lows on the HTF to classify."""
    t = HTFTrend(period="daily", swing_n=2)
    t.on_bar(bar("2026-06-01 19:00", 100, 101, 99, 100))
    assert t.current_trend() == TrendDirection.UNDEFINED


def _feed_daily_prices(t: HTFTrend, day_prices: list[tuple[str, float, float, float, float]]) -> None:
    """day_prices: list of (session_day_label, open, high, low, close).
    Each entry is one *new* session day (forces a bucket close)."""
    for i, (label, o, h, l, c) in enumerate(day_prices):
        t.on_bar(bar(f"{label} 19:00", o, h, l, c, v=100))


def test_htf_trend_bullish_with_HH_HL():
    t = HTFTrend(period="daily", swing_n=2)
    # 12 daily bars with strict-zigzag highs/lows produces swings at pos 2, 4, 6, 8
    # under N=2:  HIGH 110, LOW 80, HIGH 120 (HH), LOW 92 (HL) -> BULLISH.
    days = [
        ("2026-06-01",  94,  98,  88,  95),   # 0
        ("2026-06-02",  95,  99,  89,  96),   # 1
        ("2026-06-03", 105, 110, 100, 108),   # 2  HIGH 1 (110)
        ("2026-06-04", 100, 105,  95,  98),   # 3
        ("2026-06-05",  95, 100,  80,  90),   # 4  LOW 1 (80)
        ("2026-06-06",  95, 108,  90,  100),  # 5
        ("2026-06-07", 110, 120, 105, 115),   # 6  HIGH 2 (120) — HH vs 110
        ("2026-06-08", 110, 115, 100, 105),   # 7
        ("2026-06-09", 100, 112,  92, 100),   # 8  LOW 2 (92) — HL vs 80
        ("2026-06-10", 105, 118, 100, 115),   # 9
        ("2026-06-11", 110, 122, 105, 120),   # 10
        ("2026-06-12", 115, 125, 110, 122),   # 11
    ]
    _feed_daily_prices(t, days)
    # Add one extra bar to close the 12th bucket
    t.on_bar(bar("2026-06-13 19:00", 122, 124, 121, 123))
    assert t.current_trend() == TrendDirection.BULLISH


def test_htf_trend_bearish_with_LH_LL():
    t = HTFTrend(period="daily", swing_n=2)
    # Mirror of bullish: LOW 1 at 90, HIGH 1 at 120, LOW 2 at 80 (LL), HIGH 2 at 108 (LH)
    days = [
        ("2026-06-01", 122, 125, 115, 120),   # 0
        ("2026-06-02", 120, 122, 110, 115),   # 1
        ("2026-06-03", 105, 110,  90, 100),   # 2  LOW 1 (90)
        ("2026-06-04", 110, 115, 100, 112),   # 3
        ("2026-06-05", 115, 120, 105, 110),   # 4  HIGH 1 (120)
        ("2026-06-06", 105, 110,  95, 100),   # 5
        ("2026-06-07",  95, 100,  80,  90),   # 6  LOW 2 (80) — LL vs 90
        ("2026-06-08",  95, 105,  90, 100),   # 7
        ("2026-06-09", 100, 108,  92,  98),   # 8  HIGH 2 (108) — LH vs 120
        ("2026-06-10",  95, 100,  85,  92),   # 9
        ("2026-06-11",  90,  95,  78,  85),   # 10
        ("2026-06-12",  85,  90,  75,  80),   # 11
    ]
    _feed_daily_prices(t, days)
    t.on_bar(bar("2026-06-13 19:00", 80, 82, 78, 80))
    assert t.current_trend() == TrendDirection.BEARISH


def test_htf_trend_undefined_when_mixed_structure():
    """HH paired with LL (or LH with HL) -> undefined."""
    t = HTFTrend(period="daily", swing_n=2)
    # HIGH 1 = 110, LOW 1 = 90, HIGH 2 = 120 (HH), LOW 2 = 80 (LL) -> mixed
    days = [
        ("2026-06-01", 100, 105,  95, 100),
        ("2026-06-02", 100, 108,  98, 102),
        ("2026-06-03", 105, 110, 100, 108),   # 2  HIGH 1 = 110
        ("2026-06-04", 108, 109,  95, 100),
        ("2026-06-05",  95, 100,  90,  95),   # 4  LOW 1 = 90
        ("2026-06-06",  95, 108,  92, 100),
        ("2026-06-07", 105, 120, 100, 115),   # 6  HIGH 2 = 120 (HH)
        ("2026-06-08", 110, 115,  95, 100),
        ("2026-06-09",  95, 100,  80,  90),   # 8  LOW 2 = 80 (LL) -> mixed
        ("2026-06-10",  90,  95,  85,  92),
        ("2026-06-11",  92, 100,  88,  95),
        ("2026-06-12",  95, 105,  92, 100),
    ]
    _feed_daily_prices(t, days)
    t.on_bar(bar("2026-06-13 19:00", 100, 102, 98, 100))
    assert t.current_trend() == TrendDirection.UNDEFINED
