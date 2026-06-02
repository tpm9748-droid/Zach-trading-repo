"""Session VWAP tests."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from strategy.bars import Bar
from strategy.vwap import SessionVWAP


def bar(ts: str, o: float, h: float, l: float, c: float, v: float) -> Bar:
    t = pd.Timestamp(ts).tz_localize("US/Eastern").tz_convert("UTC")
    return Bar(ts=t, open=o, high=h, low=l, close=c, volume=v)


def test_vwap_none_before_any_bars():
    v = SessionVWAP()
    assert v.current is None


def test_vwap_single_bar_equals_typical_price():
    v = SessionVWAP()
    v.on_bar(bar("2026-06-01 18:00", 100, 102, 98, 100, 100))
    # typical = (102 + 98 + 100) / 3 = 100
    assert v.current == pytest.approx(100.0)


def test_vwap_volume_weighted_across_bars():
    v = SessionVWAP()
    # First bar: typical 100, volume 100  -> contributes 100*100 = 10000
    v.on_bar(bar("2026-06-01 18:00", 100, 102, 98, 100, 100))
    # Second bar: typical 110, volume 300 -> contributes 110*300 = 33000
    v.on_bar(bar("2026-06-01 18:01", 110, 112, 108, 110, 300))
    # VWAP = (10000 + 33000) / (100 + 300) = 43000 / 400 = 107.5
    assert v.current == pytest.approx(107.5)


def test_vwap_resets_at_session_boundary():
    v = SessionVWAP()
    # Session day 2026-06-01: bars at 18:00 and 23:00
    v.on_bar(bar("2026-06-01 18:00", 100, 102, 98, 100, 100))
    v.on_bar(bar("2026-06-01 23:00", 100, 100, 100, 100, 100))
    # Cross into session day 2026-06-02 at 18:00 the next day
    v.on_bar(bar("2026-06-02 18:00", 200, 200, 200, 200, 50))
    # New session: VWAP should be just the new bar's typical price (200)
    assert v.current == pytest.approx(200.0)
    assert v.current_session_day == date(2026, 6, 2)


def test_vwap_zero_volume_bar_does_not_break():
    v = SessionVWAP()
    v.on_bar(bar("2026-06-01 18:00", 100, 100, 100, 100, 0))
    # Still None — no volume contribution yet
    assert v.current is None
    v.on_bar(bar("2026-06-01 18:01", 100, 100, 100, 100, 50))
    assert v.current == pytest.approx(100.0)


def test_vwap_continues_across_within_session_bars():
    """Bars within the same session day accumulate, regardless of which
    sub-session (Asia/London/NY) they fall in."""
    v = SessionVWAP()
    # Asia of session day 6/1
    v.on_bar(bar("2026-06-01 19:00", 100, 100, 100, 100, 100))
    # London of session day 6/1 (next calendar day, before 17:00 ET)
    v.on_bar(bar("2026-06-02 05:00", 110, 110, 110, 110, 100))
    # NY of session day 6/1
    v.on_bar(bar("2026-06-02 10:00", 120, 120, 120, 120, 100))
    # All three accumulate -> VWAP = (100+110+120) / 3 = 110
    assert v.current == pytest.approx(110.0)
    assert v.current_session_day == date(2026, 6, 1)
