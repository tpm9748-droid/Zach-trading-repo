"""Sweep detector tests."""
from __future__ import annotations

from dataclasses import replace
from datetime import date

import pandas as pd
import pytest

from strategy.bars import Bar
from strategy.levels import Level, LevelKind
from strategy.params import DEFAULT_PARAMS, NQ_TICK_SIZE
from strategy.sweep import SweepDetector, SweepDirection, SweepEvent


def bar(minute: int, o: float, h: float, l: float, c: float, v: float = 100.0) -> Bar:
    base = pd.Timestamp("2026-06-01 09:30").tz_localize("US/Eastern").tz_convert("UTC")
    return Bar(ts=base + pd.Timedelta(minutes=minute), open=o, high=h, low=l, close=c, volume=v)


def asia_high_level(price: float, source_day=date(2026, 6, 1)) -> Level:
    return Level(price=price, kind=LevelKind.ASIA_HIGH, source_day=source_day)


def warmup(det: SweepDetector, n: int, base_price: float = 100.0, vol: float = 100.0):
    """Feed n filler bars to build up volume history."""
    for i in range(n):
        det.on_bar(bar(i, base_price, base_price + 0.5, base_price - 0.5, base_price, vol), [])


def _bar_d(minute, o, h, l, c, v, d) -> Bar:
    base = pd.Timestamp("2026-06-01 09:30").tz_localize("US/Eastern").tz_convert("UTC")
    return Bar(ts=base + pd.Timedelta(minutes=minute), open=o, high=h, low=l, close=c, volume=v, delta=d)


def test_delta_confirmation_requires_opposing_flow():
    p = replace(DEFAULT_PARAMS, sweep_use_delta_confirmation=True)
    # Up-sweep rejection with net SELLING (delta < 0) -> confirmed.
    d1 = SweepDetector(p)
    warmup(d1, p.volume_window, 99, 100)
    ev = d1.on_bar(_bar_d(50, 99, 100.75, 98, 99.5, 200, -50), [asia_high_level(100.0)])
    assert len(ev) == 1
    # Same geometry but net BUYING (delta > 0) -> rejected by the delta filter.
    d2 = SweepDetector(p)
    warmup(d2, p.volume_window, 99, 100)
    ev2 = d2.on_bar(_bar_d(50, 99, 100.75, 98, 99.5, 200, +50), [asia_high_level(100.0)])
    assert ev2 == []


def test_delta_confirmation_off_ignores_delta():
    p = DEFAULT_PARAMS  # filter off by default
    d = SweepDetector(p)
    warmup(d, p.volume_window, 99, 100)
    # Net buying on an up-sweep, but filter off -> still fires on volume alone.
    ev = d.on_bar(_bar_d(50, 99, 100.75, 98, 99.5, 200, +50), [asia_high_level(100.0)])
    assert len(ev) == 1


# ---------------------------------------------------------------------------
# Volume bootstrap
# ---------------------------------------------------------------------------

def test_no_sweep_until_volume_history_full():
    """Until volume_window bars seen, no sweep can fire."""
    d = SweepDetector(DEFAULT_PARAMS)
    level = asia_high_level(100.0)
    # Same-bar perfect sweep but no history -> ratio is None -> no fire
    ev = d.on_bar(bar(0, 99, 100.75, 98, 99.5, 1000), [level])
    assert ev == []


# ---------------------------------------------------------------------------
# Same-bar sweep
# ---------------------------------------------------------------------------

def test_same_bar_sweep_up_with_volume_fires():
    p = DEFAULT_PARAMS
    d = SweepDetector(p)
    level = asia_high_level(100.0)
    warmup(d, p.volume_window, base_price=99, vol=100)
    # Sweep up: open 99, high 100.75 (3 ticks above), low 98, close 99.5 (back inside)
    # Volume 200 = 2x SMA(100) -> passes 1.5x threshold
    events = d.on_bar(bar(50, 99, 100.75, 98, 99.5, 200), [level])
    assert len(events) == 1
    ev = events[0]
    assert ev.direction == SweepDirection.UP
    assert ev.level == level
    assert ev.penetration_bar_idx == ev.rejection_bar_idx  # same bar
    assert ev.penetration_ticks == 3
    assert ev.wick_extreme == 100.75
    assert ev.volume_ratio == 2.0


def test_same_bar_sweep_up_without_volume_no_fire():
    p = DEFAULT_PARAMS
    d = SweepDetector(p)
    level = asia_high_level(100.0)
    warmup(d, p.volume_window, base_price=99, vol=100)
    # Sweep with only 1.2x volume -> fails 1.5x
    events = d.on_bar(bar(50, 99, 100.75, 98, 99.5, 120), [level])
    assert events == []


def test_same_bar_sweep_down():
    p = DEFAULT_PARAMS
    d = SweepDetector(p)
    level = Level(price=100.0, kind=LevelKind.ASIA_LOW, source_day=date(2026, 6, 1))
    warmup(d, p.volume_window, base_price=101, vol=100)
    # Open 101, low 99.25 (3 ticks below), high 102, close 100.5 (back above)
    events = d.on_bar(bar(50, 101, 102, 99.25, 100.5, 200), [level])
    assert len(events) == 1
    ev = events[0]
    assert ev.direction == SweepDirection.DOWN
    assert ev.penetration_ticks == 3
    assert ev.wick_extreme == 99.25


# ---------------------------------------------------------------------------
# Multi-bar sweep
# ---------------------------------------------------------------------------

def test_multi_bar_sweep_up():
    """Bar T penetrates (close still above level), bar T+2 closes back inside with volume."""
    p = DEFAULT_PARAMS
    d = SweepDetector(p)
    level = asia_high_level(100.0)
    warmup(d, p.volume_window, base_price=99, vol=100)
    # Bar T: penetrate, close above (no same-bar resolution)
    events = d.on_bar(bar(50, 99.5, 100.75, 99, 100.5, 150), [level])
    assert events == []
    # Bar T+1: still above, no resolution
    events = d.on_bar(bar(51, 100.5, 100.6, 100.1, 100.25, 150), [level])
    assert events == []
    # Bar T+2: close back below 100 with good volume
    events = d.on_bar(bar(52, 100.25, 100.3, 99, 99.5, 200), [level])
    assert len(events) == 1
    ev = events[0]
    # bar_idx is the detector's internal counter (warmup=20 bars), so first
    # post-warmup bar is index 20, then 21, 22
    assert ev.penetration_bar_idx == 20
    assert ev.rejection_bar_idx == 22


def test_multi_bar_sweep_expires():
    """If close-back doesn't happen within max_rejection_bars, no event."""
    p = DEFAULT_PARAMS  # max_rejection_bars=3
    d = SweepDetector(p)
    level = asia_high_level(100.0)
    warmup(d, p.volume_window, base_price=99, vol=100)
    # Bar T: penetrate, close above
    d.on_bar(bar(50, 99.5, 100.75, 99, 100.5, 200), [level])
    # Next 4 bars: stay above the level (never close back). Expired.
    d.on_bar(bar(51, 100.5, 100.6, 100.1, 100.25, 200), [level])
    d.on_bar(bar(52, 100.25, 100.6, 100.1, 100.5, 200), [level])
    d.on_bar(bar(53, 100.5, 100.7, 100.2, 100.6, 200), [level])
    # By now, the candidate from bar 50 should have expired (3 bars elapsed without close-back)
    events = d.on_bar(bar(54, 100.6, 100.65, 99, 99.5, 200), [level])
    # The close-back here is too late — candidate should be gone
    # BUT this is also a fresh down-penetration? No, it's a fresh up-... actually bar opens 100.6 (above level) and closes below — no fresh penetration rule applies (bar.open > level, no fresh "up sweep" from below). So no event.
    assert events == []


# ---------------------------------------------------------------------------
# Penetration depth boundaries
# ---------------------------------------------------------------------------

def test_below_min_ticks_no_sweep():
    """2-tick penetration (below 3-tick minimum) -> no candidate."""
    p = DEFAULT_PARAMS
    d = SweepDetector(p)
    level = asia_high_level(100.0)
    warmup(d, p.volume_window, base_price=99, vol=100)
    # 100.5 = 2 ticks above
    events = d.on_bar(bar(50, 99, 100.5, 98, 99.5, 300), [level])
    assert events == []


def test_max_valid_ticks_boundary():
    """8-tick penetration: still valid."""
    p = DEFAULT_PARAMS
    d = SweepDetector(p)
    level = asia_high_level(100.0)
    warmup(d, p.volume_window, base_price=99, vol=100)
    events = d.on_bar(bar(50, 99, 102.0, 98, 99.5, 200), [level])  # 8 ticks
    assert len(events) == 1
    assert events[0].penetration_ticks == 8


def test_above_broken_ticks_marks_level_broken():
    """11-tick penetration: level broken, never sweeps again."""
    p = DEFAULT_PARAMS
    d = SweepDetector(p)
    level = asia_high_level(100.0)
    warmup(d, p.volume_window, base_price=99, vol=100)
    # 11 ticks above
    events = d.on_bar(bar(50, 99, 102.75, 98, 99.5, 1000), [level])
    assert events == []
    # Subsequent valid penetration on same level should also be ignored
    events2 = d.on_bar(bar(51, 99, 100.75, 98, 99.5, 1000), [level])
    assert events2 == []


def test_ambiguous_depth_zone_no_event_no_break():
    """9-10 tick penetration: skipped silently, level remains active."""
    p = DEFAULT_PARAMS
    d = SweepDetector(p)
    level = asia_high_level(100.0)
    warmup(d, p.volume_window, base_price=99, vol=100)
    # 9 ticks above = 100 + 2.25 = 102.25 -> open 99 close 99.5
    events = d.on_bar(bar(50, 99, 102.25, 98, 99.5, 1000), [level])
    assert events == []
    # Level should still be active — a valid penetration later still fires
    events2 = d.on_bar(bar(51, 99, 100.75, 98, 99.5, 1000), [level])
    assert len(events2) == 1


# ---------------------------------------------------------------------------
# Fresh-penetration rule
# ---------------------------------------------------------------------------

def test_no_fresh_penetration_if_open_already_above():
    """Bar opens above the level -> not a fresh sweep."""
    p = DEFAULT_PARAMS
    d = SweepDetector(p)
    level = asia_high_level(100.0)
    warmup(d, p.volume_window, base_price=101, vol=100)
    # Bar opens at 101 (above level 100), wicks higher, closes back
    events = d.on_bar(bar(50, 101, 102.75, 100.5, 101.5, 200), [level])
    assert events == []


# ---------------------------------------------------------------------------
# Pending-mid-flight broken
# ---------------------------------------------------------------------------

def test_pending_candidate_marked_broken_if_wick_extends_too_far():
    """Bar T penetrates 3 ticks. Bar T+1 extends wick to 11 ticks -> broken."""
    p = DEFAULT_PARAMS
    d = SweepDetector(p)
    level = asia_high_level(100.0)
    warmup(d, p.volume_window, base_price=99, vol=100)
    d.on_bar(bar(50, 99.5, 100.75, 99, 100.5, 200), [level])  # 3 ticks, pending
    d.on_bar(bar(51, 100.5, 102.75, 100.2, 102.0, 200), [level])  # extends 11 ticks total -> broken
    # Subsequent valid penetration ignored
    events = d.on_bar(bar(52, 99, 100.75, 98, 99.5, 200), [level])
    assert events == []


# ---------------------------------------------------------------------------
# Multiple levels at once
# ---------------------------------------------------------------------------

def test_multiple_levels_independent():
    """Sweep up of one level + sweep down of another in different bars."""
    p = DEFAULT_PARAMS
    d = SweepDetector(p)
    asia_high = Level(price=110.0, kind=LevelKind.ASIA_HIGH, source_day=date(2026, 6, 1))
    asia_low = Level(price=90.0, kind=LevelKind.ASIA_LOW, source_day=date(2026, 6, 1))
    warmup(d, p.volume_window, base_price=100, vol=100)
    # Sweep asia high
    events = d.on_bar(bar(50, 109, 110.75, 108, 109.5, 200), [asia_high, asia_low])
    assert len(events) == 1
    assert events[0].direction == SweepDirection.UP

    # Sweep asia low
    events = d.on_bar(bar(51, 91, 92, 89.25, 90.5, 200), [asia_high, asia_low])
    assert len(events) == 1
    assert events[0].direction == SweepDirection.DOWN


def test_once_swept_level_not_revisited():
    """A level that was swept doesn't fire again on subsequent penetrations."""
    p = DEFAULT_PARAMS
    d = SweepDetector(p)
    level = asia_high_level(100.0)
    warmup(d, p.volume_window, base_price=99, vol=100)
    # First sweep fires
    events = d.on_bar(bar(50, 99, 100.75, 98, 99.5, 200), [level])
    assert len(events) == 1
    # Second penetration should be ignored
    events = d.on_bar(bar(51, 99, 100.75, 98, 99.5, 200), [level])
    assert events == []
