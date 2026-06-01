"""Reference levels tests.

Each test builds a small synthetic bar stream, feeds it through
ReferenceLevels.on_bar, then checks active_levels() at known time points.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from strategy.bars import Bar
from strategy.levels import LevelKind, ReferenceLevels
from strategy.params import DEFAULT_PARAMS


def et(ts: str) -> pd.Timestamp:
    return pd.Timestamp(ts).tz_localize("US/Eastern").tz_convert("UTC")


def bar(ts: str, o: float, h: float, l: float, c: float, v: float = 100.0) -> Bar:
    return Bar(ts=et(ts), open=o, high=h, low=l, close=c, volume=v)


# ---------------------------------------------------------------------------
# Daily open
# ---------------------------------------------------------------------------

def test_daily_open_locked_at_first_bar_of_session():
    rl = ReferenceLevels(DEFAULT_PARAMS)
    rl.on_bar(bar("2026-06-01 18:00", 20000, 20010, 19990, 20005))
    # First bar's open is the daily open for session day 2026-06-01
    daily = [l for l in rl.active_levels() if l.kind == LevelKind.DAILY_OPEN]
    assert len(daily) == 1
    assert daily[0].price == 20000
    assert daily[0].source_day == date(2026, 6, 1)

    # Later bar in same session day doesn't change the open
    rl.on_bar(bar("2026-06-01 19:00", 20100, 20150, 20095, 20120))
    daily = [l for l in rl.active_levels() if l.kind == LevelKind.DAILY_OPEN]
    assert daily[0].price == 20000


def test_daily_open_rolls_at_session_boundary():
    rl = ReferenceLevels(DEFAULT_PARAMS)
    rl.on_bar(bar("2026-06-01 18:00", 20000, 20010, 19990, 20005))
    rl.on_bar(bar("2026-06-02 09:30", 20100, 20110, 20090, 20105))  # still session day 6/1
    daily = [l for l in rl.active_levels() if l.kind == LevelKind.DAILY_OPEN]
    assert daily[0].source_day == date(2026, 6, 1)
    assert daily[0].price == 20000

    rl.on_bar(bar("2026-06-02 18:00", 20200, 20210, 20190, 20205))  # new session day 6/2
    daily = [l for l in rl.active_levels() if l.kind == LevelKind.DAILY_OPEN]
    assert daily[0].source_day == date(2026, 6, 2)
    assert daily[0].price == 20200


# ---------------------------------------------------------------------------
# Prior session H/L
# ---------------------------------------------------------------------------

def test_no_prior_session_until_second_day():
    rl = ReferenceLevels(DEFAULT_PARAMS)
    rl.on_bar(bar("2026-06-01 18:00", 20000, 20050, 19950, 20020))
    rl.on_bar(bar("2026-06-01 23:00", 20020, 20100, 19900, 20050))
    # No prior session yet — only one session day's data
    prior = [l for l in rl.active_levels() if l.kind in (LevelKind.PRIOR_SESS_HIGH, LevelKind.PRIOR_SESS_LOW)]
    assert prior == []


def test_prior_session_reflects_completed_day():
    rl = ReferenceLevels(DEFAULT_PARAMS)
    # Session day 6/1 from 18:00 onwards
    rl.on_bar(bar("2026-06-01 18:00", 20000, 20050, 19950, 20020))
    rl.on_bar(bar("2026-06-01 23:00", 20020, 20200, 19900, 20100))  # establishes session high/low
    rl.on_bar(bar("2026-06-02 09:30", 20100, 20150, 20080, 20120))  # still session day 6/1
    rl.on_bar(bar("2026-06-02 14:00", 20120, 20130, 20050, 20060))  # still session day 6/1
    # Roll to session day 6/2
    rl.on_bar(bar("2026-06-02 18:30", 20070, 20080, 20060, 20075))

    levels_by_kind = {l.kind: l for l in rl.active_levels()}
    psh = levels_by_kind[LevelKind.PRIOR_SESS_HIGH]
    psl = levels_by_kind[LevelKind.PRIOR_SESS_LOW]
    assert psh.price == 20200  # max high of session day 6/1
    assert psl.price == 19900  # min low of session day 6/1
    assert psh.source_day == date(2026, 6, 1)


# ---------------------------------------------------------------------------
# Asia H/L locking behavior
# ---------------------------------------------------------------------------

def test_asia_not_active_during_asia_window():
    """During the Asia window, the Asia H/L of *that* session is still
    evolving and should not be exposed. Only the prior session's Asia H/L,
    if it exists, would be active."""
    rl = ReferenceLevels(DEFAULT_PARAMS)
    # First Asia bar on session day 6/1
    rl.on_bar(bar("2026-06-01 19:00", 20000, 20100, 19950, 20050))
    rl.on_bar(bar("2026-06-01 23:00", 20050, 20150, 19980, 20100))

    asia = [l for l in rl.active_levels() if l.kind in (LevelKind.ASIA_HIGH, LevelKind.ASIA_LOW)]
    assert asia == []  # current session's Asia not yet locked; no prior session


def test_asia_locks_at_3am_et():
    rl = ReferenceLevels(DEFAULT_PARAMS)
    rl.on_bar(bar("2026-06-01 19:00", 20000, 20100, 19950, 20050))
    rl.on_bar(bar("2026-06-01 23:00", 20050, 20150, 19980, 20100))
    rl.on_bar(bar("2026-06-02 02:59", 20100, 20180, 20090, 20120))  # last Asia bar

    # Still in Asia window — not locked yet
    asia = [l for l in rl.active_levels() if l.kind in (LevelKind.ASIA_HIGH, LevelKind.ASIA_LOW)]
    assert asia == []

    rl.on_bar(bar("2026-06-02 03:00", 20120, 20130, 20110, 20115))  # first London bar; Asia closed
    levels_by_kind = {l.kind: l for l in rl.active_levels()}
    assert levels_by_kind[LevelKind.ASIA_HIGH].price == 20180
    assert levels_by_kind[LevelKind.ASIA_LOW].price == 19950
    assert levels_by_kind[LevelKind.ASIA_HIGH].source_day == date(2026, 6, 1)


def test_asia_excludes_non_asia_bars():
    """A high made during NY of the *previous* session day should not pollute
    the Asia H/L of the current session day."""
    rl = ReferenceLevels(DEFAULT_PARAMS)
    # NY bar with very high high — should NOT count toward Asia
    rl.on_bar(bar("2026-06-02 10:00", 21000, 25000, 21000, 21000))  # session day 6/1, NY
    # Asia of session day 6/1 starts at 19:00 of 6/1, which is BEFORE this bar in real time.
    # So we need to construct this differently: Asia is 19:00 6/1 -> 03:00 6/2.
    # A 10:00 6/2 bar is in session day 6/1's NY, post-Asia.
    # Set Asia bars first.
    rl2 = ReferenceLevels(DEFAULT_PARAMS)
    rl2.on_bar(bar("2026-06-01 19:00", 20000, 20100, 19950, 20050))  # Asia, sd 6/1
    rl2.on_bar(bar("2026-06-02 02:00", 20050, 20150, 20000, 20100))  # Asia, sd 6/1
    rl2.on_bar(bar("2026-06-02 10:00", 21000, 25000, 21000, 21000))  # NY, sd 6/1 — should NOT update Asia
    rl2.on_bar(bar("2026-06-02 18:30", 20500, 20510, 20490, 20500))  # roll to sd 6/2, ask Asia level

    levels_by_kind = {l.kind: l for l in rl2.active_levels()}
    # Asia of sd 6/1 should be 20150 / 19950 — the NY 25000 must not leak in
    assert levels_by_kind[LevelKind.ASIA_HIGH].price == 20150
    assert levels_by_kind[LevelKind.ASIA_LOW].price == 19950


# ---------------------------------------------------------------------------
# London H/L locking behavior (mirrors Asia)
# ---------------------------------------------------------------------------

def test_london_not_active_during_london_window():
    rl = ReferenceLevels(DEFAULT_PARAMS)
    rl.on_bar(bar("2026-06-02 03:00", 20100, 20200, 20080, 20150))  # in London
    rl.on_bar(bar("2026-06-02 06:00", 20150, 20250, 20120, 20200))  # in London
    lon = [l for l in rl.active_levels() if l.kind in (LevelKind.LONDON_HIGH, LevelKind.LONDON_LOW)]
    assert lon == []


def test_london_locks_at_8am_et():
    rl = ReferenceLevels(DEFAULT_PARAMS)
    rl.on_bar(bar("2026-06-02 03:00", 20100, 20200, 20080, 20150))  # London bar
    rl.on_bar(bar("2026-06-02 06:00", 20150, 20250, 20120, 20200))  # London bar
    rl.on_bar(bar("2026-06-02 07:59", 20200, 20210, 20100, 20180))  # last London bar
    lon = [l for l in rl.active_levels() if l.kind in (LevelKind.LONDON_HIGH, LevelKind.LONDON_LOW)]
    assert lon == []  # not yet locked

    rl.on_bar(bar("2026-06-02 08:15", 20180, 20190, 20170, 20180))  # in 8:00-8:30 gap (OTHER); London locked
    levels_by_kind = {l.kind: l for l in rl.active_levels()}
    assert levels_by_kind[LevelKind.LONDON_HIGH].price == 20250
    assert levels_by_kind[LevelKind.LONDON_LOW].price == 20080
    assert levels_by_kind[LevelKind.LONDON_HIGH].source_day == date(2026, 6, 1)


def test_london_excludes_non_london_bars():
    rl = ReferenceLevels(DEFAULT_PARAMS)
    rl.on_bar(bar("2026-06-02 02:00", 20000, 21000, 20000, 20500))  # Asia bar; high should NOT count toward London
    rl.on_bar(bar("2026-06-02 03:30", 20500, 20600, 20480, 20550))  # London
    rl.on_bar(bar("2026-06-02 10:00", 20550, 22000, 20550, 21000))  # NY; high should NOT count toward London
    rl.on_bar(bar("2026-06-02 18:30", 20800, 20810, 20790, 20800))  # roll session day

    levels_by_kind = {l.kind: l for l in rl.active_levels()}
    assert levels_by_kind[LevelKind.LONDON_HIGH].price == 20600
    assert levels_by_kind[LevelKind.LONDON_LOW].price == 20480


# ---------------------------------------------------------------------------
# Prior week H/L
# ---------------------------------------------------------------------------

def test_prior_week_h_l():
    rl = ReferenceLevels(DEFAULT_PARAMS)
    # Week of 2026-05-24 (Sun). Session days: 5/24..5/29
    rl.on_bar(bar("2026-05-24 19:00", 20000, 20300, 19700, 20100))  # week 5/24
    rl.on_bar(bar("2026-05-26 10:00", 20100, 20200, 20050, 20150))  # week 5/24
    # New week starts Sun 2026-05-31
    rl.on_bar(bar("2026-05-31 19:00", 20150, 20160, 20140, 20155))  # week 5/31

    levels_by_kind = {l.kind: l for l in rl.active_levels()}
    assert levels_by_kind[LevelKind.PRIOR_WEEK_HIGH].price == 20300
    assert levels_by_kind[LevelKind.PRIOR_WEEK_LOW].price == 19700
    assert levels_by_kind[LevelKind.PRIOR_WEEK_HIGH].source_day == date(2026, 5, 24)


# ---------------------------------------------------------------------------
# Round numbers
# ---------------------------------------------------------------------------

def test_round_majors_at_100_pt_grid():
    rl = ReferenceLevels(DEFAULT_PARAMS)
    rl.on_bar(bar("2026-06-01 18:00", 20000, 20010, 19990, 20000))
    majors = sorted([l.price for l in rl.active_levels() if l.kind == LevelKind.ROUND_MAJOR])
    assert all(int(p) % 100 == 0 for p in majors)
    assert 20000.0 in majors
    assert min(majors) == 19500.0
    assert max(majors) == 20500.0


def test_round_minors_at_25_pt_grid_excluding_majors():
    rl = ReferenceLevels(DEFAULT_PARAMS)
    rl.on_bar(bar("2026-06-01 18:00", 20000, 20010, 19990, 20000))
    minors = sorted([l.price for l in rl.active_levels() if l.kind == LevelKind.ROUND_MINOR])
    # All minors should be on the 25-pt grid
    assert all(round(p * 4) % 25 == 0 for p in minors)  # multiples of 0.25... oh wait, 25 pts.
    # Better: multiples of 25
    assert all(int(p) % 25 == 0 for p in minors)
    # None of the minors should also be a major (100-pt) — those promoted to MAJOR
    assert all(int(p) % 100 != 0 for p in minors)
    # Sanity: 20025, 20050, 20075 should all be minors near current price
    assert 20025.0 in minors
    assert 20050.0 in minors
    assert 20075.0 in minors
    # 20000 should NOT appear in minors (it's a major)
    assert 20000.0 not in minors


def test_round_grid_shifts_with_price():
    rl = ReferenceLevels(DEFAULT_PARAMS)
    rl.on_bar(bar("2026-06-01 18:00", 19850, 19850, 19850, 19850))
    all_rounds = sorted([
        l.price for l in rl.active_levels()
        if l.kind in (LevelKind.ROUND_MAJOR, LevelKind.ROUND_MINOR)
    ])
    assert min(all_rounds) == 19350.0
    assert max(all_rounds) == 20350.0


# ---------------------------------------------------------------------------
# Weekly open
# ---------------------------------------------------------------------------

def test_weekly_open_locked_at_first_bar_of_week():
    rl = ReferenceLevels(DEFAULT_PARAMS)
    rl.on_bar(bar("2026-05-31 18:00", 19500, 19510, 19490, 19505))  # Sun open week 5/31
    rl.on_bar(bar("2026-06-01 10:00", 19600, 19610, 19590, 19605))  # same week, later
    wo = [l for l in rl.active_levels() if l.kind == LevelKind.WEEKLY_OPEN]
    assert len(wo) == 1
    assert wo[0].price == 19500
    assert wo[0].source_day == date(2026, 5, 31)


# ---------------------------------------------------------------------------
# EQH / EQL clusters (from confirmed N=5 swings, 2-tick tolerance)
# ---------------------------------------------------------------------------

def _eqhl_bar_stream(highs: list[float], lows: list[float], start_minute: int = 0):
    """Build a stream of bars from per-bar high/low arrays. Open and close
    are placed inside [low, high]. Volume = 100. Bars are 1 minute apart."""
    base = pd.Timestamp("2026-06-01 18:00").tz_localize("US/Eastern").tz_convert("UTC")
    out = []
    for i, (h, l) in enumerate(zip(highs, lows)):
        mid = (h + l) / 2
        out.append(Bar(
            ts=base + pd.Timedelta(minutes=start_minute + i),
            open=mid, high=h, low=l, close=mid, volume=100,
        ))
    return out


def test_no_eqh_with_only_one_swing_high():
    rl = ReferenceLevels(DEFAULT_PARAMS)
    # Construct a single swing high at idx 5 (N=5 -> 11-bar pattern)
    highs = [100, 101, 102, 103, 104, 110, 104, 103, 102, 101, 100]
    lows  = [ 90,  90,  90,  90,  90,  90,  90,  90,  90,  90,  90]
    for b in _eqhl_bar_stream(highs, lows):
        rl.on_bar(b)
    eqh = [l for l in rl.active_levels() if l.kind == LevelKind.EQH]
    assert eqh == []


def test_eqh_from_two_exactly_equal_swing_highs():
    rl = ReferenceLevels(DEFAULT_PARAMS)
    # Two swing highs at exactly 110.0 separated by valley
    # Swing high candidates at indices 5 and 15.
    # First peak: 11-bar window centered at idx 5 with peak 110
    # Valley between (low) at idx 10 to break symmetry
    # Second peak: window centered at idx 15 with peak 110
    highs = (
        [100, 101, 102, 103, 104, 110, 104, 103, 102, 101]  # peak at 5 = 110
        + [102, 103, 104, 105, 106, 110, 106, 105, 104, 103, 102]  # peak at 15 = 110
    )
    lows = [90] * len(highs)
    for b in _eqhl_bar_stream(highs, lows):
        rl.on_bar(b)
    eqh = [l for l in rl.active_levels() if l.kind == LevelKind.EQH]
    assert len(eqh) == 1
    assert eqh[0].price == 110.0


def test_eqh_within_tolerance():
    """Two swing highs 0.50 pt (= 2 ticks) apart still cluster."""
    rl = ReferenceLevels(DEFAULT_PARAMS)
    highs = (
        [100, 101, 102, 103, 104, 110.00, 104, 103, 102, 101]
        + [102, 103, 104, 105, 106, 110.50, 106, 105, 104, 103, 102]
    )
    lows = [90] * len(highs)
    for b in _eqhl_bar_stream(highs, lows):
        rl.on_bar(b)
    eqh = [l for l in rl.active_levels() if l.kind == LevelKind.EQH]
    assert len(eqh) == 1
    assert eqh[0].price == pytest.approx(110.25)  # average


def test_no_eqh_outside_tolerance():
    """Two swing highs more than 2 ticks apart do NOT cluster."""
    rl = ReferenceLevels(DEFAULT_PARAMS)
    highs = (
        [100, 101, 102, 103, 104, 110.00, 104, 103, 102, 101]
        + [102, 103, 104, 105, 106, 110.75, 106, 105, 104, 103, 102]  # 3 ticks apart
    )
    lows = [90] * len(highs)
    for b in _eqhl_bar_stream(highs, lows):
        rl.on_bar(b)
    eqh = [l for l in rl.active_levels() if l.kind == LevelKind.EQH]
    assert eqh == []


def test_eql_from_two_equal_swing_lows():
    rl = ReferenceLevels(DEFAULT_PARAMS)
    # Swing low candidates at indices 5 and 15 at 90.0
    lows = (
        [100, 99, 98, 97, 96, 90, 96, 97, 98, 99]
        + [98, 97, 96, 95, 94, 90, 94, 95, 96, 97, 98]
    )
    highs = [110] * len(lows)
    for b in _eqhl_bar_stream(highs, lows):
        rl.on_bar(b)
    eql = [l for l in rl.active_levels() if l.kind == LevelKind.EQL]
    assert len(eql) == 1
    assert eql[0].price == 90.0


def test_eqh_max_age_filter_excludes_old_swings():
    """An old swing past max_age_bars should not cluster with a new one."""
    from dataclasses import replace
    short_age_params = replace(DEFAULT_PARAMS, eqhl_max_age_bars=20)

    rl = ReferenceLevels(short_age_params)
    # First peak at idx 5
    # Then enough filler bars to push idx 5 out of the 20-bar age window
    # before the second peak forms
    highs = (
        [100, 101, 102, 103, 104, 110, 104, 103, 102, 101]  # peak at 5
        + [100] * 30  # 30 filler bars
        + [102, 103, 104, 105, 106, 110, 106, 105, 104, 103, 102]  # peak ~idx 45
    )
    lows = [90] * len(highs)
    for b in _eqhl_bar_stream(highs, lows):
        rl.on_bar(b)
    eqh = [l for l in rl.active_levels() if l.kind == LevelKind.EQH]
    # Only one swing high is recent enough → no cluster
    assert eqh == []


def test_eqh_cluster_of_three_swings():
    """Three swing highs all within tolerance produce a single EQH at their mean."""
    rl = ReferenceLevels(DEFAULT_PARAMS)
    # Three peaks at 110.00, 110.25, 110.50 — span 0.50 = 2 ticks
    highs = (
        [100, 101, 102, 103, 104, 110.00, 104, 103, 102, 101]
        + [102, 103, 104, 105, 106, 110.25, 106, 105, 104, 103]
        + [102, 103, 104, 105, 106, 110.50, 106, 105, 104, 103, 102]
    )
    lows = [90] * len(highs)
    for b in _eqhl_bar_stream(highs, lows):
        rl.on_bar(b)
    eqh = [l for l in rl.active_levels() if l.kind == LevelKind.EQH]
    assert len(eqh) == 1
    assert eqh[0].price == pytest.approx(110.25)  # mean of 110.00 + 110.25 + 110.50
