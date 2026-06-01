from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from strategy.sessions import (
    Session,
    classify,
    electronic_session_window,
    prior_session_day,
    prior_week,
    session_day_of,
    session_window,
    week_of,
)


def et(ts: str) -> pd.Timestamp:
    return pd.Timestamp(ts).tz_localize("US/Eastern").tz_convert("UTC")


def test_classify_asia():
    assert classify(et("2026-06-01 19:00")) == Session.ASIA
    assert classify(et("2026-06-01 23:30")) == Session.ASIA
    assert classify(et("2026-06-02 02:59")) == Session.ASIA


def test_classify_london():
    assert classify(et("2026-06-02 03:00")) == Session.LONDON
    assert classify(et("2026-06-02 07:59")) == Session.LONDON


def test_classify_ny():
    assert classify(et("2026-06-02 08:30")) == Session.NY
    assert classify(et("2026-06-02 15:59")) == Session.NY


def test_classify_other():
    assert classify(et("2026-06-02 16:30")) == Session.OTHER
    assert classify(et("2026-06-02 18:00")) == Session.OTHER  # post-break, pre-Asia
    assert classify(et("2026-06-02 08:15")) == Session.OTHER  # London-to-NY gap


def test_session_day_attribution():
    # 19:00 ET on 6/1 -> session day 6/1
    assert session_day_of(et("2026-06-01 19:00")) == date(2026, 6, 1)
    # 09:30 ET next morning still session day 6/1
    assert session_day_of(et("2026-06-02 09:30")) == date(2026, 6, 1)
    # 17:30 ET (in maintenance break) still attributed to the session that just closed (6/1)
    assert session_day_of(et("2026-06-02 17:30")) == date(2026, 6, 1)
    # 18:00 ET -> new session day 6/2 begins
    assert session_day_of(et("2026-06-02 18:00")) == date(2026, 6, 2)


def test_prior_session_day():
    assert prior_session_day(et("2026-06-02 09:30")) == date(2026, 5, 31)


def test_asia_window_for_session_day():
    win = session_window(date(2026, 6, 1), Session.ASIA)
    assert win.start == et("2026-06-01 19:00")
    assert win.end == et("2026-06-02 03:00")


def test_london_window():
    win = session_window(date(2026, 6, 1), Session.LONDON)
    assert win.start == et("2026-06-02 03:00")
    assert win.end == et("2026-06-02 08:00")


def test_ny_window():
    win = session_window(date(2026, 6, 1), Session.NY)
    assert win.start == et("2026-06-02 08:30")
    assert win.end == et("2026-06-02 16:00")


def test_electronic_window():
    win = electronic_session_window(date(2026, 6, 1))
    assert win.start == et("2026-06-01 18:00")
    assert win.end == et("2026-06-02 17:00")


def test_week_tagging_by_sunday():
    # 2026-06-01 is a Monday. Sunday before = 2026-05-31.
    assert week_of(et("2026-06-01 19:00")) == date(2026, 5, 31)
    # 2026-06-04 (Thu) same week
    assert week_of(et("2026-06-04 10:00")) == date(2026, 5, 31)
    # 2026-06-07 (Sun) 19:00 -> session day 6/7 -> week 6/7
    assert week_of(et("2026-06-07 19:00")) == date(2026, 6, 7)


def test_prior_week():
    assert prior_week(et("2026-06-02 09:30")) == date(2026, 5, 24)


def test_classify_naive_rejected():
    with pytest.raises(ValueError, match="tz-aware"):
        classify(pd.Timestamp("2026-06-01 19:00"))


def test_dst_transition():
    """US DST in 2026 ends Sun 2026-11-01 02:00 ET.

    A timestamp before/after the change should still classify by ET wall clock.
    """
    pre = et("2026-10-31 19:00")  # EDT
    post = et("2026-11-02 19:00")  # EST
    assert classify(pre) == Session.ASIA
    assert classify(post) == Session.ASIA
