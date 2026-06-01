"""Session classification for CME equity index futures.

All session windows are defined in US/Eastern (handles DST automatically).
Bars are stored in UTC; we convert to ET when classifying.

Session windows (per the strategy spec):
  - Electronic session day: D 18:00 ET -> D+1 17:00 ET  (CME break 17:00-18:00)
  - Asia:    19:00 - 03:00 ET
  - London:  03:00 - 08:00 ET
  - NY:      08:30 - 16:00 ET   (RTH cash equity overlap)

The 08:00-08:30 ET window is intentionally OTHER, matching the spec's
explicit gap between London close and NY open.

A "session day" is anchored to its open date. SessionDay(2026-06-01) starts
at 2026-06-01 18:00 ET and ends 2026-06-02 17:00 ET.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import Enum
from typing import Optional

import pandas as pd


ET = "US/Eastern"


class Session(Enum):
    ASIA = "asia"
    LONDON = "london"
    NY = "ny"
    OTHER = "other"  # 16:00-17:00 ET, 17:00-18:00 break (no bars), 18:00-19:00 ET


@dataclass(frozen=True)
class SessionWindow:
    """[start, end) in UTC."""
    start: pd.Timestamp
    end: pd.Timestamp

    def contains(self, ts: pd.Timestamp) -> bool:
        return self.start <= ts < self.end


def to_et(ts: pd.Timestamp) -> pd.Timestamp:
    if ts.tzinfo is None:
        raise ValueError("ts must be tz-aware")
    return ts.tz_convert(ET)


def classify(ts: pd.Timestamp) -> Session:
    """Which session window does this timestamp fall into?"""
    et = to_et(ts)
    t = et.time()
    if time(19, 0) <= t or t < time(3, 0):
        return Session.ASIA
    if time(3, 0) <= t < time(8, 0):
        return Session.LONDON
    if time(8, 30) <= t < time(16, 0):
        return Session.NY
    return Session.OTHER


def session_day_of(ts: pd.Timestamp) -> date:
    """The session day a timestamp belongs to.

    A session day spans D 18:00 ET -> D+1 17:00 ET. The 17:00-18:00 ET
    maintenance break is dead air; we attribute it to the session that just
    closed (D), so session day D effectively covers D 18:00 -> D+1 17:59:59.
      - 2026-06-01 19:00 ET -> session day 2026-06-01
      - 2026-06-02 09:30 ET -> session day 2026-06-01
      - 2026-06-02 17:30 ET -> session day 2026-06-01 (in break, post-close)
      - 2026-06-02 18:00 ET -> session day 2026-06-02 (new session opens)
    """
    et = to_et(ts)
    if et.time() >= time(18, 0):
        return et.date()
    return (et - timedelta(days=1)).date()


def session_window(session_day: date, session: Session) -> SessionWindow:
    """Return [start, end) in UTC for a given (session_day, session)."""
    if session == Session.ASIA:
        # 19:00 D -> 03:00 D+1
        start_et = _et_dt(session_day, time(19, 0))
        end_et = _et_dt(session_day + timedelta(days=1), time(3, 0))
    elif session == Session.LONDON:
        # 03:00 D+1 -> 08:00 D+1
        start_et = _et_dt(session_day + timedelta(days=1), time(3, 0))
        end_et = _et_dt(session_day + timedelta(days=1), time(8, 0))
    elif session == Session.NY:
        # 08:30 D+1 -> 16:00 D+1
        start_et = _et_dt(session_day + timedelta(days=1), time(8, 30))
        end_et = _et_dt(session_day + timedelta(days=1), time(16, 0))
    else:
        raise ValueError(f"Cannot build window for {session}")
    return SessionWindow(
        start=pd.Timestamp(start_et).tz_convert("UTC"),
        end=pd.Timestamp(end_et).tz_convert("UTC"),
    )


def electronic_session_window(session_day: date) -> SessionWindow:
    """Full electronic session: D 18:00 ET -> D+1 17:00 ET."""
    start_et = _et_dt(session_day, time(18, 0))
    end_et = _et_dt(session_day + timedelta(days=1), time(17, 0))
    return SessionWindow(
        start=pd.Timestamp(start_et).tz_convert("UTC"),
        end=pd.Timestamp(end_et).tz_convert("UTC"),
    )


def _et_dt(d: date, t: time) -> pd.Timestamp:
    """Localize a naive (date, time) to US/Eastern."""
    return pd.Timestamp(datetime.combine(d, t)).tz_localize(ET)


def prior_session_day(ts: pd.Timestamp) -> date:
    """The session day immediately preceding the one containing ts."""
    return session_day_of(ts) - timedelta(days=1)


def week_of(ts: pd.Timestamp) -> date:
    """Return the Sunday-date of the futures week containing ts.

    Futures week: Sun 18:00 ET -> Fri 17:00 ET. We tag the week by its
    Sunday date.
    """
    sd = session_day_of(ts)
    # Sunday's weekday() == 6
    days_back = (sd.weekday() - 6) % 7
    return sd - timedelta(days=days_back)


def prior_week(ts: pd.Timestamp) -> date:
    return week_of(ts) - timedelta(days=7)
