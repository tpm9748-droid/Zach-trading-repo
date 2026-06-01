"""Reference levels — the "where stops cluster" layer.

Maintains per-session-day and per-week running H/L stats incrementally as
bars arrive. Exposes `active_levels()` returning the currently tradable
reference levels at the time of the last seen bar.

What's here:
  - Prior session H/L (full electronic session, D-1)
  - Asia session H/L   (locked at 03:00 ET, the Asia close)
  - London session H/L (locked at 08:00 ET, the London close)
  - Prior week H/L
  - Daily open  (first bar of current session day, locked immediately)
  - Weekly open (first bar of current futures week, locked immediately)
  - Round-number levels: tiered into MAJOR (default 100pt) and MINOR
    (default 25pt). Minor levels at a major boundary are classified as
    major (higher tier wins). Both tiers are generated within +/- 500 pts
    of the current price.
  - Equal highs / equal lows (EQH/EQL): clusters of >=2 confirmed swing
    points (N=eqhl_swing_lookback) whose prices all lie within
    eqhl_tolerance_ticks * tick_size of each other, age-filtered to the
    last eqhl_max_age_bars bars. The cluster price emitted is the mean of
    the cluster's swing prices.

Convention: a "session day" SD spans D 18:00 ET -> D+1 17:00 ET. Asia of
SD spans D 19:00 ET -> D+1 03:00 ET. See strategy/sessions.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Optional

import pandas as pd

from strategy.bars import Bar
from strategy.params import NQ_TICK_SIZE, StrategyParams
from strategy.sessions import (
    Session,
    classify,
    session_day_of,
    session_window,
    week_of,
)
from strategy.swings import SwingDetector, SwingKind, SwingPoint


class LevelKind(Enum):
    PRIOR_SESS_HIGH = "prior_sess_high"
    PRIOR_SESS_LOW = "prior_sess_low"
    ASIA_HIGH = "asia_high"
    ASIA_LOW = "asia_low"
    LONDON_HIGH = "london_high"
    LONDON_LOW = "london_low"
    PRIOR_WEEK_HIGH = "prior_week_high"
    PRIOR_WEEK_LOW = "prior_week_low"
    DAILY_OPEN = "daily_open"
    WEEKLY_OPEN = "weekly_open"
    ROUND_MAJOR = "round_major"
    ROUND_MINOR = "round_minor"
    EQH = "eqh"
    EQL = "eql"


@dataclass(frozen=True)
class Level:
    """A single reference level at a moment in time.

    `source_day` is the session day or week the level was derived from
    (None for round numbers).
    """
    price: float
    kind: LevelKind
    source_day: Optional[date] = None


@dataclass
class _SessionStats:
    electronic_high: Optional[float] = None
    electronic_low: Optional[float] = None
    asia_high: Optional[float] = None
    asia_low: Optional[float] = None
    london_high: Optional[float] = None
    london_low: Optional[float] = None


@dataclass
class _WeekStats:
    high: Optional[float] = None
    low: Optional[float] = None


class ReferenceLevels:
    """Incremental tracker. Call on_bar() for each bar in time order."""

    def __init__(self, params: StrategyParams, round_half_range_pts: float = 500.0) -> None:
        self.params = params
        self.round_half_range_pts = round_half_range_pts
        self._sessions: dict[date, _SessionStats] = {}
        self._weeks: dict[date, _WeekStats] = {}
        self._daily_opens: dict[date, float] = {}
        self._weekly_opens: dict[date, float] = {}
        self._last_bar: Optional[Bar] = None
        # Owns its own swing detector at N=eqhl_swing_lookback for EQH/EQL.
        # Independent of the LTF swing detector used for CHoCH (different N).
        self._swing_detector = SwingDetector(n=params.eqhl_swing_lookback)
        self._bar_count = 0

    # ---- update path -------------------------------------------------------

    def on_bar(self, bar: Bar) -> None:
        sd = session_day_of(bar.ts)
        wk = week_of(bar.ts)

        # First bar seen for this session day locks the daily open.
        if sd not in self._daily_opens:
            self._daily_opens[sd] = bar.open
        if wk not in self._weekly_opens:
            self._weekly_opens[wk] = bar.open

        # Per-session-day full electronic H/L
        s = self._sessions.setdefault(sd, _SessionStats())
        s.electronic_high = bar.high if s.electronic_high is None else max(s.electronic_high, bar.high)
        s.electronic_low = bar.low if s.electronic_low is None else min(s.electronic_low, bar.low)

        # Asia / London H/L only update while inside their respective windows
        sess = classify(bar.ts)
        if sess == Session.ASIA:
            s.asia_high = bar.high if s.asia_high is None else max(s.asia_high, bar.high)
            s.asia_low = bar.low if s.asia_low is None else min(s.asia_low, bar.low)
        elif sess == Session.LONDON:
            s.london_high = bar.high if s.london_high is None else max(s.london_high, bar.high)
            s.london_low = bar.low if s.london_low is None else min(s.london_low, bar.low)

        # Per-week H/L
        w = self._weeks.setdefault(wk, _WeekStats())
        w.high = bar.high if w.high is None else max(w.high, bar.high)
        w.low = bar.low if w.low is None else min(w.low, bar.low)

        self._last_bar = bar
        self._swing_detector.on_bar(bar)
        self._bar_count += 1

    # ---- query path --------------------------------------------------------

    def active_levels(self) -> list[Level]:
        """Levels considered tradable as of the most recent bar's timestamp."""
        if self._last_bar is None:
            return []
        out: list[Level] = []
        now = self._last_bar.ts

        out.extend(self._prior_session_levels(now))
        out.extend(self._asia_levels(now))
        out.extend(self._london_levels(now))
        out.extend(self._prior_week_levels(now))
        out.extend(self._daily_open_level(now))
        out.extend(self._weekly_open_level(now))
        out.extend(self._round_levels(self._last_bar.close))
        out.extend(self._eqhl_levels(SwingKind.HIGH, LevelKind.EQH))
        out.extend(self._eqhl_levels(SwingKind.LOW, LevelKind.EQL))
        return out

    # ---- helpers -----------------------------------------------------------

    def _prior_session_levels(self, now: pd.Timestamp) -> list[Level]:
        sd = session_day_of(now)
        prior = sd - timedelta(days=1)
        s = self._sessions.get(prior)
        if s is None or s.electronic_high is None:
            return []
        return [
            Level(price=s.electronic_high, kind=LevelKind.PRIOR_SESS_HIGH, source_day=prior),
            Level(price=s.electronic_low, kind=LevelKind.PRIOR_SESS_LOW, source_day=prior),
        ]

    def _active_asia_session_day(self, now: pd.Timestamp) -> Optional[date]:
        """The session day whose Asia is most recently locked (Asia.end <= now)."""
        current_sd = session_day_of(now)
        # Current session's Asia closes at 03:00 ET on calendar (current_sd + 1)
        current_asia_end = session_window(current_sd, Session.ASIA).end
        if now >= current_asia_end and current_sd in self._sessions:
            return current_sd
        prior = current_sd - timedelta(days=1)
        if prior in self._sessions:
            return prior
        return None

    def _asia_levels(self, now: pd.Timestamp) -> list[Level]:
        sd = self._active_asia_session_day(now)
        if sd is None:
            return []
        s = self._sessions[sd]
        if s.asia_high is None or s.asia_low is None:
            return []
        return [
            Level(price=s.asia_high, kind=LevelKind.ASIA_HIGH, source_day=sd),
            Level(price=s.asia_low, kind=LevelKind.ASIA_LOW, source_day=sd),
        ]

    def _active_london_session_day(self, now: pd.Timestamp) -> Optional[date]:
        """The session day whose London is most recently locked (London.end <= now)."""
        current_sd = session_day_of(now)
        current_london_end = session_window(current_sd, Session.LONDON).end
        if now >= current_london_end and current_sd in self._sessions:
            return current_sd
        prior = current_sd - timedelta(days=1)
        if prior in self._sessions:
            return prior
        return None

    def _london_levels(self, now: pd.Timestamp) -> list[Level]:
        sd = self._active_london_session_day(now)
        if sd is None:
            return []
        s = self._sessions[sd]
        if s.london_high is None or s.london_low is None:
            return []
        return [
            Level(price=s.london_high, kind=LevelKind.LONDON_HIGH, source_day=sd),
            Level(price=s.london_low, kind=LevelKind.LONDON_LOW, source_day=sd),
        ]

    def _prior_week_levels(self, now: pd.Timestamp) -> list[Level]:
        wk = week_of(now)
        prior = wk - timedelta(days=7)
        w = self._weeks.get(prior)
        if w is None or w.high is None:
            return []
        return [
            Level(price=w.high, kind=LevelKind.PRIOR_WEEK_HIGH, source_day=prior),
            Level(price=w.low, kind=LevelKind.PRIOR_WEEK_LOW, source_day=prior),
        ]

    def _daily_open_level(self, now: pd.Timestamp) -> list[Level]:
        sd = session_day_of(now)
        if sd not in self._daily_opens:
            return []
        return [Level(price=self._daily_opens[sd], kind=LevelKind.DAILY_OPEN, source_day=sd)]

    def _weekly_open_level(self, now: pd.Timestamp) -> list[Level]:
        wk = week_of(now)
        if wk not in self._weekly_opens:
            return []
        return [Level(price=self._weekly_opens[wk], kind=LevelKind.WEEKLY_OPEN, source_day=wk)]

    def _eqhl_levels(self, swing_kind: SwingKind, level_kind: LevelKind) -> list[Level]:
        """Cluster recent confirmed swings of `swing_kind` into EQH/EQL levels.

        - Age filter: only swings whose bar_idx >= current - max_age_bars.
        - Tolerance: cluster span (max-price - min-price) <= tolerance_pts.
        - Min cluster size: 2.
        - Emitted price: mean of the cluster's swing prices.
        - Emitted source_day: session day of the cluster's most recent swing.
        """
        if self._bar_count == 0:
            return []
        current_idx = self._bar_count - 1
        min_bar_idx = current_idx - self.params.eqhl_max_age_bars
        tolerance_pts = self.params.eqhl_tolerance_ticks * NQ_TICK_SIZE

        recent = [
            s for s in self._swing_detector.all_swings
            if s.kind == swing_kind and s.bar_idx >= min_bar_idx
        ]
        clusters = _cluster_by_price(recent, tolerance_pts)
        out: list[Level] = []
        for cluster in clusters:
            avg_price = sum(s.price for s in cluster) / len(cluster)
            newest = max(cluster, key=lambda s: s.bar_idx)
            out.append(Level(
                price=avg_price,
                kind=level_kind,
                source_day=session_day_of(newest.ts),
            ))
        return out

    def _round_levels(self, price: float) -> list[Level]:
        out: list[Level] = []
        half = self.round_half_range_pts
        major_step = self.params.round_number_major_step_pts
        minor_step = self.params.round_number_minor_step_pts

        # Generate minor grid first.
        out.extend(self._grid_levels(price, half, minor_step, LevelKind.ROUND_MINOR))
        # Then majors. Drop any minor that coincides with a major (within
        # half a tick to tolerate float drift).
        major_prices = []
        major_levels = self._grid_levels(price, half, major_step, LevelKind.ROUND_MAJOR)
        major_prices = {round(l.price * 4) for l in major_levels}  # 0.25 grid
        out = [l for l in out if round(l.price * 4) not in major_prices]
        out.extend(major_levels)
        return out

    @staticmethod
    def _grid_levels(price: float, half_range: float, step: float, kind: LevelKind) -> list[Level]:
        lo = math.ceil((price - half_range) / step) * step
        hi = math.floor((price + half_range) / step) * step
        steps = int(round((hi - lo) / step))
        return [Level(price=lo + i * step, kind=kind, source_day=None) for i in range(steps + 1)]


def _cluster_by_price(swings: list[SwingPoint], tolerance_pts: float) -> list[list[SwingPoint]]:
    """Group swings whose price-range (max - min) <= tolerance_pts, min cluster size 2.

    Greedy single-linkage by sorted price. A new swing joins the current
    cluster only if including it keeps the cluster span <= tolerance_pts.
    """
    if not swings:
        return []
    by_price = sorted(swings, key=lambda s: s.price)
    clusters: list[list[SwingPoint]] = []
    current: list[SwingPoint] = [by_price[0]]
    anchor = by_price[0].price
    for s in by_price[1:]:
        if s.price - anchor <= tolerance_pts:
            current.append(s)
        else:
            if len(current) >= 2:
                clusters.append(current)
            current = [s]
            anchor = s.price
    if len(current) >= 2:
        clusters.append(current)
    return clusters
