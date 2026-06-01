"""Liquidity sweep detector.

A sweep has three components:

  1. Penetration   bar's wick crosses a reference level by 3-8 ticks
                   (configurable via params.penetration_min_ticks /
                   penetration_max_valid_ticks).
  2. Close-back    a bar (the same one, or one of the next
                   max_rejection_bars) closes back on the original side.
  3. Volume conf.  the rejection bar's volume >= volume_mult * SMA(volume)
                   over the trailing volume_window bars.

Penetrations deeper than penetration_broken_ticks (>10) mean the level
was broken, not swept — we mark the level resolved and stop watching it.

A "fresh" penetration requires the bar to open on the original side of
the level (bar.open <= level.price for an up-sweep, >= for a down-sweep).
This eliminates false signals where price was already on the other side
of the level and just kept drifting.

Once a sweep fires on a level (or the level is marked broken), the level
is recorded as resolved and ignored on subsequent bars. Level identity:
(kind, source_day, rounded_price).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd

from strategy.bars import Bar
from strategy.levels import Level, LevelKind
from strategy.params import NQ_TICK_SIZE, StrategyParams


class SweepDirection(Enum):
    UP = "up"      # wick punched above resistance, then closed back below
    DOWN = "down"  # wick punched below support, then closed back above


@dataclass(frozen=True)
class SweepEvent:
    level: Level
    direction: SweepDirection
    penetration_bar_idx: int  # bar where the wick first crossed the level
    rejection_bar_idx: int    # bar that closed back inside (>= penetration)
    wick_extreme: float       # furthest point the wick reached (used for stops)
    penetration_ticks: int    # tick depth at the wick extreme
    volume_ratio: float       # rejection bar volume / trailing SMA
    ts: pd.Timestamp          # rejection bar's timestamp


@dataclass
class _Pending:
    """Penetration awaiting close-back confirmation."""
    level: Level
    direction: SweepDirection
    penetration_bar_idx: int
    wick_extreme: float


def _level_key(level: Level) -> tuple:
    """Hashable identity for a Level. Source_day=None for round numbers etc.
    is preserved so identical-priced round numbers across sessions share a key."""
    return (level.kind, level.source_day, round(level.price, 4))


def _is_fresh_up_penetration(bar: Bar, level_price: float) -> bool:
    """The bar wicked above the level, having opened at or below it."""
    return bar.high > level_price and bar.open <= level_price


def _is_fresh_down_penetration(bar: Bar, level_price: float) -> bool:
    return bar.low < level_price and bar.open >= level_price


def _ticks_above(price: float, level_price: float) -> int:
    return int(round((price - level_price) / NQ_TICK_SIZE))


def _ticks_below(price: float, level_price: float) -> int:
    return int(round((level_price - price) / NQ_TICK_SIZE))


class SweepDetector:
    def __init__(self, params: StrategyParams) -> None:
        self.params = params
        self._bar_count = 0
        # Trailing volume history. We keep last `volume_window` values.
        self._volume_history: list[float] = []
        self._pending: list[_Pending] = []
        self._resolved: set[tuple] = set()  # level_key -> done (swept or broken)

    def on_bar(self, bar: Bar, active_levels: list[Level]) -> list[SweepEvent]:
        """Advance state and emit any newly confirmed sweeps."""
        bar_idx = self._bar_count
        self._bar_count += 1

        events: list[SweepEvent] = []

        # 1. Update pending candidates with this new bar.
        still_pending: list[_Pending] = []
        for p in self._pending:
            outcome = self._update_pending(p, bar, bar_idx)
            if outcome is None:
                still_pending.append(p)  # keep waiting
            elif isinstance(outcome, SweepEvent):
                events.append(outcome)
                self._resolved.add(_level_key(p.level))
            else:
                # outcome == "broken" or "expired" — drop, optionally mark resolved
                if outcome == "broken":
                    self._resolved.add(_level_key(p.level))
        self._pending = still_pending

        # 2. Check this bar for fresh penetrations on each active level.
        for level in active_levels:
            key = _level_key(level)
            if key in self._resolved:
                continue
            # An UP sweep wicks above a level from below.
            if _is_fresh_up_penetration(bar, level.price):
                depth = _ticks_above(bar.high, level.price)
                if depth > self.params.penetration_broken_ticks:
                    self._resolved.add(key)
                    continue
                if self.params.penetration_min_ticks <= depth <= self.params.penetration_max_valid_ticks:
                    ev = self._evaluate_new_candidate(
                        bar, bar_idx, level, SweepDirection.UP,
                        wick_extreme=bar.high, depth=depth,
                    )
                    if ev is not None:
                        events.append(ev)
                        self._resolved.add(key)
                # depths 9-10 are an ambiguous "deep" zone: not a valid sweep
                # candidate, but level not yet broken either. Skip silently.

            # A DOWN sweep wicks below a level from above.
            if _is_fresh_down_penetration(bar, level.price):
                depth = _ticks_below(bar.low, level.price)
                if depth > self.params.penetration_broken_ticks:
                    self._resolved.add(key)
                    continue
                if self.params.penetration_min_ticks <= depth <= self.params.penetration_max_valid_ticks:
                    ev = self._evaluate_new_candidate(
                        bar, bar_idx, level, SweepDirection.DOWN,
                        wick_extreme=bar.low, depth=depth,
                    )
                    if ev is not None:
                        events.append(ev)
                        self._resolved.add(key)

        # 3. Update volume history AFTER we've checked confirmation against
        # the prior trailing window (current bar's volume excluded from SMA).
        self._volume_history.append(bar.volume)
        if len(self._volume_history) > self.params.volume_window + 1:
            self._volume_history.pop(0)

        return events

    # ---- helpers -----------------------------------------------------------

    def _trailing_volume_sma(self) -> Optional[float]:
        """SMA of the last `volume_window` bars, EXCLUDING the current bar.

        Returns None until enough history exists.
        """
        history = self._volume_history  # current bar's volume not yet appended
        if len(history) < self.params.volume_window:
            return None
        window = history[-self.params.volume_window:]
        return sum(window) / len(window)

    def _evaluate_new_candidate(
        self, bar: Bar, bar_idx: int, level: Level, direction: SweepDirection,
        *, wick_extreme: float, depth: int,
    ) -> Optional[SweepEvent]:
        """Check if this fresh penetration is also a same-bar sweep.

        If the bar's close is already back inside the level AND volume
        confirms, fire the sweep on this bar. Otherwise return None and
        queue as pending.
        """
        closed_inside = (
            bar.close <= level.price if direction == SweepDirection.UP
            else bar.close >= level.price
        )
        if closed_inside:
            ratio = self._volume_ratio(bar)
            if ratio is not None and ratio >= self.params.volume_mult:
                return SweepEvent(
                    level=level, direction=direction,
                    penetration_bar_idx=bar_idx, rejection_bar_idx=bar_idx,
                    wick_extreme=wick_extreme, penetration_ticks=depth,
                    volume_ratio=ratio, ts=bar.ts,
                )
            # Closed inside but volume insufficient — candidate is finished
            # (no further bars can confirm a same-bar sweep). Discard silently.
            return None

        # Closed outside — pending until close-back occurs.
        self._pending.append(_Pending(
            level=level, direction=direction,
            penetration_bar_idx=bar_idx, wick_extreme=wick_extreme,
        ))
        return None

    def _update_pending(self, p: _Pending, bar: Bar, bar_idx: int):
        """Advance one pending candidate. Returns:
            - SweepEvent if confirmed
            - "broken" if the wick extended past the broken threshold
            - "expired" if max_rejection_bars elapsed
            - None to keep waiting
        """
        # Update wick extreme if price reached further this bar.
        if p.direction == SweepDirection.UP:
            p.wick_extreme = max(p.wick_extreme, bar.high)
            depth = _ticks_above(p.wick_extreme, p.level.price)
        else:
            p.wick_extreme = min(p.wick_extreme, bar.low)
            depth = _ticks_below(p.wick_extreme, p.level.price)

        if depth > self.params.penetration_broken_ticks:
            return "broken"

        # Check close-back-inside
        closed_inside = (
            bar.close <= p.level.price if p.direction == SweepDirection.UP
            else bar.close >= p.level.price
        )
        if closed_inside:
            ratio = self._volume_ratio(bar)
            if ratio is not None and ratio >= self.params.volume_mult:
                return SweepEvent(
                    level=p.level, direction=p.direction,
                    penetration_bar_idx=p.penetration_bar_idx,
                    rejection_bar_idx=bar_idx,
                    wick_extreme=p.wick_extreme,
                    penetration_ticks=depth,
                    volume_ratio=ratio, ts=bar.ts,
                )
            # Insufficient volume; treat as expired (no valid sweep)
            return "expired"

        # Still penetrating. Has the window elapsed?
        if bar_idx - p.penetration_bar_idx >= self.params.max_rejection_bars:
            return "expired"
        return None

    def _volume_ratio(self, bar: Bar) -> Optional[float]:
        sma = self._trailing_volume_sma()
        if sma is None or sma == 0:
            return None
        return bar.volume / sma
