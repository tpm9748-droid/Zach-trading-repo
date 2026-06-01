"""Swing points and Change-of-Character (CHoCH) detection.

A swing point is a fractal: bar at position i is a swing **high** if its
high is strictly greater than the highs of N bars on its left and N bars
on its right. Symmetrically for swing **low** using lows. With N=5 we use
this for HTF structure (later: equal-highs clusters). With N=3 we use this
for LTF structure (CHoCH after a sweep).

Confirmation lag: a swing at bar i cannot be confirmed until bar i+N has
been processed, because we need N bars to its right. This module enforces
that — `on_bar` never returns a swing earlier than is causally possible.

CHoCH (Change of Character): after a directional move, the first time
price breaks the *opposite-direction* most recent confirmed swing. After
a sweep-up of a resistance level we look for CHoCH-down: price closing
below the most recent confirmed swing **low**. "Break" can be defined by
the bar's close (default, stricter) or its wick (more permissive). The
mode is configurable via `BreakMode`.

This module is pure: it doesn't manage trade state. The strategy state
machine (module 6) decides *when* to start watching for a CHoCH, *what
direction*, and *for how many bars*. Here we just provide the primitives.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional

import pandas as pd

from strategy.bars import Bar


class SwingKind(Enum):
    HIGH = "high"
    LOW = "low"


class BreakMode(Enum):
    CLOSE = "close"  # bar's close crosses the level
    WICK = "wick"    # bar's wick (high/low) crosses the level


@dataclass(frozen=True)
class SwingPoint:
    """A confirmed fractal swing.

    bar_idx is the index of the swing bar in the feed order (0-based; matches
    the engine cursor when the detector is fed every bar from the start).
    confirmed_at_idx is the index of the bar whose arrival confirmed the
    swing — always bar_idx + N.
    """
    bar_idx: int
    confirmed_at_idx: int
    ts: pd.Timestamp
    price: float
    kind: SwingKind


class SwingDetector:
    """Incremental fractal swing detector.

    Call on_bar(bar) for every bar in order. Returns a newly-confirmed
    SwingPoint if the bar at position (current - N) qualifies; else None.

    Strict-greater rule: ties on the outer bars disqualify the candidate.
    That avoids registering every consecutive equal high as a swing. If real
    data produces too few swings under this rule we'll relax to >= on one
    side.
    """

    def __init__(self, n: int) -> None:
        if n < 1:
            raise ValueError("swing lookback n must be >= 1")
        self.n = n
        self._bars: list[Bar] = []
        self._swings: list[SwingPoint] = []

    @property
    def all_swings(self) -> list[SwingPoint]:
        return list(self._swings)

    @property
    def highs(self) -> list[SwingPoint]:
        return [s for s in self._swings if s.kind == SwingKind.HIGH]

    @property
    def lows(self) -> list[SwingPoint]:
        return [s for s in self._swings if s.kind == SwingKind.LOW]

    def on_bar(self, bar: Bar) -> Optional[SwingPoint]:
        self._bars.append(bar)
        # Need at least 2N+1 bars total to evaluate a candidate at position N
        # (bars [0..2N], candidate at N).
        if len(self._bars) < 2 * self.n + 1:
            return None

        confirming_idx = len(self._bars) - 1
        candidate_idx = confirming_idx - self.n
        candidate = self._bars[candidate_idx]
        left = self._bars[candidate_idx - self.n : candidate_idx]
        right = self._bars[candidate_idx + 1 : confirming_idx + 1]

        # Swing high: candidate strictly greater than all neighbor highs
        if all(candidate.high > b.high for b in left) and all(candidate.high > b.high for b in right):
            sp = SwingPoint(
                bar_idx=candidate_idx,
                confirmed_at_idx=confirming_idx,
                ts=candidate.ts,
                price=candidate.high,
                kind=SwingKind.HIGH,
            )
            self._swings.append(sp)
            return sp

        # Swing low: candidate strictly less than all neighbor lows
        if all(candidate.low < b.low for b in left) and all(candidate.low < b.low for b in right):
            sp = SwingPoint(
                bar_idx=candidate_idx,
                confirmed_at_idx=confirming_idx,
                ts=candidate.ts,
                price=candidate.low,
                kind=SwingKind.LOW,
            )
            self._swings.append(sp)
            return sp

        return None

    def most_recent(
        self,
        kind: SwingKind,
        before_bar_idx: Optional[int] = None,
        before_confirmation_idx: Optional[int] = None,
    ) -> Optional[SwingPoint]:
        """Most recent confirmed swing of `kind`.

        - before_bar_idx: restrict to swings whose bar is strictly before this index
        - before_confirmation_idx: restrict to swings confirmed at or before this index
          (i.e., swings the engine could have known about by then)

        Use before_confirmation_idx when asking "what swings did the engine
        know about at bar t?" — pass t.
        """
        result: Optional[SwingPoint] = None
        for s in self._swings:
            if s.kind != kind:
                continue
            if before_bar_idx is not None and s.bar_idx >= before_bar_idx:
                continue
            if before_confirmation_idx is not None and s.confirmed_at_idx > before_confirmation_idx:
                continue
            result = s  # swings list is append-only in chronological order
        return result


# ---------------------------------------------------------------------------
# CHoCH break helpers
# ---------------------------------------------------------------------------

def broke_swing_down(bar: Bar, swing: SwingPoint, mode: BreakMode = BreakMode.CLOSE) -> bool:
    """Did this bar break a swing **low** downward?"""
    if swing.kind != SwingKind.LOW:
        raise ValueError(f"broke_swing_down requires a LOW swing, got {swing.kind}")
    if mode == BreakMode.CLOSE:
        return bar.close < swing.price
    return bar.low < swing.price


def broke_swing_up(bar: Bar, swing: SwingPoint, mode: BreakMode = BreakMode.CLOSE) -> bool:
    """Did this bar break a swing **high** upward?"""
    if swing.kind != SwingKind.HIGH:
        raise ValueError(f"broke_swing_up requires a HIGH swing, got {swing.kind}")
    if mode == BreakMode.CLOSE:
        return bar.close > swing.price
    return bar.high > swing.price


@dataclass(frozen=True)
class CHoCH:
    """A change-of-character event."""
    direction: str  # "down" (broke a swing low) or "up" (broke a swing high)
    bar_idx: int
    ts: pd.Timestamp
    broken_swing: SwingPoint
