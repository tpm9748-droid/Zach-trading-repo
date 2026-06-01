"""Fair Value Gap (FVG) and Order Block (OB) detection.

Fair Value Gap (a.k.a. price imbalance): a 3-bar pattern where the wick
of bar 1 and the wick of bar 3 do not overlap, leaving a price range that
was traversed without two-sided trade. Formally:

    Bullish FVG (created by an up-impulse):
        low[c3] > high[c1]           gap range = [high[c1], low[c3]]
    Bearish FVG (created by a down-impulse):
        high[c3] < low[c1]           gap range = [high[c3], low[c1]]

We track every FVG that forms and its state over subsequent bars:

    ACTIVE   no bar has touched the gap range yet
    PARTIAL  some bar touched the range but didn't close fully through
    FILLED   a bar's close passed all the way through the gap
    INVERTED the gap was filled and later acted as opposite-polarity S/R
             (state machine sets this; we don't auto-promote here in v1)

Order Block: the last opposing-polarity candle immediately before a
displacement leg starts. After a bullish CHoCH (price reversed up), the
OB is the last bearish candle before the impulse-up; the OB's range is
expected to act as support on retest.

These detectors are pure / append-only: no lookahead. State updates on
existing FVGs use only the current bar's data.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd

from strategy.bars import Bar


class FVGKind(Enum):
    BULLISH = "bullish"  # gap below an up-impulse
    BEARISH = "bearish"  # gap above a down-impulse


class FVGState(Enum):
    ACTIVE = "active"
    PARTIAL = "partial"
    FILLED = "filled"
    INVERTED = "inverted"


@dataclass
class FVG:
    """Fair value gap. State is mutable; created_at_idx / kind / bounds are not."""
    kind: FVGKind
    upper: float           # higher edge of the gap
    lower: float           # lower edge of the gap
    created_at_idx: int    # bar index of c3 (the bar whose arrival made the FVG visible)
    ts: pd.Timestamp       # c3 timestamp
    state: FVGState = FVGState.ACTIVE
    first_touched_at_idx: Optional[int] = None
    filled_at_idx: Optional[int] = None
    inverted_at_idx: Optional[int] = None

    @property
    def size(self) -> float:
        return self.upper - self.lower

    @property
    def midpoint(self) -> float:
        return (self.upper + self.lower) / 2

    def contains_price(self, price: float) -> bool:
        return self.lower <= price <= self.upper


class FVGDetector:
    """Incremental FVG detector. Feed every bar via on_bar.

    Returns the newly-created FVG (if one formed on the trailing 3 bars)
    each call, else None. Independently, ACTIVE/PARTIAL FVGs have their
    state advanced based on the new bar.
    """

    def __init__(self) -> None:
        self._bars: list[Bar] = []
        self._fvgs: list[FVG] = []
        self._bar_count = 0

    @property
    def all_fvgs(self) -> list[FVG]:
        return list(self._fvgs)

    def active_fvgs(self) -> list[FVG]:
        """ACTIVE or PARTIAL (not yet fully filled)."""
        return [f for f in self._fvgs if f.state in (FVGState.ACTIVE, FVGState.PARTIAL)]

    def find_in_range(
        self,
        start_idx: int,
        end_idx: int,
        kind: Optional[FVGKind] = None,
    ) -> list[FVG]:
        """FVGs whose creation index (c3) is in [start_idx, end_idx]."""
        out = []
        for f in self._fvgs:
            if not (start_idx <= f.created_at_idx <= end_idx):
                continue
            if kind is not None and f.kind != kind:
                continue
            out.append(f)
        return out

    def on_bar(self, bar: Bar) -> Optional[FVG]:
        self._bars.append(bar)
        current_idx = self._bar_count
        self._bar_count += 1

        # Advance state of existing non-terminal FVGs based on this bar.
        for f in self._fvgs:
            if f.state in (FVGState.FILLED, FVGState.INVERTED):
                continue
            self._update_state(f, bar, current_idx)

        # Check for a newly formed FVG on the trailing 3 bars.
        if len(self._bars) < 3:
            return None
        c1, c2, c3 = self._bars[-3], self._bars[-2], self._bars[-1]
        new_fvg = _detect(c1, c2, c3, created_at_idx=current_idx, ts=c3.ts)
        if new_fvg is not None:
            self._fvgs.append(new_fvg)
        return new_fvg

    @staticmethod
    def _update_state(f: FVG, bar: Bar, bar_idx: int) -> None:
        if f.kind == FVGKind.BULLISH:
            # Filled when a bar closes below the lower edge (price punched all
            # the way through the gap from above).
            if bar.close <= f.lower:
                f.state = FVGState.FILLED
                if f.filled_at_idx is None:
                    f.filled_at_idx = bar_idx
                return
            # Touched the range but didn't fill -> PARTIAL.
            if bar.low <= f.upper and f.state == FVGState.ACTIVE:
                f.state = FVGState.PARTIAL
                f.first_touched_at_idx = bar_idx
        else:  # BEARISH
            if bar.close >= f.upper:
                f.state = FVGState.FILLED
                if f.filled_at_idx is None:
                    f.filled_at_idx = bar_idx
                return
            if bar.high >= f.lower and f.state == FVGState.ACTIVE:
                f.state = FVGState.PARTIAL
                f.first_touched_at_idx = bar_idx


def _detect(c1: Bar, c2: Bar, c3: Bar, *, created_at_idx: int, ts: pd.Timestamp) -> Optional[FVG]:
    """Return an FVG if c1/c2/c3 form one, else None."""
    if c3.low > c1.high:
        return FVG(
            kind=FVGKind.BULLISH,
            upper=c3.low,
            lower=c1.high,
            created_at_idx=created_at_idx,
            ts=ts,
        )
    if c3.high < c1.low:
        return FVG(
            kind=FVGKind.BEARISH,
            upper=c1.low,
            lower=c3.high,
            created_at_idx=created_at_idx,
            ts=ts,
        )
    return None


# ---------------------------------------------------------------------------
# Order block
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OrderBlock:
    """Last opposing-polarity candle before a displacement leg.

    `direction` is the direction of the displacement, NOT the candle's own
    polarity. A bullish displacement (price moved up) yields a bullish OB
    whose underlying candle is bearish (the last down-close before the
    up-thrust). Same idea for bearish.
    """
    direction: FVGKind     # reuse: BULLISH or BEARISH displacement
    bar_idx: int
    ts: pd.Timestamp
    upper: float           # candle's high
    lower: float           # candle's low

    @property
    def size(self) -> float:
        return self.upper - self.lower


def find_order_block(
    bars: list[Bar],
    displacement_start_idx: int,
    displacement_direction: FVGKind,
    max_lookback: int = 50,
) -> Optional[OrderBlock]:
    """Find the OB by walking backwards from displacement_start_idx - 1.

    Returns the first candle whose polarity is opposite to the displacement
    direction, within `max_lookback` bars. Bars with close == open (doji)
    don't qualify as either polarity and are skipped.

    bars[displacement_start_idx] should be the first bar of the displacement
    leg itself. The OB is *before* that bar.
    """
    if displacement_direction == FVGKind.BULLISH:
        needed_is_bearish = True
    else:
        needed_is_bearish = False

    start = displacement_start_idx - 1
    stop = max(-1, start - max_lookback)
    for i in range(start, stop, -1):
        if i < 0 or i >= len(bars):
            continue
        b = bars[i]
        if b.is_bearish and needed_is_bearish:
            return OrderBlock(
                direction=displacement_direction, bar_idx=i, ts=b.ts,
                upper=b.high, lower=b.low,
            )
        if b.is_bullish and not needed_is_bearish:
            return OrderBlock(
                direction=displacement_direction, bar_idx=i, ts=b.ts,
                upper=b.high, lower=b.low,
            )
    return None
