"""Higher-timeframe (HTF) aggregation and trend classification.

Two HTF tracks for the continuation setup:
  - Daily bias: HH/HL on daily bars (1 bar per session day, 18:00 ET reset)
  - 4h HTF trend: HH/HL on 4h bars (6 buckets per session day, aligned to
    18:00 / 22:00 / 02:00 / 06:00 / 10:00 / 14:00 ET)

Trend classification: bullish if the last two confirmed swing highs AND
the last two confirmed swing lows are both increasing; bearish if both
decreasing; otherwise undefined. With swing N=2 on aggregated bars we
need at least 5 aggregated bars to see any swing, and at least ~10 for
two of each direction.

Each `HTFTrend` instance owns its own aggregator + swing detector.
Feed 1m bars in via on_bar(); query current_trend() any time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time, timedelta
from enum import Enum
from typing import Callable, Literal, Optional

import pandas as pd

from strategy.bars import Bar
from strategy.sessions import ET, session_day_of
from strategy.swings import SwingDetector, SwingKind, SwingPoint


class TrendDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    UNDEFINED = "undefined"


# ---------------------------------------------------------------------------
# Bucket-start functions: 1m bar timestamp -> HTF bucket start (UTC)
# ---------------------------------------------------------------------------


def daily_bucket_start(ts: pd.Timestamp) -> pd.Timestamp:
    """The session-day boundary (18:00 ET) for this bar's session day."""
    sd = session_day_of(ts)
    return (
        pd.Timestamp(f"{sd} 18:00:00").tz_localize(ET).tz_convert("UTC")
    )


def four_hour_bucket_start(ts: pd.Timestamp) -> pd.Timestamp:
    """4h buckets aligned to the session open at 18:00 ET.

    Buckets: 18-22, 22-02, 02-06, 06-10, 10-14, 14-18 ET.
    """
    sd = session_day_of(ts)
    session_start_et = pd.Timestamp(f"{sd} 18:00:00").tz_localize(ET)
    et = ts.tz_convert(ET)
    hours_since = (et - session_start_et).total_seconds() / 3600
    bucket = int(hours_since // 4)
    bucket_start_et = session_start_et + pd.Timedelta(hours=bucket * 4)
    return bucket_start_et.tz_convert("UTC")


# ---------------------------------------------------------------------------
# Timeframe aggregator
# ---------------------------------------------------------------------------


@dataclass
class _OpenBar:
    bucket_start: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float


class TimeframeAggregator:
    """Builds higher-timeframe Bars from incoming 1m Bars.

    Caller supplies a `bucket_fn` mapping a bar timestamp to its bucket-start
    timestamp. When a new bar arrives with a different bucket than the
    currently-open one, the open bucket is closed (returned) and a new one
    started. Bars at the bucket-start ts open the bucket; subsequent bars
    in the same bucket extend it.
    """

    def __init__(self, bucket_fn: Callable[[pd.Timestamp], pd.Timestamp]) -> None:
        self._bucket_fn = bucket_fn
        self._open: Optional[_OpenBar] = None

    def on_bar(self, bar: Bar) -> Optional[Bar]:
        """Returns the just-closed HTF bar if this bar starts a new bucket,
        else None. The current bucket is always extended by this bar."""
        bucket = self._bucket_fn(bar.ts)
        closed: Optional[Bar] = None
        if self._open is None:
            self._open = _OpenBar(bucket, bar.open, bar.high, bar.low, bar.close, bar.volume)
            return None
        if bucket != self._open.bucket_start:
            # Close the previous bucket
            closed = Bar(
                ts=self._open.bucket_start,
                open=self._open.open, high=self._open.high,
                low=self._open.low, close=self._open.close,
                volume=self._open.volume,
            )
            self._open = _OpenBar(bucket, bar.open, bar.high, bar.low, bar.close, bar.volume)
            return closed
        # Same bucket: extend
        self._open.high = max(self._open.high, bar.high)
        self._open.low = min(self._open.low, bar.low)
        self._open.close = bar.close
        self._open.volume += bar.volume
        return None

    def flush(self) -> Optional[Bar]:
        """Close out the current open bucket without waiting for a new one.
        For end-of-stream finalization."""
        if self._open is None:
            return None
        closed = Bar(
            ts=self._open.bucket_start,
            open=self._open.open, high=self._open.high,
            low=self._open.low, close=self._open.close,
            volume=self._open.volume,
        )
        self._open = None
        return closed


# ---------------------------------------------------------------------------
# HTF trend tracker
# ---------------------------------------------------------------------------


class HTFTrend:
    def __init__(self, period: Literal["daily", "4h"], swing_n: int = 2) -> None:
        if period == "daily":
            bucket_fn = daily_bucket_start
        elif period == "4h":
            bucket_fn = four_hour_bucket_start
        else:
            raise ValueError(f"Unknown period: {period!r}")
        self.period = period
        self._aggregator = TimeframeAggregator(bucket_fn)
        self._swing_detector = SwingDetector(n=swing_n)
        self._htf_bar_count = 0

    def on_bar(self, bar: Bar) -> None:
        closed = self._aggregator.on_bar(bar)
        if closed is not None:
            self._swing_detector.on_bar(closed)
            self._htf_bar_count += 1

    @property
    def htf_bar_count(self) -> int:
        """Number of completed HTF bars seen so far."""
        return self._htf_bar_count

    @property
    def swing_highs(self) -> list[SwingPoint]:
        """Confirmed HTF swing highs, chronological. Used by the continuation
        setup to pick a target level in the trend direction."""
        return self._swing_detector.highs

    @property
    def swing_lows(self) -> list[SwingPoint]:
        """Confirmed HTF swing lows, chronological."""
        return self._swing_detector.lows

    def current_trend(self) -> TrendDirection:
        """Bullish if last two confirmed swing highs AND lows are both rising;
        bearish if both falling; else undefined."""
        highs = self._swing_detector.highs
        lows = self._swing_detector.lows
        if len(highs) < 2 or len(lows) < 2:
            return TrendDirection.UNDEFINED
        h_rising = highs[-1].price > highs[-2].price
        l_rising = lows[-1].price > lows[-2].price
        h_falling = highs[-1].price < highs[-2].price
        l_falling = lows[-1].price < lows[-2].price
        if h_rising and l_rising:
            return TrendDirection.BULLISH
        if h_falling and l_falling:
            return TrendDirection.BEARISH
        return TrendDirection.UNDEFINED
