"""Session-anchored Volume-Weighted Average Price (VWAP).

Standard intraday VWAP: cumulative sum of (typical_price * volume) divided
by cumulative volume, reset at session open. We use the futures session
boundary — 18:00 ET — as the reset point. That matches our session-day
attribution everywhere else in the codebase.

The continuation setup uses VWAP as the retracement zone: in a bullish
HTF regime we look for price to pull back to (or below) VWAP and resume
upward; in a bearish regime, the mirror.

This module is pure and incremental — `on_bar(bar)` per bar, `current`
property exposes the running VWAP. None before the first bar of a session.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from strategy.bars import Bar
from strategy.sessions import session_day_of


class SessionVWAP:
    """Tracks one VWAP value, which resets at each session-day boundary."""

    def __init__(self) -> None:
        self._current_session_day: Optional[date] = None
        self._cum_pv: float = 0.0  # sum(typical * volume)
        self._cum_v: float = 0.0   # sum(volume)

    def on_bar(self, bar: Bar) -> None:
        sd = session_day_of(bar.ts)
        if sd != self._current_session_day:
            # Boundary crossed: start fresh.
            self._cum_pv = 0.0
            self._cum_v = 0.0
            self._current_session_day = sd
        typical = (bar.high + bar.low + bar.close) / 3.0
        self._cum_pv += typical * bar.volume
        self._cum_v += bar.volume

    @property
    def current(self) -> Optional[float]:
        """Current session VWAP. None until at least one bar with volume > 0."""
        if self._cum_v <= 0:
            return None
        return self._cum_pv / self._cum_v

    @property
    def current_session_day(self) -> Optional[date]:
        return self._current_session_day
