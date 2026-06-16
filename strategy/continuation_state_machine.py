"""Continuation-strategy state machine (module 12).

A trend-following setup that runs **in parallel** to the sweep state machine
(they never share state). The idea, in the user's words: "follow the trend,
look for entries off a retest of the midpoint of VWAP." Session VWAP is the
intraday midpoint; in an HTF uptrend we wait for price to pull back to VWAP
and resume up. Mirror for a downtrend.

Lifecycle per setup:

    IDLE
      | HTF trend matches a direction AND price closes on the trend side
      | of VWAP (arms a long in an uptrend / a short in a downtrend)
      v
    WATCHING_FOR_RETEST       (held while the HTF trend persists)
      | a later bar pulls back and touches VWAP, with the required
      | confluence on that bar: buying/selling pressure, absorption at
      | VWAP, an active FVG straddling VWAP (each gate toggleable via params)
      v
    PENDING_ENTRY             (one bar; fills next-bar-open)
      v
    IN_TRADE                  (stop / target / HTF-flip watch)
      | stop hit | target hit | HTF trend flips against us
      v
    PENDING_EXIT              (one bar; fills next-bar-open)
      v
    IDLE (Trade emitted, setup_kind="continuation")

Stop: just beyond the most recent LTF swing on the entry side (swing low for
a long, swing high for a short), buffered by stop_buffer_ticks — i.e. the
swing that formed into the VWAP retest.
Target: the nearest HTF swing level in the trend direction that clears the
min_rr filter (a real structural level, not a synthetic R multiple). If no
visible HTF level clears R:R, the setup is skipped — no trade.

Execution model: signals fire on bar T's close; fills at bar T+1 open —
the same strict next-bar rule the sweep machine uses.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from strategy.bars import Bar
from strategy.fvg import FVGDetector, FVGKind, FVGState
from strategy.htf import HTFTrend, TrendDirection
from strategy.params import NQ_TICK_SIZE, StrategyParams
from strategy.pressure import AbsorptionDetector, PressureDetector
from strategy.sweep_state_machine import Trade
from strategy.swings import SwingDetector, SwingKind
from strategy.vwap import SessionVWAP


class ContState(Enum):
    IDLE = "idle"
    WATCHING_FOR_RETEST = "watching_for_retest"
    PENDING_ENTRY = "pending_entry"
    IN_TRADE = "in_trade"
    PENDING_EXIT = "pending_exit"


@dataclass
class _Cont:
    bias: str                 # "long" (uptrend) or "short" (downtrend)
    armed_bar_idx: int
    # Set when a retest qualifies:
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    rr_at_entry: Optional[float] = None
    # Open trade:
    entry_bar_idx: Optional[int] = None
    entry_price: Optional[float] = None
    # Pending exit:
    pending_exit_reason: Optional[str] = None


class ContinuationStateMachine:
    def __init__(self, params: StrategyParams) -> None:
        self.params = params
        self._state = ContState.IDLE
        self._setup: Optional[_Cont] = None
        self._bar_count = 0
        # Own sub-detectors — fully independent of the sweep machine.
        self._vwap = SessionVWAP()
        self._htf = HTFTrend(period=params.cont_htf_period)
        self._pressure = PressureDetector(params)
        self._absorption = AbsorptionDetector(params)
        self._fvg = FVGDetector()
        self._swings = SwingDetector(n=params.ltf_swing_lookback)
        self._closed_trades: list[Trade] = []

    @property
    def state(self) -> ContState:
        return self._state

    @property
    def closed_trades(self) -> list[Trade]:
        return list(self._closed_trades)

    def on_bar(self, bar: Bar) -> list[Trade]:
        """Advance one bar. Returns Trades that closed on this bar (0 or 1)."""
        bar_idx = self._bar_count
        self._bar_count += 1

        # Advance every sub-detector exactly once per bar (causal, no lookahead).
        self._vwap.on_bar(bar)
        self._htf.on_bar(bar)
        self._fvg.on_bar(bar)
        self._swings.on_bar(bar)
        vwap = self._vwap.current
        pressure = self._pressure.on_bar(bar)
        absorption = self._absorption.on_bar(bar, near_level=vwap)

        newly_closed: list[Trade] = []

        # PENDING_* states consume a full bar — they are the next-bar-open fill
        # for a signal that fired on the previous bar.
        if self._state == ContState.PENDING_ENTRY:
            self._pending_entry_step(bar, bar_idx)
            return newly_closed
        if self._state == ContState.PENDING_EXIT:
            trade = self._pending_exit_step(bar, bar_idx)
            if trade is not None:
                newly_closed.append(trade)
                self._closed_trades.append(trade)
            return newly_closed

        trend = self._htf.current_trend()

        if self._state == ContState.IDLE:
            self._idle_step(bar, bar_idx, trend, vwap)
        elif self._state == ContState.WATCHING_FOR_RETEST:
            self._watching_step(bar, bar_idx, trend, vwap, pressure, absorption)
        elif self._state == ContState.IN_TRADE:
            self._in_trade_step(bar, bar_idx, trend)

        return newly_closed

    # ---- per-state handlers ----------------------------------------------

    def _idle_step(self, bar, bar_idx, trend, vwap) -> None:
        if vwap is None:
            return
        # Arm a directional bias only when the HTF trend agrees with price's
        # posture relative to VWAP.
        if trend == TrendDirection.BULLISH and bar.close > vwap:
            self._setup = _Cont(bias="long", armed_bar_idx=bar_idx)
            self._state = ContState.WATCHING_FOR_RETEST
        elif trend == TrendDirection.BEARISH and bar.close < vwap:
            self._setup = _Cont(bias="short", armed_bar_idx=bar_idx)
            self._state = ContState.WATCHING_FOR_RETEST

    def _watching_step(self, bar, bar_idx, trend, vwap, pressure, absorption) -> None:
        setup = self._setup
        if vwap is None:
            return
        # Abandon if the HTF trend no longer supports our bias.
        if setup.bias == "long" and trend != TrendDirection.BULLISH:
            self._reset()
            return
        if setup.bias == "short" and trend != TrendDirection.BEARISH:
            self._reset()
            return
        # Don't evaluate a retest on the arming bar itself.
        if bar_idx <= setup.armed_bar_idx:
            return

        # Retest = price pulls back and touches VWAP from the trend side.
        if setup.bias == "long":
            touched = bar.low <= vwap
        else:
            touched = bar.high >= vwap
        if not touched:
            return

        if not self._confluence_ok(setup.bias, vwap, pressure, absorption):
            return

        # Place the stop beyond the most recent LTF swing on the entry side.
        stop_price = self._stop_price(setup.bias, bar_idx)
        if stop_price is None:
            return
        entry_est = bar.close  # provisional; real fill is next-bar open
        risk = (entry_est - stop_price) if setup.bias == "long" else (stop_price - entry_est)
        if risk <= 0:
            return

        target_price = self._pick_target(setup.bias, entry_est, risk)
        if target_price is None:
            return

        reward = abs(target_price - entry_est)
        setup.stop_price = stop_price
        setup.target_price = target_price
        setup.rr_at_entry = reward / risk
        self._state = ContState.PENDING_ENTRY

    def _pending_entry_step(self, bar, bar_idx) -> None:
        setup = self._setup
        entry = bar.open  # next-bar-open fill
        risk = (entry - setup.stop_price) if setup.bias == "long" else (setup.stop_price - entry)
        reward = abs(setup.target_price - entry)
        if risk <= 0 or reward / risk < self.params.min_rr_ratio:
            # R:R degraded on the realized fill — abandon.
            self._reset()
            return
        setup.entry_bar_idx = bar_idx
        setup.entry_price = entry
        setup.rr_at_entry = reward / risk
        self._state = ContState.IN_TRADE

    def _in_trade_step(self, bar, bar_idx, trend) -> None:
        setup = self._setup
        reason: Optional[str] = None
        if setup.bias == "long":
            if bar.low <= setup.stop_price:
                reason = "stop"
            elif bar.high >= setup.target_price:
                reason = "target"
            elif trend == TrendDirection.BEARISH:
                reason = "invalidated"
        else:
            if bar.high >= setup.stop_price:
                reason = "stop"
            elif bar.low <= setup.target_price:
                reason = "target"
            elif trend == TrendDirection.BULLISH:
                reason = "invalidated"
        if reason is not None:
            setup.pending_exit_reason = reason
            self._state = ContState.PENDING_EXIT

    def _pending_exit_step(self, bar, bar_idx) -> Optional[Trade]:
        setup = self._setup
        exit_price = bar.open  # next-bar-open fill
        if setup.bias == "long":
            pnl = exit_price - setup.entry_price
            direction = "long"
        else:
            pnl = setup.entry_price - exit_price
            direction = "short"
        trade = Trade(
            setup_kind="continuation",
            direction=direction,
            entry_bar_idx=setup.entry_bar_idx,
            entry_price=setup.entry_price,
            stop_price=setup.stop_price,
            target_price=setup.target_price,
            exit_bar_idx=bar_idx,
            exit_price=exit_price,
            exit_reason=setup.pending_exit_reason,
            pnl_points=pnl,
            rr_at_entry=setup.rr_at_entry,
        )
        self._reset()
        return trade

    # ---- helpers ---------------------------------------------------------

    def _reset(self) -> None:
        self._state = ContState.IDLE
        self._setup = None

    def _confluence_ok(self, bias, vwap, pressure, absorption) -> bool:
        p = self.params
        if p.cont_require_pressure:
            ok = pressure.buying_pressure if bias == "long" else pressure.selling_pressure
            if not ok:
                return False
        if p.cont_require_absorption and not absorption.absorption:
            return False
        if p.cont_require_fvg_at_vwap and not self._fvg_at_vwap(bias, vwap):
            return False
        return True

    def _fvg_at_vwap(self, bias, vwap) -> bool:
        want = FVGKind.BULLISH if bias == "long" else FVGKind.BEARISH
        for f in self._fvg.active_fvgs():
            if f.kind == want and f.contains_price(vwap):
                return True
        return False

    def _stop_price(self, bias, bar_idx) -> Optional[float]:
        buf = self.params.stop_buffer_ticks * NQ_TICK_SIZE
        if bias == "long":
            swing = self._swings.most_recent(SwingKind.LOW, before_confirmation_idx=bar_idx)
            if swing is None:
                return None
            return swing.price - buf
        swing = self._swings.most_recent(SwingKind.HIGH, before_confirmation_idx=bar_idx)
        if swing is None:
            return None
        return swing.price + buf

    def _pick_target(self, bias, entry_est, risk) -> Optional[float]:
        """Nearest HTF swing level in the trend direction that clears min_rr."""
        min_rr = self.params.min_rr_ratio
        if bias == "long":
            best: Optional[float] = None
            for s in self._htf.swing_highs:
                if s.price <= entry_est:
                    continue
                if (s.price - entry_est) / risk < min_rr:
                    continue
                if best is None or s.price < best:
                    best = s.price
            return best
        best = None
        for s in self._htf.swing_lows:
            if s.price >= entry_est:
                continue
            if (entry_est - s.price) / risk < min_rr:
                continue
            if best is None or s.price > best:
                best = s.price
        return best
