"""Sweep-strategy state machine.

Wires together: sweep events -> CHoCH watch -> OB/FVG identification ->
entry trigger -> stop/target management -> trade closure.

Lifecycle per setup:

    IDLE
      | sweep event arrives
      v
    WATCHING_FOR_CHOCH       (window: params.max_choch_bars bars)
      | LTF close breaks the opposing pre-sweep swing
      v
    WATCHING_FOR_ENTRY       (window: params.max_entry_bars bars)
      | bar first touches OB lower edge (short) / OB upper edge (long)
      | or first touches any unfilled FVG inside the displacement leg
      v
    PENDING_ENTRY            (one bar; fills next-bar-open)
      v
    IN_TRADE                 (stop / target / invalidation watch)
      | stop hit | target hit | LTF close back through OB
      v
    PENDING_EXIT             (one bar; fills next-bar-open)
      v
    IDLE (Trade emitted)

Execution model: signals fire on bar T's close; fills happen at bar T+1's
open. This is the "strict next-bar fill" rule the user picked for v1.

Concurrency: ONE active setup at a time. If a sweep arrives while we
already have a setup in flight, it's ignored. Future work: run two
parallel state machines (one for sweep-up->short, one for sweep-down->long)
so both directions can be live simultaneously.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd

from strategy.bars import Bar
from strategy.fvg import FVG, FVGDetector, FVGKind, FVGState, OrderBlock, find_order_block
from strategy.levels import Level, LevelKind
from strategy.params import NQ_TICK_SIZE, StrategyParams
from strategy.sweep import SweepDirection, SweepEvent
from strategy.swings import (
    BreakMode,
    SwingDetector,
    SwingKind,
    broke_swing_down,
    broke_swing_up,
)


# ---------------------------------------------------------------------------
# Output type: closed Trade record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Trade:
    setup_kind: str          # "sweep" or "continuation"
    direction: str           # "long" or "short"
    entry_bar_idx: int
    entry_price: float
    stop_price: float
    target_price: float
    exit_bar_idx: int
    exit_price: float
    exit_reason: str         # "stop", "target", "invalidated"
    pnl_points: float        # signed; +ve = winner
    rr_at_entry: float       # reward / risk computed at trade open
    # Sweep-specific provenance. None for continuation trades, which have no
    # swept level or CHoCH. Continuation provenance (the retest level) is left
    # implicit in v1 — these stay None.
    swept_level_kind: Optional[LevelKind] = None
    swept_level_price: Optional[float] = None
    sweep_bar_idx: Optional[int] = None  # rejection bar of the sweep
    choch_bar_idx: Optional[int] = None


# ---------------------------------------------------------------------------
# Setup target mapping
# ---------------------------------------------------------------------------


PAIRED_OPPOSITES: dict[LevelKind, LevelKind] = {
    LevelKind.ASIA_HIGH: LevelKind.ASIA_LOW,
    LevelKind.ASIA_LOW: LevelKind.ASIA_HIGH,
    LevelKind.LONDON_HIGH: LevelKind.LONDON_LOW,
    LevelKind.LONDON_LOW: LevelKind.LONDON_HIGH,
    LevelKind.PRIOR_SESS_HIGH: LevelKind.PRIOR_SESS_LOW,
    LevelKind.PRIOR_SESS_LOW: LevelKind.PRIOR_SESS_HIGH,
    LevelKind.PRIOR_WEEK_HIGH: LevelKind.PRIOR_WEEK_LOW,
    LevelKind.PRIOR_WEEK_LOW: LevelKind.PRIOR_WEEK_HIGH,
}


def _paired_target(swept_level: Level, active_levels: list[Level]) -> Optional[float]:
    """The opposite-pair level (same source_day), if one is active."""
    pair_kind = PAIRED_OPPOSITES.get(swept_level.kind)
    if pair_kind is None:
        return None
    for lvl in active_levels:
        if lvl.kind == pair_kind and lvl.source_day == swept_level.source_day:
            return lvl.price
    return None


def _has_opposite_side_level(
    swept_level: Level, active_levels: list[Level], direction: str
) -> bool:
    """Idle-time check: is there ANY level on the target side to aim at?

    Cheap gate so we don't start a reversal setup that can never find a target.
    Target side is below for a short (sweep up), above for a long (sweep down).
    The full R:R-filtered target is chosen later, at entry.
    """
    for lvl in active_levels:
        if lvl is swept_level:
            continue
        if direction == "short" and lvl.price < swept_level.price:
            return True
        if direction == "long" and lvl.price > swept_level.price:
            return True
    return False


def _find_target_price(
    swept_level: Level,
    active_levels: list[Level],
    direction: str,
    entry_price: float,
    stop_price: float,
    min_rr: float,
) -> Optional[float]:
    """Nearest opposite-direction level that clears the min-R:R filter.

    Priority chain — first rule that yields a qualifying level wins:
      1. paired session extreme (Asia<->Asia, London<->London, ...)
      2. nearest EQH/EQL on the target side
      3. nearest ROUND_MAJOR on the target side
      4. nearest level of any kind on the target side

    "Qualifying" means the level is far enough from entry that
    reward/risk >= min_rr. Among qualifying levels we take the *nearest* to
    entry — the most conservative target that still clears R:R. This is the
    expansion that lets EQH/EQL/round-number sweeps (the bulk of real
    signals) trade, instead of only paired session extremes.
    """
    risk = abs(entry_price - stop_price)
    if risk <= 0:
        return None
    min_dist = min_rr * risk

    def qualifies(price: float) -> bool:
        if direction == "short":  # target below entry
            return price <= entry_price - min_dist
        return price >= entry_price + min_dist  # long: target above entry

    def nearest(levels: list[Level]) -> Optional[float]:
        prices = [l.price for l in levels if l is not swept_level and qualifies(l.price)]
        if not prices:
            return None
        # Nearest to entry: highest qualifying price for a short (target is
        # below, so highest = closest), lowest for a long.
        return max(prices) if direction == "short" else min(prices)

    # 1. paired session extreme
    paired = _paired_target(swept_level, active_levels)
    if paired is not None and qualifies(paired):
        return paired
    # 2. EQH / EQL
    eq = nearest([l for l in active_levels if l.kind in (LevelKind.EQH, LevelKind.EQL)])
    if eq is not None:
        return eq
    # 3. ROUND_MAJOR
    rm = nearest([l for l in active_levels if l.kind == LevelKind.ROUND_MAJOR])
    if rm is not None:
        return rm
    # 4. any kind
    return nearest(active_levels)


# ---------------------------------------------------------------------------
# State enum + internal setup record
# ---------------------------------------------------------------------------


class SetupState(Enum):
    IDLE = "idle"
    WATCHING_FOR_CHOCH = "watching_for_choch"
    WATCHING_FOR_ENTRY = "watching_for_entry"
    PENDING_ENTRY = "pending_entry"
    IN_TRADE = "in_trade"
    PENDING_EXIT = "pending_exit"


@dataclass
class _Setup:
    sweep: SweepEvent
    direction: str             # "short" (sweep up) or "long" (sweep down)
    # Target is resolved at entry (needs entry+stop for the R:R filter), so it
    # starts unset.
    target_price: Optional[float] = None
    # After CHoCH:
    choch_bar_idx: Optional[int] = None
    ob: Optional[OrderBlock] = None
    displacement_kind: Optional[FVGKind] = None  # the direction of the impulse leg
    entry_watch_start_idx: Optional[int] = None
    # Displacement extreme: lowest low (short) or highest high (long)
    # observed since CHoCH. Used to require that price actually moved away
    # from the OB before counting a retest as an entry.
    displacement_extreme: Optional[float] = None
    # Pending entry:
    pending_entry_price: Optional[float] = None
    # Open trade:
    entry_bar_idx: Optional[int] = None
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    rr_at_entry: Optional[float] = None
    # Pending exit:
    pending_exit_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class SweepStateMachine:
    def __init__(self, params: StrategyParams) -> None:
        self.params = params
        self._state = SetupState.IDLE
        self._setup: Optional[_Setup] = None
        self._bar_count = 0
        self._bars: list[Bar] = []
        # Most recent active-level snapshot — set each bar so the entry step
        # (which only receives bar/bar_idx) can resolve a target.
        self._active_levels: list[Level] = []
        # Sub-detectors maintained at LTF resolution.
        self._swing_detector = SwingDetector(n=params.ltf_swing_lookback)
        self._fvg_detector = FVGDetector()
        # Completed trades (caller can read or via return value).
        self._closed_trades: list[Trade] = []

    @property
    def state(self) -> SetupState:
        return self._state

    @property
    def closed_trades(self) -> list[Trade]:
        return list(self._closed_trades)

    def on_bar(
        self,
        bar: Bar,
        active_levels: list[Level],
        sweep_events: list[SweepEvent],
    ) -> list[Trade]:
        """Advance one bar. Returns Trades that closed on this bar (usually 0 or 1)."""
        bar_idx = self._bar_count
        self._bar_count += 1
        self._bars.append(bar)
        self._active_levels = active_levels
        self._swing_detector.on_bar(bar)
        self._fvg_detector.on_bar(bar)

        newly_closed: list[Trade] = []

        # PENDING_* states consume a full bar (they represent the next-bar-open
        # fill for a signal that fired on the previous bar). Resolve and return.
        if self._state == SetupState.PENDING_ENTRY:
            self._pending_entry_step(bar, bar_idx)
            return newly_closed
        if self._state == SetupState.PENDING_EXIT:
            trade = self._pending_exit_step(bar, bar_idx)
            if trade is not None:
                newly_closed.append(trade)
                self._closed_trades.append(trade)
            return newly_closed

        # Non-pending states may chain within a single bar (e.g., a sweep that
        # arrives AND its CHoCH break occurs on the same bar). We stop chaining
        # if we would enter a PENDING_* state — that signal needs the next bar.
        for _ in range(len(SetupState)):
            prev_state = self._state
            if self._state == SetupState.IDLE:
                self._idle_step(bar, bar_idx, active_levels, sweep_events)
            elif self._state == SetupState.WATCHING_FOR_CHOCH:
                self._watching_for_choch_step(bar, bar_idx)
            elif self._state == SetupState.WATCHING_FOR_ENTRY:
                self._watching_for_entry_step(bar, bar_idx)
            elif self._state == SetupState.IN_TRADE:
                self._in_trade_step(bar, bar_idx)
            else:
                break
            if self._state == prev_state:
                break
            if self._state in (SetupState.PENDING_ENTRY, SetupState.PENDING_EXIT):
                break

        return newly_closed

    # ---- per-state handlers ----------------------------------------------

    def _idle_step(self, bar, bar_idx, active_levels, sweep_events) -> None:
        # Multiple sweeps can fire on the same bar (e.g. ASIA_HIGH and a
        # ROUND_MAJOR colocated at the same price). Take the first event that
        # has at least one level on its target side; the rest are dropped. The
        # actual target (R:R-filtered) is resolved later, at entry.
        for ev in sweep_events:
            direction = "short" if ev.direction == SweepDirection.UP else "long"
            if not _has_opposite_side_level(ev.level, active_levels, direction):
                continue
            self._setup = _Setup(sweep=ev, direction=direction)
            self._state = SetupState.WATCHING_FOR_CHOCH
            return

    def _watching_for_choch_step(self, bar, bar_idx) -> None:
        setup = self._setup
        # Deadline check
        bars_since_sweep = bar_idx - setup.sweep.rejection_bar_idx
        if bars_since_sweep > self.params.max_choch_bars:
            self._reset()
            return

        # Find the most recent opposing swing whose bar is at or before the
        # sweep rejection, and which is visible at this bar.
        if setup.direction == "short":
            swing = self._swing_detector.most_recent(
                SwingKind.LOW,
                before_bar_idx=setup.sweep.rejection_bar_idx + 1,
                before_confirmation_idx=bar_idx,
            )
            if swing is None:
                return
            if broke_swing_down(bar, swing, BreakMode.CLOSE):
                self._on_choch(bar, bar_idx)
        else:
            swing = self._swing_detector.most_recent(
                SwingKind.HIGH,
                before_bar_idx=setup.sweep.rejection_bar_idx + 1,
                before_confirmation_idx=bar_idx,
            )
            if swing is None:
                return
            if broke_swing_up(bar, swing, BreakMode.CLOSE):
                self._on_choch(bar, bar_idx)

    def _on_choch(self, bar, bar_idx) -> None:
        setup = self._setup
        # Displacement kind matches the sweep's reversion direction.
        setup.displacement_kind = (
            FVGKind.BEARISH if setup.direction == "short" else FVGKind.BULLISH
        )
        # OB sits immediately before the CHoCH bar.
        ob = find_order_block(
            self._bars,
            displacement_start_idx=bar_idx,
            displacement_direction=setup.displacement_kind,
            max_lookback=self.params.max_choch_bars,
        )
        # If we have neither an OB nor any FVG in the leg yet, the setup
        # may still complete as FVGs form. Don't abandon yet — only abandon
        # if entry window elapses with no zone touch.
        setup.choch_bar_idx = bar_idx
        setup.ob = ob
        setup.entry_watch_start_idx = bar_idx
        self._state = SetupState.WATCHING_FOR_ENTRY

    def _watching_for_entry_step(self, bar, bar_idx) -> None:
        setup = self._setup
        # Entry-window timeout
        if bar_idx - setup.entry_watch_start_idx > self.params.max_entry_bars:
            self._reset()
            return

        # Always track displacement extreme — including the CHoCH bar itself,
        # whose move IS the start of displacement.
        if setup.direction == "short":
            setup.displacement_extreme = (
                bar.low if setup.displacement_extreme is None
                else min(setup.displacement_extreme, bar.low)
            )
        else:
            setup.displacement_extreme = (
                bar.high if setup.displacement_extreme is None
                else max(setup.displacement_extreme, bar.high)
            )

        # Skip entry-trigger evaluation on the CHoCH bar itself. The CHoCH
        # bar's high (for a short) is artifactually inside the OB zone because
        # price just came from above it — that isn't a retest from below.
        if bar_idx <= setup.choch_bar_idx:
            return

        zones = self._current_entry_zones(setup, bar_idx)
        if not zones:
            return

        # Only consider zones the price has actually moved AWAY from since CHoCH.
        # For a short setup, the entry zone is above; price must have dipped
        # below the zone's lower edge before a retest counts as entry.
        if setup.direction == "short":
            eligible = [z for z in zones if setup.displacement_extreme < z[0]]
        else:
            eligible = [z for z in zones if setup.displacement_extreme > z[1]]
        if not eligible:
            return

        entry_price = self._first_touched_zone_edge(bar, setup.direction, eligible)
        if entry_price is None:
            return

        # Compute stop, then resolve the target (R:R-filtered priority chain).
        stop_price = self._stop_price(setup, entry_price)
        risk = abs(entry_price - stop_price)
        if risk <= 0:
            self._reset()
            return
        target = _find_target_price(
            setup.sweep.level, self._active_levels, setup.direction,
            entry_price, stop_price, self.params.min_rr_ratio,
        )
        if target is None:
            # No level on the target side clears R:R — abandon the setup.
            self._reset()
            return
        setup.target_price = target
        reward = abs(target - entry_price)
        rr = reward / risk  # >= min_rr by construction of _find_target_price
        if rr < self.params.min_rr_ratio:
            self._reset()
            return

        setup.pending_entry_price = entry_price
        setup.rr_at_entry = rr
        self._state = SetupState.PENDING_ENTRY

    def _pending_entry_step(self, bar, bar_idx) -> None:
        """On the bar AFTER an entry signal, open the trade at this bar's open."""
        setup = self._setup
        setup.entry_bar_idx = bar_idx
        setup.entry_price = bar.open  # next-bar-open fill
        setup.stop_price = self._stop_price(setup, bar.open)
        # Re-check R:R with realized entry; if it degraded below threshold, abort.
        risk = abs(setup.entry_price - setup.stop_price)
        reward = abs(setup.target_price - setup.entry_price)
        if risk <= 0 or reward / risk < self.params.min_rr_ratio:
            self._reset()
            return
        setup.rr_at_entry = reward / risk
        self._state = SetupState.IN_TRADE

    def _in_trade_step(self, bar, bar_idx) -> None:
        setup = self._setup
        reason: Optional[str] = None
        if setup.direction == "short":
            # Stop above, target below
            if bar.high >= setup.stop_price:
                reason = "stop"
            elif bar.low <= setup.target_price:
                reason = "target"
            elif setup.ob is not None and bar.close > setup.ob.upper:
                reason = "invalidated"
        else:
            if bar.low <= setup.stop_price:
                reason = "stop"
            elif bar.high >= setup.target_price:
                reason = "target"
            elif setup.ob is not None and bar.close < setup.ob.lower:
                reason = "invalidated"
        if reason is not None:
            setup.pending_exit_reason = reason
            self._state = SetupState.PENDING_EXIT

    def _pending_exit_step(self, bar, bar_idx) -> Optional[Trade]:
        setup = self._setup
        exit_price = bar.open  # next-bar-open fill
        if setup.direction == "short":
            pnl = setup.entry_price - exit_price
        else:
            pnl = exit_price - setup.entry_price
        trade = Trade(
            setup_kind="sweep",
            direction=setup.direction,
            swept_level_kind=setup.sweep.level.kind,
            swept_level_price=setup.sweep.level.price,
            sweep_bar_idx=setup.sweep.rejection_bar_idx,
            choch_bar_idx=setup.choch_bar_idx,
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
        self._state = SetupState.IDLE
        self._setup = None

    def _current_entry_zones(self, setup: _Setup, bar_idx: int) -> list[tuple[float, float]]:
        """Return (lower, upper) ranges for OB + ACTIVE FVGs in the displacement leg.

        We exclude FVGs created on the *current* bar: their wick by construction
        sits at the FVG's near edge, which would spuriously satisfy a "first
        touch" check. They become eligible on the bar after creation.
        """
        zones: list[tuple[float, float]] = []
        if setup.ob is not None:
            zones.append((setup.ob.lower, setup.ob.upper))
        fvgs = self._fvg_detector.find_in_range(
            setup.choch_bar_idx, bar_idx - 1, kind=setup.displacement_kind
        )
        for f in fvgs:
            if f.state == FVGState.ACTIVE:
                zones.append((f.lower, f.upper))
        return zones

    @staticmethod
    def _first_touched_zone_edge(
        bar: Bar, direction: str, zones: list[tuple[float, float]],
    ) -> Optional[float]:
        """For a short, the entry zones are above current price; entry fires
        when the bar's high reaches the LOWER edge of any zone (price coming
        back up from below). Return the lowest-edge zone touched.

        Mirror for long: bar.low reaches upper edge; return highest-edge.
        """
        if direction == "short":
            candidates = [z for z in zones if bar.high >= z[0]]
            if not candidates:
                return None
            return min(z[0] for z in candidates)
        else:
            candidates = [z for z in zones if bar.low <= z[1]]
            if not candidates:
                return None
            return max(z[1] for z in candidates)

    def _stop_price(self, setup: _Setup, entry_price: float) -> float:
        buf = self.params.stop_buffer_ticks * NQ_TICK_SIZE
        if setup.direction == "short":
            return setup.sweep.wick_extreme + buf
        return setup.sweep.wick_extreme - buf
