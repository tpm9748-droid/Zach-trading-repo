"""Sweep state machine tests.

Most tests synthesize SweepEvent objects directly instead of running the
full sweep detector. That lets us verify the state machine's logic in
isolation — bar construction is already tedious without also having to
satisfy the sweep detector's volume/penetration rules.

Full end-to-end (sweep detector + state machine + engine) is covered by
the smoke test in module 8.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import date
from typing import Optional

import pandas as pd
import pytest

from strategy.bars import Bar
from strategy.levels import Level, LevelKind
from strategy.params import DEFAULT_PARAMS, NQ_TICK_SIZE
from strategy.sweep import SweepDirection, SweepEvent
from strategy.sweep_state_machine import (
    SetupState,
    SweepStateMachine,
    Trade,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def bar(minute: int, o: float, h: float, l: float, c: float, v: float = 100.0) -> Bar:
    base = pd.Timestamp("2026-06-01 09:30").tz_localize("US/Eastern").tz_convert("UTC")
    return Bar(ts=base + pd.Timedelta(minutes=minute), open=o, high=h, low=l, close=c, volume=v)


def asia_high(price: float, sd=date(2026, 6, 1)) -> Level:
    return Level(price=price, kind=LevelKind.ASIA_HIGH, source_day=sd)


def asia_low(price: float, sd=date(2026, 6, 1)) -> Level:
    return Level(price=price, kind=LevelKind.ASIA_LOW, source_day=sd)


def round_major(price: float) -> Level:
    return Level(price=price, kind=LevelKind.ROUND_MAJOR, source_day=None)


def sweep_up_event(
    swept_level: Level, *, penetration_bar_idx: int, wick_extreme: float,
    rejection_bar_idx: Optional[int] = None,
    volume_ratio: float = 2.0,
    bars: Optional[list[Bar]] = None,
) -> SweepEvent:
    return SweepEvent(
        level=swept_level,
        direction=SweepDirection.UP,
        penetration_bar_idx=penetration_bar_idx,
        rejection_bar_idx=rejection_bar_idx if rejection_bar_idx is not None else penetration_bar_idx,
        wick_extreme=wick_extreme,
        penetration_ticks=int(round((wick_extreme - swept_level.price) / NQ_TICK_SIZE)),
        volume_ratio=volume_ratio,
        ts=bars[penetration_bar_idx].ts if bars else pd.Timestamp("2026-06-01 09:30", tz="US/Eastern").tz_convert("UTC"),
    )


def sweep_down_event(
    swept_level: Level, *, penetration_bar_idx: int, wick_extreme: float,
    rejection_bar_idx: Optional[int] = None,
    bars: Optional[list[Bar]] = None,
) -> SweepEvent:
    return SweepEvent(
        level=swept_level,
        direction=SweepDirection.DOWN,
        penetration_bar_idx=penetration_bar_idx,
        rejection_bar_idx=rejection_bar_idx if rejection_bar_idx is not None else penetration_bar_idx,
        wick_extreme=wick_extreme,
        penetration_ticks=int(round((swept_level.price - wick_extreme) / NQ_TICK_SIZE)),
        volume_ratio=2.0,
        ts=bars[penetration_bar_idx].ts if bars else pd.Timestamp("2026-06-01 09:30", tz="US/Eastern").tz_convert("UTC"),
    )


def feed(
    sm: SweepStateMachine,
    bars: list[Bar],
    sweep_event_by_idx: dict[int, SweepEvent],
    active_levels_by_idx: Optional[dict[int, list[Level]]] = None,
    default_levels: Optional[list[Level]] = None,
) -> list[Trade]:
    """Run all bars through the state machine. Optionally inject a sweep event
    on specific bar indices. Returns the list of closed Trades."""
    closed: list[Trade] = []
    for i, b in enumerate(bars):
        events = []
        if i in sweep_event_by_idx:
            events.append(sweep_event_by_idx[i])
        levels = (
            active_levels_by_idx.get(i, default_levels or [])
            if active_levels_by_idx else (default_levels or [])
        )
        closed.extend(sm.on_bar(b, levels, events))
    return closed


# ---------------------------------------------------------------------------
# Basic state transitions
# ---------------------------------------------------------------------------


def test_idle_when_no_sweep_events():
    sm = SweepStateMachine(DEFAULT_PARAMS)
    closed = sm.on_bar(bar(0, 100, 101, 99, 100), [], [])
    assert closed == []
    assert sm.state == SetupState.IDLE


def test_sweep_with_no_paired_target_is_skipped():
    """Sweep of a round number with no other level on the target side ->
    nothing to aim at -> setup not started. (With other opposite-side levels
    present, the expanded target rule would let it trade.)"""
    sm = SweepStateMachine(DEFAULT_PARAMS)
    swept = round_major(20000.0)
    ev = sweep_up_event(swept, penetration_bar_idx=0, wick_extreme=20000.75)
    closed = sm.on_bar(bar(0, 19999, 20000.75, 19998, 19999.5), [swept], [ev])
    assert closed == []
    assert sm.state == SetupState.IDLE


def test_sweep_with_paired_target_enters_choch_watch():
    sm = SweepStateMachine(DEFAULT_PARAMS)
    swept = asia_high(100.0)
    pair = asia_low(80.0)
    ev = sweep_up_event(swept, penetration_bar_idx=0, wick_extreme=100.75)
    sm.on_bar(bar(0, 99, 100.75, 98.5, 99.5), [swept, pair], [ev])
    assert sm.state == SetupState.WATCHING_FOR_CHOCH


# ---------------------------------------------------------------------------
# CHoCH timeout
# ---------------------------------------------------------------------------


def test_choch_timeout_resets_setup():
    """If no CHoCH happens within max_choch_bars, setup is abandoned."""
    sm = SweepStateMachine(DEFAULT_PARAMS)
    swept = asia_high(100.0)
    pair = asia_low(80.0)
    bars_seq = [
        bar(0, 99, 100.75, 98.5, 99.5),  # sweep bar
    ]
    # Add max_choch_bars+2 filler bars where price stays above the prior swing low (none anyway)
    for i in range(1, DEFAULT_PARAMS.max_choch_bars + 3):
        bars_seq.append(bar(i, 99, 100, 98.5, 99.5))
    ev = sweep_up_event(swept, penetration_bar_idx=0, wick_extreme=100.75, bars=bars_seq)
    feed(sm, bars_seq, {0: ev}, default_levels=[swept, pair])
    assert sm.state == SetupState.IDLE


# ---------------------------------------------------------------------------
# Full short happy path
# ---------------------------------------------------------------------------


def _build_short_happy_bars() -> list[Bar]:
    """A bar sequence designed for a clean short setup:

      - bars 0..6: warmup containing a swing low at idx 3 at price 95
      - bar 7:     sweep bar (bullish, wick to 100.75) — sweep event injected
      - bar 8:     CHoCH-down (close 93.5 < swing low 95). High intentionally
                   below OB.lower so entry doesn't fire on CHoCH bar.
      - bars 9-10: continue displacement down
      - bar 11:    retraces UP to OB lower edge (entry trigger fires)
      - bar 12:    next-bar-open fill of the entry
      - bar 13:    drops to target=80 -> stop/target check signals exit
      - bar 14:    next-bar-open fill of exit
    """
    return [
        bar(0,  100, 101,    99,   100),
        bar(1,  100, 100.5,  98,   99),
        bar(2,   99, 99.5,   97,   98),
        bar(3,   98, 98.5,   95,   96),    # swing low at 95
        bar(4,   96, 97.5,   96,   97),
        bar(5,   97, 98.5,   97,   98),
        bar(6,   98, 99,     97.5, 98.5),  # bullish
        bar(7,   98.5, 100.75, 98.5, 99.5),  # sweep bar (bullish OB candidate, range [98.5, 100.75])
        bar(8,   98, 98,     92,   93.5),  # CHoCH-down: close < 95. High 98 < OB.lower (98.5)
        bar(9,   93.5, 93.5, 89,   89.5),
        bar(10,  89.5, 90,   87,   87.5),
        bar(11,  87.5, 99.5, 87,   99),    # retrace UP to OB.lower (98.5) -> entry trigger
        bar(12,  99,   99.5, 98,   98.5),  # entry fills at bar 12 open (99). Stays away from target/stop
        bar(13,  98.5, 98.5, 80,   80.5),  # drops to target = 80 -> exit signal
        bar(14,  80.5, 81,   79,   80),    # exit fills at bar 14 open (80.5)
    ]


def test_short_happy_path_target_hit():
    sm = SweepStateMachine(DEFAULT_PARAMS)
    swept = asia_high(100.0)
    pair = asia_low(80.0)
    bars_seq = _build_short_happy_bars()
    ev = sweep_up_event(swept, penetration_bar_idx=7, wick_extreme=100.75, bars=bars_seq)
    closed = feed(sm, bars_seq, {7: ev}, default_levels=[swept, pair])
    assert sm.state == SetupState.IDLE
    assert len(closed) == 1
    t = closed[0]
    assert t.direction == "short"
    assert t.exit_reason == "target"
    assert t.entry_bar_idx == 12
    assert t.entry_price == 99.0
    assert t.exit_bar_idx == 14
    assert t.exit_price == 80.5
    # Stop = sweep wick (100.75) + 2 ticks (0.5) = 101.25
    assert t.stop_price == 101.25
    assert t.target_price == 80.0
    # PnL = 99 - 80.5 = 18.5 points
    assert t.pnl_points == pytest.approx(18.5)
    # R:R at entry = (99 - 80) / (101.25 - 99) = 19 / 2.25 ≈ 8.44
    assert t.rr_at_entry == pytest.approx(19 / 2.25)


# ---------------------------------------------------------------------------
# Stop hit
# ---------------------------------------------------------------------------


def test_short_stops_out():
    """Same setup; bar 13 spikes back UP past the stop instead of dropping."""
    sm = SweepStateMachine(DEFAULT_PARAMS)
    swept = asia_high(100.0)
    pair = asia_low(80.0)
    bars_seq = _build_short_happy_bars()
    # Replace bar 13 with one that hits the stop (>= 101.25 high)
    bars_seq[13] = bar(13, 98.5, 102, 98, 101.5)
    # Bar 14 fills the stop at next-bar open
    bars_seq[14] = bar(14, 101.5, 102, 101, 101.5)
    ev = sweep_up_event(swept, penetration_bar_idx=7, wick_extreme=100.75, bars=bars_seq)
    closed = feed(sm, bars_seq, {7: ev}, default_levels=[swept, pair])
    assert len(closed) == 1
    t = closed[0]
    assert t.exit_reason == "stop"
    # PnL = entry (99) - exit (101.5) = -2.5
    assert t.pnl_points == pytest.approx(-2.5)


# ---------------------------------------------------------------------------
# R:R filter
# ---------------------------------------------------------------------------


def test_ob_invalidation_exits_when_enabled():
    """A bar closing back above the OB upper edge (without hitting the stop)
    invalidates the trade when the rule is enabled (default)."""
    sm = SweepStateMachine(DEFAULT_PARAMS)
    swept = asia_high(100.0)
    pair = asia_low(80.0)
    bars_seq = _build_short_happy_bars()
    # OB.upper = 100.75 (bar 7). Close 100.9 > 100.75, high 101.0 < stop 101.25.
    bars_seq[13] = bar(13, 99, 101.0, 98.5, 100.9)
    bars_seq[14] = bar(14, 100.9, 101.0, 100.5, 100.8)  # exit fill
    ev = sweep_up_event(swept, penetration_bar_idx=7, wick_extreme=100.75, bars=bars_seq)
    closed = feed(sm, bars_seq, {7: ev}, default_levels=[swept, pair])
    assert len(closed) == 1
    assert closed[0].exit_reason == "invalidated"


def test_ob_invalidation_skipped_when_disabled():
    """With the rule off, the same bar does NOT exit the trade early."""
    params = replace(DEFAULT_PARAMS, sweep_use_ob_invalidation=False)
    sm = SweepStateMachine(params)
    swept = asia_high(100.0)
    pair = asia_low(80.0)
    bars_seq = _build_short_happy_bars()
    bars_seq[13] = bar(13, 99, 101.0, 98.5, 100.9)
    bars_seq[14] = bar(14, 100.9, 101.0, 100.5, 100.8)
    ev = sweep_up_event(swept, penetration_bar_idx=7, wick_extreme=100.75, bars=bars_seq)
    closed = feed(sm, bars_seq, {7: ev}, default_levels=[swept, pair])
    assert closed == []  # no early invalidation -> trade still open
    assert sm.state == SetupState.IN_TRADE


def test_low_rr_rejects_trade():
    """Same setup but target too close -> R:R below min_rr_ratio -> skipped."""
    sm = SweepStateMachine(DEFAULT_PARAMS)
    swept = asia_high(100.0)
    pair = asia_low(95.0)  # only 4-pt reward vs 2.25-pt risk -> ~1.78 R:R, below 3
    bars_seq = _build_short_happy_bars()
    ev = sweep_up_event(swept, penetration_bar_idx=7, wick_extreme=100.75, bars=bars_seq)
    closed = feed(sm, bars_seq, {7: ev}, default_levels=[swept, pair])
    assert closed == []
    assert sm.state == SetupState.IDLE


# ---------------------------------------------------------------------------
# Long mirror (sweep down)
# ---------------------------------------------------------------------------


def _build_long_happy_bars() -> list[Bar]:
    """Mirror of the short setup: sweep down of Asia low at 100 -> long."""
    return [
        bar(0,  100,  101,   99,    100),
        bar(1,  100,  102,   99.5,  101),
        bar(2,  101,  103,   100.5, 102),
        bar(3,  102,  105,   101.5, 104),  # swing high at 105
        bar(4,  104,  104.5, 102.5, 103),
        bar(5,  103,  103.5, 101.5, 102),
        bar(6,  102,  102.5, 101,   101.5),  # bearish
        bar(7,  101.5, 101.5, 99.25, 100.5),  # sweep down (synth event); range [99.25, 101.5]
        bar(8,  102,  108,   102,   106.5),  # CHoCH-up: close (106.5) > swing high (105). Low 102 > OB.upper (101.5)
        bar(9,  106.5, 110.5, 106,  110),
        bar(10, 110,  113,   110,   112.5),
        bar(11, 112.5, 113,  101.5, 102),  # retraces DOWN to OB.upper (101.5) -> long entry trigger
        bar(12, 102,  102.5, 101.5, 102),  # entry fills at bar 12 open (102)
        bar(13, 102,  120,   101.5, 119.5),  # reaches target (Asia high = 120)
        bar(14, 119.5, 121, 119,   120),  # exit fills at bar 14 open (119.5)
    ]


def test_long_happy_path_target_hit():
    sm = SweepStateMachine(DEFAULT_PARAMS)
    swept = asia_low(100.0)
    pair = asia_high(120.0)
    bars_seq = _build_long_happy_bars()
    ev = sweep_down_event(swept, penetration_bar_idx=7, wick_extreme=99.25, bars=bars_seq)
    closed = feed(sm, bars_seq, {7: ev}, default_levels=[swept, pair])
    assert len(closed) == 1
    t = closed[0]
    assert t.direction == "long"
    assert t.exit_reason == "target"
    assert t.entry_price == 102.0
    assert t.exit_price == 119.5
    # PnL = 119.5 - 102 = 17.5 points
    assert t.pnl_points == pytest.approx(17.5)
    # Stop = 99.25 - 0.5 = 98.75. Risk = 102 - 98.75 = 3.25. Reward = 120 - 102 = 18. R:R ~ 5.54
    assert t.stop_price == pytest.approx(98.75)
    assert t.target_price == 120.0
