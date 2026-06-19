"""Continuation state machine tests (module 12).

The HTF (4h) trend needs thousands of 1m bars to form organically, so for
the state-logic tests we inject a fake HTF (fixed trend + target swing
levels) and a fixed VWAP. The real LTF swing detector still runs on the
crafted bars — it supplies the stop. Confluence gates are toggled per test.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

from strategy.bars import Bar
from strategy.continuation_state_machine import ContState, ContinuationStateMachine
from strategy.fvg import FVG, FVGKind, FVGState
from strategy.htf import TrendDirection
from strategy.params import DEFAULT_PARAMS
from strategy.pressure import AbsorptionReading, PressureReading

BASE = pd.Timestamp("2026-06-01 18:00").tz_localize("US/Eastern").tz_convert("UTC")

GATES_OFF = replace(
    DEFAULT_PARAMS,
    cont_require_pressure=False,
    cont_require_absorption=False,
    cont_require_fvg_at_vwap=False,
)


def bar(o, h, l, c, v=1.0, i=0) -> Bar:
    return Bar(ts=BASE + pd.Timedelta(minutes=i), open=o, high=h, low=l, close=c, volume=v)


class FakeVWAP:
    def __init__(self, value: float) -> None:
        self.value = value

    def on_bar(self, bar) -> None:
        pass

    @property
    def current(self):
        return self.value


class FakeHTF:
    def __init__(self, trend, highs=(), lows=()) -> None:
        self.trend = trend
        self._highs = [SimpleNamespace(price=p) for p in highs]
        self._lows = [SimpleNamespace(price=p) for p in lows]

    def on_bar(self, bar) -> None:
        pass

    def current_trend(self):
        return self.trend

    @property
    def swing_highs(self):
        return self._highs

    @property
    def swing_lows(self):
        return self._lows


def make_cm(params=GATES_OFF, *, vwap=100.0, trend=TrendDirection.BULLISH,
            highs=(), lows=()) -> ContinuationStateMachine:
    cm = ContinuationStateMachine(params)
    cm._vwap = FakeVWAP(vwap)
    cm._htf = FakeHTF(trend, highs=highs, lows=lows)
    return cm


def feed(cm, bars) -> list:
    trades = []
    for i, b in enumerate(bars):
        trades.extend(cm.on_bar(bar(*b, i=i)))
    return trades


# Phase A: arm long at idx0, form a confirmed LTF swing low (n=3) at idx3
# (low 100.5), no retest yet (all lows stay above VWAP=100).
LONG_PREFIX = [
    (102, 103, 101, 102),     # 0 arm long (close > vwap)
    (102, 103, 101, 102),     # 1
    (102, 103, 101, 102),     # 2
    (101.5, 102, 100.5, 101.5),  # 3 swing-low candidate (low 100.5)
    (102, 103, 101, 102),     # 4
    (102, 103, 101, 102),     # 5
    (102, 103, 101, 102),     # 6 confirms swing low @3
    (101, 101, 99.5, 100.5),  # 7 retest: low <= vwap -> PENDING_ENTRY
    (100.5, 101, 100.3, 100.8),  # 8 fill -> IN_TRADE (entry = open = 100.5)
]


# --- arming ----------------------------------------------------------------

def test_arms_long_in_uptrend_above_vwap():
    cm = make_cm(trend=TrendDirection.BULLISH)
    feed(cm, [(102, 103, 101, 102)])
    assert cm.state == ContState.WATCHING_FOR_RETEST
    assert cm._setup.bias == "long"


def test_does_not_arm_when_trend_undefined():
    cm = make_cm(trend=TrendDirection.UNDEFINED)
    feed(cm, [(102, 103, 101, 102)])
    assert cm.state == ContState.IDLE


def test_does_not_arm_long_when_close_below_vwap():
    cm = make_cm(trend=TrendDirection.BULLISH)
    feed(cm, [(99, 100, 98, 99)])  # close below vwap
    assert cm.state == ContState.IDLE


# --- long lifecycle --------------------------------------------------------

def test_long_lifecycle_to_target():
    cm = make_cm(highs=[112.0])
    bars = LONG_PREFIX + [
        (105, 112, 104, 111),       # 9 target hit (high >= 112)
        (110, 110.5, 109.5, 110),   # 10 exit fill (open = 110)
    ]
    trades = feed(cm, bars)
    assert len(trades) == 1
    t = trades[0]
    assert t.setup_kind == "continuation"
    assert t.direction == "long"
    assert t.exit_reason == "target"
    assert t.entry_price == pytest.approx(100.5)
    assert t.stop_price == pytest.approx(100.0)   # swing low 100.5 - 2 ticks
    assert t.target_price == pytest.approx(112.0)
    assert t.exit_price == pytest.approx(110.0)
    assert t.pnl_points == pytest.approx(9.5)
    assert t.swept_level_kind is None             # continuation carries no level


def test_long_lifecycle_to_stop():
    cm = make_cm(highs=[112.0])
    bars = LONG_PREFIX + [
        (100.4, 100.6, 99.0, 99.5),   # 9 stop hit (low <= 100.0)
        (99.5, 99.8, 99.0, 99.3),     # 10 exit fill (open = 99.5)
    ]
    trades = feed(cm, bars)
    assert len(trades) == 1
    t = trades[0]
    assert t.exit_reason == "stop"
    assert t.exit_price == pytest.approx(99.5)
    assert t.pnl_points == pytest.approx(-1.0)


def test_htf_flip_invalidates_open_long():
    cm = make_cm(highs=[112.0])
    feed(cm, LONG_PREFIX)  # now IN_TRADE
    assert cm.state == ContState.IN_TRADE
    # Trend flips bearish while we're long -> invalidation exit.
    cm._htf.trend = TrendDirection.BEARISH
    trades = feed_continue(cm, [
        (100.5, 101, 100.2, 100.6),   # neither stop nor target -> invalidated
        (100.6, 100.8, 100.4, 100.7), # exit fill (open = 100.6)
    ], start=len(LONG_PREFIX))
    assert len(trades) == 1
    assert trades[0].exit_reason == "invalidated"


def test_no_trade_when_no_htf_target_clears_rr():
    cm = make_cm(highs=[])  # no HTF level to target
    trades = feed(cm, LONG_PREFIX)
    assert trades == []
    # Retest happened but produced no target -> still watching.
    assert cm.state == ContState.WATCHING_FOR_RETEST


# --- short lifecycle (mirror) ---------------------------------------------

def test_short_lifecycle_to_target():
    cm = make_cm(trend=TrendDirection.BEARISH, vwap=100.0, lows=[88.0])
    bars = [
        (98, 99, 97, 98),         # 0 arm short (close < vwap)
        (98, 99, 97, 98),         # 1
        (98, 99, 97, 98),         # 2
        (98.5, 99.5, 98, 98.5),   # 3 swing-high candidate (high 99.5)
        (98, 99, 97, 98),         # 4
        (98, 99, 97, 98),         # 5
        (98, 99, 97, 98),         # 6 confirms swing high @3
        (99, 100.5, 99, 99.5),    # 7 retest: high >= vwap -> PENDING_ENTRY
        (99.5, 99.8, 99.2, 99.4), # 8 fill -> IN_TRADE (entry = open = 99.5)
        (95, 96, 88, 89),         # 9 target hit (low <= 88)
        (90, 90.5, 89.5, 90),     # 10 exit fill (open = 90)
    ]
    trades = feed(cm, bars)
    assert len(trades) == 1
    t = trades[0]
    assert t.direction == "short"
    assert t.exit_reason == "target"
    assert t.entry_price == pytest.approx(99.5)
    assert t.stop_price == pytest.approx(100.0)   # swing high 99.5 + 2 ticks
    assert t.target_price == pytest.approx(88.0)
    assert t.pnl_points == pytest.approx(9.5)      # 99.5 - 90


# --- confluence gating -----------------------------------------------------

def test_pressure_gate_blocks_entry():
    # Require pressure; the retest bar (idx7) has no buying pressure -> blocked.
    params = replace(GATES_OFF, cont_require_pressure=True)
    cm = make_cm(params=params, highs=[112.0])
    trades = feed(cm, LONG_PREFIX)
    assert trades == []
    assert cm.state == ContState.WATCHING_FOR_RETEST


def _buy_reading(buying=True):
    return PressureReading(
        close_position_in_range=0.9, body_fraction=0.8,
        body_vs_trailing_avg=1.5, buying_pressure=buying, selling_pressure=False,
    )


def _absorb_reading(absorption=True):
    return AbsorptionReading(
        volume_ratio=2.0, range_ratio=0.5, overlaps_level=True, absorption=absorption,
    )


def test_confluence_all_pass():
    cm = ContinuationStateMachine(DEFAULT_PARAMS)  # all gates required
    # Active bullish FVG straddling VWAP=100.
    cm._fvg._fvgs.append(FVG(kind=FVGKind.BULLISH, upper=101, lower=99,
                             created_at_idx=0, ts=BASE, state=FVGState.ACTIVE))
    assert cm._confluence_ok("long", 100.0, _buy_reading(), _absorb_reading()) is True


def test_confluence_blocked_by_each_gate():
    cm = ContinuationStateMachine(DEFAULT_PARAMS)
    cm._fvg._fvgs.append(FVG(kind=FVGKind.BULLISH, upper=101, lower=99,
                             created_at_idx=0, ts=BASE, state=FVGState.ACTIVE))
    # No buying pressure
    assert cm._confluence_ok("long", 100.0, _buy_reading(False), _absorb_reading()) is False
    # No absorption
    assert cm._confluence_ok("long", 100.0, _buy_reading(), _absorb_reading(False)) is False
    # No FVG at VWAP (vwap outside the gap range)
    assert cm._confluence_ok("long", 200.0, _buy_reading(), _absorb_reading()) is False


def test_reclaim_mode_blocks_touch_without_close_back():
    # In reclaim mode a bar that dips to VWAP but closes below it (long) is
    # NOT an entry — it must close back on the trend side.
    params = replace(GATES_OFF, cont_entry_mode="reclaim")
    cm = make_cm(params=params, highs=[112.0])
    bars = LONG_PREFIX[:7] + [
        (101, 101, 99.5, 99.8),   # 7 dips to VWAP but closes below -> no reclaim
        (99.8, 100, 99.5, 99.9),  # 8
    ]
    trades = feed(cm, bars)
    assert trades == []
    assert cm.state == ContState.WATCHING_FOR_RETEST


def test_reclaim_mode_enters_on_close_back():
    # Same retest that DOES close back above VWAP fires in reclaim mode.
    params = replace(GATES_OFF, cont_entry_mode="reclaim")
    cm = make_cm(params=params, highs=[112.0])
    bars = LONG_PREFIX + [
        (105, 112, 104, 111),       # target hit
        (110, 110.5, 109.5, 110),   # exit fill
    ]
    trades = feed(cm, bars)
    assert len(trades) == 1
    assert trades[0].exit_reason == "target"


def feed_continue(cm, bars, start):
    """Feed more bars continuing the index counter (for mid-stream changes)."""
    trades = []
    for k, b in enumerate(bars):
        trades.extend(cm.on_bar(bar(*b, i=start + k)))
    return trades
