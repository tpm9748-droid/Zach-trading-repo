"""End-to-end smoke test on synthetic bars.

This is the first test that drives the entire pipeline together — bars
through levels, sweep detector, state machine, engine, metrics — with no
mocking of internal components. It builds a hand-crafted bar sequence
designed to produce a single, known sweep trade and verifies the engine
reports it correctly.

The sequence:

  bars 0..19   Asia session (19:00-19:19 ET on 2026-06-01).
               Establishes Asia HIGH = 100 (bar 0) and Asia LOW = 80 (bar 1).
               Twenty bars also satisfies the sweep detector's 20-bar
               volume bootstrap.

  bars 20..26  London (03:00-03:06 ET on 2026-06-02). Asia is now locked.
               These bars establish a confirmed N=3 LTF swing low at bar 23
               with price 95 — the level the CHoCH will break.

  bar 27       Sweep bar (03:07 ET). Wick to 100.75 (3 ticks above Asia
               HIGH), close back to 99, volume 200 (2x SMA). Detected
               same-bar by the sweep detector.

  bar 28       CHoCH-down: close 92.5 < swing low 95. The state machine's
               internal swing detector knows about the bar-23 swing by now
               (confirmed at bar 26).

  bars 29..32  Continued displacement downward.

  bar 33       Big retrace UP to OB lower edge (96.5). State machine signals
               entry.

  bar 34       Entry fills at bar 34 open (96).

  bar 35       Drops to Asia LOW = 80. Target hit; state machine signals exit.

  bar 36       Exit fills at bar 36 open (80.5).
"""
from __future__ import annotations

import pandas as pd
import pytest

from backtest.engine import run_backtest
from strategy.bars import Bar
from strategy.levels import LevelKind
from strategy.params import DEFAULT_PARAMS


def _et_bar(ts_str: str, o: float, h: float, l: float, c: float, v: float = 100.0) -> Bar:
    ts = pd.Timestamp(ts_str).tz_localize("US/Eastern").tz_convert("UTC")
    return Bar(ts=ts, open=o, high=h, low=l, close=c, volume=v)


def _build_synthetic_bars() -> list[Bar]:
    bars: list[Bar] = []

    # --- Asia session: 19:00-19:19 ET on 2026-06-01 ---
    # bar 0: establishes Asia HIGH = 100
    bars.append(_et_bar("2026-06-01 19:00", 95, 100, 94, 98))
    # bar 1: establishes Asia LOW = 80
    bars.append(_et_bar("2026-06-01 19:01", 98, 99, 80, 85))
    # bars 2-19: filler, lows > 80 and highs < 100
    for i in range(2, 20):
        ts = f"2026-06-01 19:{i:02d}"
        bars.append(_et_bar(ts, 90, 91, 89, 90))

    # --- London (03:00 ET on 2026-06-02): Asia just locked ---
    # bars 20-26: pre-sweep, with a confirmed swing low at bar 23 (price 95)
    # Lows: 96, 95.5, 95.2, 95, 95.5, 96, 96.5 -> swing low at bar 23 confirmed
    # at bar 26 with N=3 LTF.
    london_seq = [
        ("03:00", 96.5, 97,    96,    96.5),  # bar 20
        ("03:01", 96.5, 96.7,  95.5,  96),    # bar 21
        ("03:02", 96,   96.5,  95.2,  95.5),  # bar 22
        ("03:03", 95.5, 95.7,  95,    95.2),  # bar 23  <- swing low at 95
        ("03:04", 95.2, 95.7,  95.5,  95.7),  # bar 24  Wait: low=95.5 but I wrote 95.5
        ("03:05", 95.7, 96.2,  96,    96),    # bar 25
        ("03:06", 96,   96.7,  96.5,  96.5),  # bar 26  (must be valid OHLC)
    ]
    # Fix OHLC validity: in bar 24 above, low=95.5 must equal min, but I had 95.7 wick.
    # Replace with valid values:
    london_seq = [
        ("03:00", 96.5, 97,    96,    96.5),  # bar 20: low=96
        ("03:01", 96.5, 96.7,  95.5,  96),    # bar 21: low=95.5
        ("03:02", 96,   96.2,  95.2,  95.5),  # bar 22: low=95.2
        ("03:03", 95.5, 95.6,  95,    95.2),  # bar 23: low=95  <-- swing
        ("03:04", 95.2, 95.8,  95.2,  95.7),  # bar 24: low=95.2 (>95)
        ("03:05", 95.7, 96.2,  95.7,  96),    # bar 25: low=95.7 (>95)
        ("03:06", 96,   96.7,  96,    96.5),  # bar 26: low=96 (>95)
    ]
    for hm, o, h, l, c in london_seq:
        bars.append(_et_bar(f"2026-06-02 {hm}", o, h, l, c))

    # bar 27: sweep up of Asia HIGH (100). 3-tick penetration, close back inside,
    # volume 200 = 2x SMA. Bullish (open<close) so the OB walker picks it up.
    # OB range = [bar.low, bar.high] = [98.5, 100.75].
    bars.append(_et_bar("2026-06-02 03:07", 99, 100.75, 98.5, 99.5, v=200))

    # bar 28: CHoCH-down. close 92.5 < swing low 95. (Entry trigger on this bar
    # is skipped by the state machine's CHoCH-bar guard.)
    bars.append(_et_bar("2026-06-02 03:08", 99, 99, 92, 92.5))

    # bars 29-32: displacement continues down. Each bar creates a bearish FVG
    # (kept tightly below the OB so no premature entry triggers via FVG.lower).
    bars.append(_et_bar("2026-06-02 03:09", 92.5, 96,   89, 89.5))  # FVG with bar 27: [96, 98.5]
    bars.append(_et_bar("2026-06-02 03:10", 89.5, 91,   87, 87.5))  # FVG with bar 28: [91, 92]
    bars.append(_et_bar("2026-06-02 03:11", 87.5, 88,   85, 85.5))  # FVG with bar 29: [88, 89]
    bars.append(_et_bar("2026-06-02 03:12", 85.5, 86,   83, 83.5))  # FVG with bar 30: [86, 87]

    # bar 33: big retrace UP. Close = 99 fills all four bearish FVGs at once
    # (each upper edge <= 98.5 <= 99). After state updates, only the OB remains
    # in the entry-zone set -> entry triggers at OB.lower = 98.5.
    bars.append(_et_bar("2026-06-02 03:13", 83.5, 100, 83, 99))

    # bar 34: PENDING_ENTRY fills at this bar's open (98).
    bars.append(_et_bar("2026-06-02 03:14", 98, 98.5, 92, 93))

    # bar 35: drops to target = Asia LOW (80). low=80 -> target hit.
    bars.append(_et_bar("2026-06-02 03:15", 93, 93.5, 80, 80.5))

    # bar 36: PENDING_EXIT fills at bar open (80.5).
    bars.append(_et_bar("2026-06-02 03:16", 80.5, 80.5, 79, 80))

    # A few trailing bars (state machine should be IDLE).
    for i in range(17, 25):
        bars.append(_et_bar(f"2026-06-02 03:{i:02d}", 80, 81, 79, 80))

    return bars


def test_smoke_short_sweep_target_hit():
    bars = _build_synthetic_bars()
    result = run_backtest(bars, DEFAULT_PARAMS, collect_sweep_events=True)

    # The engine should have seen the sweep AND closed exactly one trade.
    # (Multiple sweep events may fire on bar 27 since the price 100 is also
    # a ROUND_MAJOR; the state machine takes the ASIA_HIGH one.)
    asia_high_sweeps = [e for e in result.sweep_events if e.level.kind == LevelKind.ASIA_HIGH]
    assert len(asia_high_sweeps) == 1, f"expected 1 Asia-high sweep, got {len(asia_high_sweeps)}"

    assert len(result.trades) == 1, (
        f"expected 1 trade, got {len(result.trades)}; "
        f"sweep events: {[(e.level.kind, e.level.price) for e in result.sweep_events]}"
    )
    trade = result.trades[0]
    assert trade.direction == "short"
    assert trade.swept_level_kind == LevelKind.ASIA_HIGH
    assert trade.swept_level_price == 100.0
    assert trade.target_price == 80.0
    assert trade.exit_reason == "target"
    assert trade.entry_price == 98.0
    assert trade.exit_price == 80.5
    assert trade.pnl_points == pytest.approx(17.5)
    # Stop = 100.75 + 2 ticks (0.5) = 101.25
    assert trade.stop_price == 101.25

    # Metrics sanity
    assert result.metrics.overall.n_trades == 1
    assert result.metrics.overall.n_wins == 1
    assert result.metrics.overall.win_rate == 1.0
    assert result.metrics.overall.total_pnl_points == pytest.approx(17.5)
    assert result.metrics.overall.max_drawdown_points == 0.0
    # By-kind breakdown should show one ASIA_HIGH trade
    assert LevelKind.ASIA_HIGH in result.metrics.by_level_kind
    assert result.metrics.by_level_kind[LevelKind.ASIA_HIGH].n_trades == 1


def test_smoke_no_trade_when_no_sweep():
    """Replace bar 27 (sweep) with a flat bar -> no sweep, no trade."""
    bars = _build_synthetic_bars()
    # Make bar 27 not penetrate Asia HIGH at all
    bars[27] = _et_bar("2026-06-02 03:07", 96.5, 97, 96.5, 96.5, v=200)
    result = run_backtest(bars, DEFAULT_PARAMS)
    assert result.trades == []
    assert result.metrics.overall.n_trades == 0
