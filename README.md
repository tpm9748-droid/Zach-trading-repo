# NQ Sweep + Continuation Backtester

A bar-by-bar backtester for NASDAQ futures (NQ) implementing two ICT-style strategies:

1. **Sweep strategy** — reversal play. Detects liquidity sweeps of session-scale reference levels (Asia/London/prior-session/prior-week H/L, EQH/EQL clusters, round numbers), waits for a change-of-character (CHoCH) on the LTF, then enters on a retest of the order block or fair-value-gap zone. Stop beyond the sweep wick, target at the opposite paired session extreme.
2. **Continuation strategy** *(in progress)* — trend-following play. In a bullish HTF regime, enters longs on pullbacks to session VWAP with FVG confluence, buying pressure, and absorption at the retracement. Mirror for shorts.

## Why a custom engine

A vectorized backtester (compute everything in pandas at once) is fast but lookahead-prone: it's easy to accidentally use future bars in a rolling computation. Sequential ICT logic — "sweep, then within N bars wait for CHoCH, then locate OB, then wait for retest" — is hard to vectorize cleanly. We use a **bar-by-bar event loop**: at each bar `t`, the engine advances a cursor; every signal module sees only bars `0..t`. The `BarSeries` cursor makes lookahead structurally impossible.

## Quickstart

```bash
# Set up environment (one-time)
python3 -m venv .venv
.venv/bin/pip install -e .

# Run tests
.venv/bin/pytest

# Run backtest on the included NQ May-2026 data
.venv/bin/python scripts/run_backtest.py
```

## Repository layout

```
strategy/                 # All signal/decision modules. Pure logic, no I/O.
  params.py               # Single source of truth for all thresholds.
  bars.py                 # Bar dataclass + BarSeries with cursor.
  sessions.py             # Asia/London/NY/electronic session windows in ET.
  levels.py               # Reference levels (session H/L, EQH/EQL, opens, rounds).
  swings.py               # Fractal swing detector + CHoCH break helpers.
  fvg.py                  # Fair value gaps + order blocks.
  sweep.py                # Sweep detector (penetration + close-back + volume).
  sweep_state_machine.py  # Full sweep-trade lifecycle.
  vwap.py                 # Session-anchored VWAP.
  htf.py                  # Higher-timeframe aggregation + HH/HL trend.

backtest/                 # Driver, loader, metrics.
  engine.py               # Event loop wiring all modules together.
  metrics.py              # Per-cohort trade stats + breakdowns.
  data_loader.py          # Databento OHLCV-1m CSV.zst -> list[Bar].

tests/                    # 174 tests, all green.
scripts/                  # Runnable utilities.
  run_backtest.py         # Load real data, run backtest, print summary.
data/                     # Databento data dumps (gitignored).
```

## Module-by-module

### Foundation

**[strategy/bars.py](strategy/bars.py)** — `Bar` is an immutable OHLCV dataclass with tz-aware UTC timestamp. `BarSeries` holds a list of bars plus a cursor; `advance()` moves forward, `current`/`at(i)`/`window(n)`/`slice(a,b)` are the only access paths. Any attempt to read past the cursor raises `IndexError`. This is the lookahead guardrail.

**[strategy/sessions.py](strategy/sessions.py)** — `classify(ts)` returns `Session.ASIA` / `LONDON` / `NY` / `OTHER` for a UTC timestamp. Sessions are defined in ET wall-clock (DST-aware): Asia 19:00–03:00, London 03:00–08:00, NY 08:30–16:00 (with an intentional 08:00–08:30 gap per the strategy spec). Session day = 18:00 ET → 17:59 ET next day; futures week = Sunday 18:00 ET → Friday 17:00 ET.

**[strategy/params.py](strategy/params.py)** — every tunable threshold lives here as a frozen dataclass field. Change one number, change it everywhere. Includes NQ contract specs (`NQ_TICK_SIZE=0.25`, `NQ_POINT_VALUE=20.0`).

### Signal modules

**[strategy/levels.py](strategy/levels.py)** — `ReferenceLevels` is fed every bar via `on_bar(bar)`. After each bar, `active_levels()` returns all currently-tradable reference levels:
- Prior session H/L (full electronic, locked when next session opens)
- Asia H/L (locked at 03:00 ET)
- London H/L (locked at 08:00 ET)
- Prior week H/L (locked at next Sunday 18:00 ET)
- Daily open (first bar of current session day)
- Weekly open (first bar of current futures week)
- Round numbers, tiered: `ROUND_MAJOR` (every 100pt) + `ROUND_MINOR` (every 25pt, excluding majors); ±500 pts of current price
- EQH/EQL — clusters of ≥2 confirmed N=5 swing points within 2 ticks of each other, age-filtered to last 1 week

**[strategy/swings.py](strategy/swings.py)** — `SwingDetector(n)` is the fractal swing detector. Bar `i` is a swing high if its high is strictly greater than the highs of N bars on each side. A swing is confirmed when bar `i+N` arrives (causal). `most_recent(kind, before_confirmation_idx=t)` answers "what swings did the engine know about at bar t?" — returns only swings that were causally visible by then. Plus stateless `broke_swing_down(bar, swing, mode)` / `broke_swing_up(...)` helpers used for CHoCH detection (default mode: `CLOSE`).

**[strategy/fvg.py](strategy/fvg.py)** — `FVGDetector` finds 3-bar fair value gaps and tracks their state lifecycle (`ACTIVE` → `PARTIAL` → `FILLED`). Bullish FVG: `c3.low > c1.high`. Bearish: `c3.high < c1.low`. `find_order_block(bars, displacement_start_idx, direction, max_lookback)` walks backwards to find the OB — the last opposing-polarity candle before the displacement leg.

**[strategy/sweep.py](strategy/sweep.py)** — `SweepDetector` checks each bar's wick against every active reference level. A sweep requires: (1) fresh penetration from the original side, 3–8 ticks deep; (2) close-back-inside within `max_rejection_bars=3` (default); (3) rejection bar volume ≥ 1.5× trailing 20-bar SMA. Penetrations >10 ticks mark the level as broken. Once a level is swept or broken it's resolved permanently (within the detector instance).

**[strategy/sweep_state_machine.py](strategy/sweep_state_machine.py)** — the wiring. States:

```
IDLE → (sweep w/ paired target) → WATCHING_FOR_CHOCH
       → (CHoCH break within 15 bars) → WATCHING_FOR_ENTRY
       → (bar touches OB or active FVG, R:R ≥ 3:1) → PENDING_ENTRY
       → (next-bar-open fill) → IN_TRADE
       → (stop/target/invalidation) → PENDING_EXIT
       → (next-bar-open fill) → IDLE (Trade emitted)
```

Pending states consume a full bar each (signal at `t`, fill at `t+1` open). Within-bar chaining allowed for non-pending states. Single setup at a time in v1.

**[strategy/vwap.py](strategy/vwap.py)** — `SessionVWAP.on_bar(bar)` accumulates volume-weighted typical price; resets at 18:00 ET session boundary. `current` exposes the running VWAP. Used by the continuation strategy.

**[strategy/htf.py](strategy/htf.py)** — `HTFTrend(period="daily" | "4h", swing_n=2)` aggregates 1m bars into the chosen HTF bucket, runs a swing detector on the aggregated bars, and `current_trend()` returns `BULLISH` / `BEARISH` / `UNDEFINED` based on whether the last two confirmed swing highs AND lows are both rising or both falling. Buckets ET-aligned: 4h at 18/22/02/06/10/14, daily at 18:00 session boundary.

### Engine + metrics

**[backtest/engine.py](backtest/engine.py)** — `run_backtest(bars, params)`. The loop:
```
while bars.has_more:
    bars.advance()
    levels.on_bar(current)
    sweep_events = sweep_detector.on_bar(current, levels.active_levels())
    new_trades = state_machine.on_bar(current, levels.active_levels(), sweep_events)
```
Returns `BacktestResult(trades, metrics, bar_count, sweep_events)`.

**[backtest/metrics.py](backtest/metrics.py)** — `compute_metrics(trades) -> BacktestMetrics`. Overall `TradeStats` + breakdowns by `level_kind`, `direction`, `exit_reason`. Per-trade Sharpe (mean/stdev of trade PnLs). All values in **points** (×$20 for dollars).

**[backtest/data_loader.py](backtest/data_loader.py)** — `load_databento_ohlcv(path, symbol_filter)` reads Databento CSV.zst into `list[Bar]`. `symbol_filter='auto'` picks the highest-volume symbol (front-month) when the file contains multiple contracts.

## Strategy parameters (resolved)

| Param | Value | Notes |
|---|---|---|
| Prior session H/L | Full electronic (18:00 → 17:59 ET) | |
| Asia H/L window | 19:00–03:00 ET | locks at 03:00 |
| London H/L window | 03:00–08:00 ET | locks at 08:00 |
| Prior week H/L | Sun 18:00 ET → Fri 17:00 ET | |
| EQH/EQL swing lookback (HTF) | N=5 | strict-greater fractal |
| EQH/EQL tolerance | 2 ticks (0.50 pts) | cluster span |
| EQH/EQL max age | 1 week of 1m bars | |
| Round-number tiers | major 100pt, minor 25pt | majors override minors |
| Penetration valid | 3–8 ticks | per spec, highest reversion rate |
| Penetration broken | >10 ticks | level marked dead |
| Max rejection bars | 3 | sweep close-back window |
| Volume confirmation | ≥1.5× trailing 20-bar SMA | |
| LTF swing lookback | N=3 | for CHoCH structure |
| max_choch_bars | 15 (=15 min on 1m) | |
| max_entry_bars | 30 | after CHoCH |
| Stop buffer | 2 ticks beyond sweep wick | |
| Target (v1) | opposite paired session extreme | |
| min R:R | 3.0 | strict filter |
| Position size | 1 contract, no pyramiding | |
| Execution | next-bar-open fill | "strict no-lookahead" |
| HTF trend filter | log only, don't reject | per user choice |
| Exclude Asia shorts | on (`sweep_exclude_asia_shorts`) | only validated edge: Asia shorts win ~14% |
| News filter | not yet implemented | |
| VWAP anchor | 18:00 ET session open | |
| Daily bias | HH/HL on daily | |
| HTF trend timeframe | 4h, HH/HL | |

## Project status

**Completed (modules 1–10, 13)**:

✅ Foundation (Bar/BarSeries/sessions/params)
✅ Reference levels (incl. EQH/EQL)
✅ Swing + CHoCH detection
✅ FVG + order block
✅ Sweep detector
✅ Sweep state machine
✅ Event loop + metrics
✅ End-to-end smoke test on synthetic bars
✅ VWAP
✅ HTF aggregation + trend
✅ Databento OHLCV loader
✅ Buying pressure + absorption detectors (OHLCV approximations)
✅ Continuation state machine (parallel setup using VWAP + HTF + FVG + pressure)

**Known scope gap**:

⚠️ The v1 target rule only handles paired session extremes (Asia↔Asia, London↔London, etc.). On real May-2026 NQ data, this means **85% of detected sweeps are silently skipped** because they're at EQH/EQL/round-number levels (which match the spec's "highest density" stop clusters but have no paired counterpart). Fix is queued for after module 12.

## Test suite

174 tests, all green:

```
tests/test_bars.py                   foundation
tests/test_sessions.py               session classification
tests/test_levels.py                 reference levels (incl. EQH/EQL)
tests/test_swings.py                 swings + CHoCH
tests/test_fvg.py                    FVGs + order blocks
tests/test_sweep.py                  sweep detector
tests/test_sweep_state_machine.py    full sweep lifecycle (incl. happy paths)
tests/test_metrics.py                metric calculations
tests/test_engine.py                 engine smoke
tests/test_smoke.py                  end-to-end on synthetic bars
tests/test_data_loader.py            Databento loader
tests/test_vwap.py                   VWAP
tests/test_htf.py                    HTF aggregation + trend
tests/test_pressure.py               pressure + absorption (module 11)
tests/test_continuation.py           continuation state machine (module 12)
```

## Data

Pre-aggregated Databento NQ 1-min OHLCV bars (May 2026, 28,740 front-month bars) live under `data/` (gitignored). Format: CSV.zst from Databento's parent-symbol query (`NQ.FUT` with `stype_in=parent`). The loader handles the multi-symbol situation via `symbol_filter='auto'`.

## Continuing this work in a new Claude session

See `HANDOFF.md` for the cold-start brief.
