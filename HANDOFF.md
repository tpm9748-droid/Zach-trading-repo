# Handoff — continuing this work in a new Claude session

This document is the cold-start brief. A new Claude session will not have any conversational memory of how we got here. Read this file plus README.md and you'll have the full picture.

## How to start the new session

Open a fresh Claude session in this repo (`/Users/thomasmurphy/Zach-trading-repo/` or wherever it's cloned). Paste this as the first message:

> Read HANDOFF.md and README.md. Then check `git log --oneline -10` and run `.venv/bin/pytest -q` to confirm we're at a green baseline. After that, summarize the project state back to me and wait for instructions.

That single message is enough — Claude can recover full context from those two files.

---

## What this project is

A bar-by-bar backtester for NASDAQ futures (NQ) implementing two ICT-style strategies (sweep reversal + VWAP-pullback continuation). See `README.md` for the full architectural overview and per-module breakdown.

## What's complete vs. what's left

**Done** — modules 1–13 (all built):
- Foundation, sessions, params, reference levels, swings/CHoCH, FVG/OB, sweep detector, sweep state machine, engine, metrics, smoke test, VWAP, HTF (daily + 4h), Databento loader.
- **Module 11** (pressure + absorption, OHLCV proxies) — `strategy/pressure.py`.
- **Module 12** (continuation state machine, parallel to sweep) — `strategy/continuation_state_machine.py`.
- **Target-rule expansion** — `_find_target_price` now falls back paired → EQH/EQL → ROUND_MAJOR → any opposite-side level, R:R-filtered, resolved at entry. Closed the ~85% sweep-skip gap.
- **Walk-forward harness** — `backtest/trades_loader.py` (tick trades → 1m bars, with aggressor delta) + `scripts/walkforward.py` (per-front-month-contract validation; add a config to its CONFIGS dict and it's tested across NQZ5/NQH6/NQM6 with May held out).
- **Order-flow delta** — `Bar.delta` + `sweep_use_delta_confirmation` (opt-in).

177 tests green. Six contracts of tick data live under `data/GLBX-20260601-LTGN97VDN8/` (Dec2025–May2026); the May OHLCV-1m file is `data/GLBX-20260601-J9M5CGTAY5/`.

## Research findings (validated via walk-forward; see memory `oos-validation-defaults-win`)

The strategy was rigorously validated OOS across 3 front-month contracts (NQZ5/NQH6/NQM6) with May held out. Baseline sweep win rate sits at ~24–25% = its 3:1 breakeven. **Most levers overfit May and failed OOS** (loosened thresholds, absorption continuation, exclude-all-Asia, OB-invalidation-off, continuation reclaim, HTF trend-alignment at both 4h and daily). **Three levers validated** (all OFF/opt-in except #1):

1. `sweep_exclude_asia_shorts=True` (**default on**) — Asia shorts are the consistent loser; excluding them lifted clean-OOS +166→+202 and held the May holdout.
2. `sweep_breakeven_at_r=1.0` (opt-in; **strongest**) — pull stop to entry after +1R. Combined OOS +35→**+143**, win 26%→36%, same trade count, and improves the May holdout (+75→+152). No tick data needed. Strongest candidate to promote to default (needs intrabar-fill care first — a wide bar hitting +R then returning can book a same-bar BE stop; would change the long happy-path target test).
3. `sweep_use_delta_confirmation=True` (opt-in, **tick-data only**) — aggressor delta opposing the sweep lifts win rate to ~37% (OOS +35→+255). No-op on OHLCV (delta=0). Caveat: ~halves trades; thin May (n=7, 0 wins). Best stacked config: be_1r + delta = OOS +287 / 47.8% win (but thin May −23).

**Statistical significance (`scripts/robustness.py`, bootstrap + Monte-Carlo).** Sobering but important: of the configs, ONLY `be_1r + delta_confirm` is statistically distinguishable from zero (P(expectancy>0) ≈ 96–97%, survives ~1pt/trade costs, ~4× lower drawdown than DEFAULTS). DEFAULTS (76%), be_1r (88%), delta alone (94%) all have CIs crossing 0 — leads, not proof. This config is codified as `RECOMMENDED_TICK_PARAMS` in params.py (tick-data only; n≈37, still marginal). Sample size is the wall — the reason more data is the gating need.

**Directional edge is beta, not tradeable** (alpha-vs-beta check): profit skewed long only because the sample was net up-trending; no trend filter captured it OOS. The validated levers all move WIN RATE (direction-neutral), so they are not beta artifacts.

**Continuation has no edge** in any tested variant and is shelved (inert by default: gates on ⇒ 0 trades).

Validation harness: `scripts/walkforward.py` — add a config to its CONFIGS dict and it's tested across contracts. All experiment configs from this work are still in there.

## Best next steps
- **More / newer tick data** is the gating need — confirm the delta + be_1r+delta edges and resolve the thin May/NQM6 windows before trusting live. The Dec2025–May2026 archive is fully used.
- **Promote `be_1r` to default** once intrabar-fill is modeled (it's the most robust win and needs no tick data).
- Keep delta confirmation parameter-free (sign only) — a magnitude threshold would be tuning on the same data.

## Critical context the next session needs

### Strategy spec
The user pasted a detailed strategy doc early in the conversation. Key concepts: liquidity sweeps at stop-cluster levels (HoD, LoD, EQH, EQL, round numbers), 3-stage sweep mechanic (penetration → rejection → volume confirmation), CHoCH for confirmation, OB/FVG for entry, stop beyond sweep wick, target at opposite session extreme. There's also a continuation idea bolted on: HTF bullish + daily bullish + retrace to VWAP + FVG at VWAP + absorption + buying pressure → continuation long. Mirror for short.

### Resolved design decisions (DO NOT re-decide unless user explicitly revisits)
- **Timezone**: ET (US/Eastern), DST-aware. All bar timestamps stored UTC, classified by ET wall clock.
- **Session day**: D 18:00 ET → D+1 17:59 ET (the 17:00–18:00 maintenance break belongs to the closing day).
- **Bar timeframe**: 1-minute primary, with HTF aggregation for daily + 4h trend.
- **Data vendor**: Databento (parent symbol NQ.FUT, pre-aggregated OHLCV-1m). Symbol filter `'auto'` selects front-month.
- **Execution model**: strict next-bar-open fills. Signal at bar T → fill at bar T+1 open. Pending states consume a full bar.
- **Concurrency**: ONE sweep setup at a time in v1. Future: two parallel state machines for bi-directional concurrency. The continuation state machine should be independent (can fire while a sweep trade is in progress).
- **Target rule** (sweep): opposite paired session extreme, must clear 3:1 R:R. (NEEDS EXPANSION — see scope gap above.)
- **Stop**: sweep wick extreme + 2 ticks (shorts) / - 2 ticks (longs).
- **CHoCH break mode**: `CLOSE` (stricter than wick).
- **Round numbers**: tiered — `ROUND_MAJOR` (100pt) overrides `ROUND_MINOR` (25pt). ±500pt of current price.
- **EQH/EQL**: ≥2 confirmed N=5 swings within 2 ticks, max age 1 week. Cluster price = mean.
- **LTF swing N**: 3 (for CHoCH).
- **HTF trend timeframe**: 4h (also tracked for daily bias).
- **HTF trend logic**: HH/HL → bullish, LH/LL → bearish, else undefined. Two latest swings of each kind.
- **VWAP**: session-anchored (resets 18:00 ET), typical = (h+l+c)/3.
- **Pressure/absorption (module 11)**: OHLCV approximations first; tick-based versions later if needed. The user's existing trades data (`data/GLBX-20260601-LTGN97VDN8/`) IS tick-level with aggressor side, so true delta is available — but module 11 should default to OHLCV proxies.
- **News filter**: not yet implemented; flag in trade log if added later.

### Conventions enforced throughout
- All metrics in **points** (multiply by NQ_POINT_VALUE = $20 for dollars).
- All thresholds in `strategy/params.py` as a single `StrategyParams` dataclass.
- Every signal module is incremental: `on_bar(bar)` per bar, no batch processing.
- Every module has its own unit test file in `tests/`.
- Hand-crafted bar sequences for tests (no random data).
- Strict-greater fractal rule (ties don't make swings).
- TZ-aware timestamps everywhere; `Bar.__post_init__` rejects naive timestamps.

## Files to read first in the new session

In this order:
1. `README.md` — architecture + module summaries
2. `HANDOFF.md` — this file
3. `strategy/params.py` — all tunable thresholds in one place
4. `strategy/sweep_state_machine.py` — the most consequential module; has detailed module-level docstring of state lifecycle
5. `tests/test_smoke.py` — the end-to-end happy path; shows how the modules wire together
6. `scripts/run_backtest.py` — how to drive the engine on real data
7. `strategy/vwap.py` and `strategy/htf.py` — the new (mod 9 + 10) building blocks that the continuation state machine will consume

## Running the existing backtest

```bash
cd /Users/thomasmurphy/Zach-trading-repo
.venv/bin/python scripts/run_backtest.py
```

Expected output today (May-2026 NQ data, sweep strategy only, before target-rule expansion):
```
Bars processed:  28,740
Sweep events:    107
Trades:          0
```
With 91 of 107 sweeps being EQH/EQL (no paired target → skipped). This is the scope gap, not a bug.

## Building modules 11 + 12 — direction

### Module 11: pressure + absorption
**File**: `strategy/pressure.py` (new)

Two per-bar metrics, both pure OHLCV approximations:

**Buying pressure** (per bar):
- `close_position_in_range` = (close - low) / (high - low) ∈ [0, 1]. Close-in-upper-third (≥ 0.67) = bullish bar.
- `body_fraction` = abs(close - open) / (high - low) ∈ [0, 1]. Larger body = stronger conviction.
- `body_vs_trailing_avg` = body_size / SMA(body_size, n=20). Relative-size proxy.

A clean "buying pressure" signal: close in upper third AND body_fraction ≥ 0.5 AND body_vs_trailing_avg ≥ 1.0. Mirror for selling pressure.

**Absorption** (per bar at a level):
- High volume (≥1.5× trailing 20-bar SMA, same threshold as sweep).
- Small range relative to recent (current_range < avg_range * 0.7, say).
- Bar overlaps the level being tested.

A bar that satisfies all three near a tested level = absorption.

Both detectors take optional aggressor-delta inputs (for future tick-based upgrade) but default to OHLCV proxies. Interface:
```python
class PressureDetector:
    def on_bar(self, bar: Bar, delta: Optional[float] = None) -> PressureReading: ...

class AbsorptionDetector:
    def on_bar(self, bar: Bar, near_level: Optional[float] = None, delta: Optional[float] = None) -> AbsorptionReading: ...
```

Tests: hand-crafted bar sequences with known close-position / body / volume profiles.

### Module 12: continuation state machine
**File**: `strategy/continuation_state_machine.py` (new)

Independent of `sweep_state_machine.py`. Drives its own state lifecycle:

```
IDLE
  → (HTF bullish AND price retraces down toward VWAP) → WATCHING_FOR_RETEST
  → (bar touches or dips below VWAP, with FVG at VWAP, with buying pressure, with absorption) → PENDING_ENTRY
  → (next-bar-open fill) → IN_TRADE
  → (stop hit / target hit / HTF flips) → PENDING_EXIT
  → (next-bar-open fill) → IDLE (Trade with setup_kind="continuation")
```

Mirror for shorts.

**Engine integration**: `backtest/engine.py` currently runs only the sweep state machine. Add the continuation state machine as a parallel module — call `continuation_sm.on_bar(...)` after `sweep_sm.on_bar(...)`, collect its trades into the same trade list. Distinguish via `Trade.setup_kind`. The `BacktestMetrics` already has per-direction and per-level-kind breakdowns; add a `by_setup_kind` breakdown.

**Stop/target for continuation**:
- Stop: below the swing low that formed before the VWAP touch (long); above the swing high (short). Use the LTF swing detector.
- Target: next opposite HTF level (the next HTF resistance for long, support for short). For a first cut, use the most recent HTF swing high (for long) at +N points away that gives ≥ 3:1 R:R.

**R:R filter**: same `min_rr_ratio = 3.0` from params.

### After module 12: target-rule expansion
**File to edit**: `strategy/sweep_state_machine.py`, function `_find_target_price`.

Current logic only checks `PAIRED_OPPOSITES`. Replace with a priority chain:
1. Paired session extreme (current behavior).
2. Nearest opposite-direction EQH/EQL in `active_levels` that gives ≥ min_rr R:R.
3. Nearest opposite-direction ROUND_MAJOR that gives ≥ min_rr R:R.
4. Fallback: nearest opposite-direction level of any kind that gives ≥ min_rr R:R.

Pick the first that clears R:R. Then run the backtest again — expect many more trades.

## Test status at handoff

```
177 passed
```

Last commit before handoff (push after writing this file):

```
<filled in after push>
```

## How the user works

- Wants pause-and-review at each module before continuing.
- Prefers strict execution rules (no lookahead, next-bar fills) over realism (intrabar fills, slippage modeling) for v1.
- Open to iteration and refinement — make defaults reasonable, expose params for tuning.
- Doesn't want fake-data tuning — only commit to design changes based on real-data evidence.
- The user's GitHub: `tpm9748-droid/Zach-trading-repo`, branch `main`.

## What NOT to do

- Don't refactor working code without a reason.
- Don't add features the user hasn't asked for.
- Don't commit large data files (`.gitignore` handles this — keep it in place).
- Don't run a backtest with synthetic bars and report it as if it's a real result.
- Don't change the strict no-lookahead invariants in `BarSeries`.
- Don't merge the sweep and continuation state machines — they're meant to run in parallel.

## Open questions awaiting user input

1. The 85% sweep-skip rate from the target-rule gap — fix it before module 12, after module 12, or after the smoke run with both setups? (Recommendation: after module 12.)
2. Module 11 needs explicit thresholds for "high volume," "small range," "strong body." Defaults proposed above; user may want to tune.
3. Continuation stop placement isn't fully specified — using "LTF swing before the retest" as a reasonable default but should confirm.
