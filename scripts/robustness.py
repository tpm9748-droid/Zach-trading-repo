"""Statistical robustness of a config's out-of-sample edge.

For each named config, pools the sweep trades across the clean OOS windows
(NQH6 + NQM6 Mar–Apr; the degenerate NQZ5 tail and the in-sample May are
excluded) and reports:

  - bootstrap 95% CI on per-trade expectancy, and P(expectancy > 0)
  - bootstrap 95% CI on win rate
  - Monte-Carlo max-drawdown distribution (shuffled trade order)

The point: with only ~50–80 trades, an OOS total can look good by luck. If the
expectancy CI straddles 0 (P(>0) well under ~0.95), the "edge" is not
statistically established at this sample size — treat it as a lead, not proof.

Costs: pass --cost_pts to subtract a per-trade round-trip cost (slippage +
commission, in points) before the stats, to see the NET edge.

Usage:
    .venv/bin/python scripts/robustness.py
    .venv/bin/python scripts/robustness.py --cost_pts 0.75
"""
from __future__ import annotations

import argparse
import pickle
from dataclasses import replace

import numpy as np
import pandas as pd

from backtest.engine import run_backtest
from strategy.params import DEFAULT_PARAMS

CACHE = "scripts/_wf_bars.pkl"
CLEAN_OOS = [
    ("NQH6", "2025-12-18", "2026-03-19"),
    ("NQM6", "2026-03-19", "2026-05-01"),
]
CONFIGS = {
    "DEFAULTS": dict(),
    "be_1r": dict(sweep_breakeven_at_r=1.0),
    "delta_confirm": dict(sweep_use_delta_confirmation=True),
    "be_1r_delta": dict(sweep_breakeven_at_r=1.0, sweep_use_delta_confirmation=True),
}
N_BOOT = 5000
RNG = np.random.default_rng(42)


def _slice(bars, start, end):
    s, e = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    return [b for b in bars if s <= b.ts < e]


def oos_pnls(bars_by_sym, overrides):
    """Raw (gross) per-trade sweep PnLs pooled over the clean OOS windows."""
    params = replace(DEFAULT_PARAMS, **overrides)
    pnls = []
    for sym, s, e in CLEAN_OOS:
        res = run_backtest(_slice(bars_by_sym[sym], s, e), params)
        pnls += [t.pnl_points for t in res.trades if t.setup_kind == "sweep"]
    return np.array(pnls, dtype=float)


def _stats(pnls):
    n = len(pnls)
    idx = RNG.integers(0, n, size=(N_BOOT, n))
    s = pnls[idx]
    means, wins = s.mean(axis=1), (s > 0).mean(axis=1)
    # Vectorized Monte-Carlo max-drawdown over shuffled trade order.
    perms = np.argsort(RNG.random((2000, n)), axis=1)
    eq = np.cumsum(pnls[perms], axis=1)
    dd = (np.maximum.accumulate(eq, axis=1) - eq).max(axis=1)
    return means, wins, dd


def _row(name, pnls):
    n = len(pnls)
    if n < 2:
        return f"{name:14} {n:>3}  (too few trades)"
    means, wins, dd = _stats(pnls)
    m_lo, m_hi = np.percentile(means, [2.5, 97.5])
    w_lo, w_hi = np.percentile(wins, [2.5, 97.5])
    dd50, dd95 = np.percentile(dd, [50, 95])
    return (f"{name:14} {n:>3} {pnls.sum():>+8.1f} {pnls.mean():>+7.2f} "
            f"[{m_lo:>+6.2f},{m_hi:>+6.2f}] {(means > 0).mean():>6.1%} "
            f"[{w_lo:>4.0%},{w_hi:>4.0%}] {dd50:>6.0f}/{dd95:>6.0f}")


def main(cost_levels: list[float]) -> None:
    bars_by_sym = pickle.load(open(CACHE, "rb"))
    gross = {name: oos_pnls(bars_by_sym, ov) for name, ov in CONFIGS.items()}
    print(f"Clean OOS = NQH6 + NQM6(Mar-Apr).  bootstrap N={N_BOOT}.")
    for cost in cost_levels:
        print(f"\n=== cost/trade = {cost:.2f} pts ===")
        print(f"{'config':14} {'n':>3} {'pnl':>8} {'/trade':>7} "
              f"{'exp 95% CI':>16} {'P(>0)':>6} {'win 95% CI':>13} {'maxDD 50/95':>14}")
        for name, p in gross.items():
            print(_row(name, p - cost))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--costs", type=float, nargs="+", default=[0.0, 0.5, 0.75, 1.0])
    main(ap.parse_args().costs)
