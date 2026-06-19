"""Walk-forward validation across front-month contracts.

Builds independent per-contract 1m samples from the Dec-2025 trades archive
and runs a parameter config against each, so any change can be checked for
robustness across periods/regimes instead of a single month.

    .venv/bin/python scripts/walkforward.py            # DEFAULTS
    .venv/bin/python scripts/walkforward.py sweep_cand # a named config below

Contracts (front-month, by traded volume in the archive):
    NQH6  Dec-2025 -> Mar-2026   clean out-of-sample
    NQM6  Mar-2026 -> Jun-2026   overlaps the May in-sample month (labelled)
    NQZ5  partial (archive starts Dec-1, expiry ~Dec-19)  small sample
"""
from __future__ import annotations

import os
import pickle
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.engine import run_backtest  # noqa: E402
from backtest.trades_loader import load_trades_dir_ohlcv  # noqa: E402
from strategy.params import DEFAULT_PARAMS  # noqa: E402

TRADES_DIR = "data/GLBX-20260601-LTGN97VDN8"
CONTRACTS = ["NQH6", "NQM6", "NQZ5"]
CACHE = "scripts/_wf_bars.pkl"

# Front-month windows: each contract is used ONLY during the dates it is the
# liquid front month, so the samples don't overlap in calendar time and each
# market period is counted once. May-2026 (NQM6) is the in-sample month used
# for tuning — labelled and excluded from the OOS verdict.
# (contract, start_date_incl, end_date_excl, label, is_oos)
WINDOWS = [
    ("NQZ5", None,         "2025-12-18", "Dec (partial tail)",   True),
    ("NQH6", "2025-12-18", "2026-03-19", "Dec18-Mar19",          True),
    ("NQM6", "2026-03-19", "2026-05-01", "Mar19-Apr30",          True),
    ("NQM6", "2026-05-01", "2026-06-01", "May (IN-SAMPLE)",      False),
]


def _slice(bars, start, end):
    import pandas as pd
    s = pd.Timestamp(start, tz="UTC") if start else None
    e = pd.Timestamp(end, tz="UTC") if end else None
    return [b for b in bars if (s is None or b.ts >= s) and (e is None or b.ts < e)]

# Named configs to validate. Add new candidate changes here — they get tested
# across every contract automatically.
_CONT_NOGATE = dict(
    cont_require_pressure=False, cont_require_absorption=False, cont_require_fvg_at_vwap=False,
)
CONFIGS = {
    "DEFAULTS": dict(),
    "sweep_cand": dict(volume_mult=1.2, penetration_min_ticks=2, ltf_swing_lookback=2),
    # Structural sweep test: turn off the OB-close invalidation exit.
    "no_ob_invalid": dict(sweep_use_ob_invalidation=False),
    # Continuation redesign: VWAP touch vs reclaim (gates off to isolate the trigger).
    "cont_touch_nogate": {**_CONT_NOGATE, "cont_entry_mode": "touch"},
    "cont_reclaim_nogate": {**_CONT_NOGATE, "cont_entry_mode": "reclaim"},
}


def get_bars() -> dict[str, list]:
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    print("Aggregating trades -> 1m bars per contract (one-time)...")
    bars = load_trades_dir_ohlcv(TRADES_DIR, CONTRACTS)
    with open(CACHE, "wb") as f:
        pickle.dump(bars, f)
    return bars


def stats(trades, kind):
    ts = [t for t in trades if t.setup_kind == kind]
    n = len(ts)
    if n == 0:
        return 0, 0.0, 0.0
    w = sum(1 for t in ts if t.pnl_points > 0)
    return n, w / n, sum(t.pnl_points for t in ts)


def line(label, n, wr, pnl):
    if n == 0:
        return f"  {label:24s}  (no trades)"
    return f"  {label:24s}  n={n:>4} win={wr:>5.1%} pnl={pnl:>+9.2f}  ({pnl / n:+.2f}/trade)"


def main(config_name: str) -> None:
    if config_name not in CONFIGS:
        print(f"Unknown config {config_name!r}. Options: {list(CONFIGS)}")
        sys.exit(1)
    params = replace(DEFAULT_PARAMS, **CONFIGS[config_name])
    bars_by_sym = get_bars()

    print(f"\n=== WALK-FORWARD: config '{config_name}' ===")
    oos = {"sweep": [0, 0, 0.0], "continuation": [0, 0, 0.0]}  # n, wins, pnl (OOS only)
    for sym, start, end, label, is_oos in WINDOWS:
        bars = _slice(bars_by_sym.get(sym, []), start, end)
        tag = "OOS" if is_oos else "in-sample"
        if not bars:
            print(f"\n{sym} {label} [{tag}]: (no bars)")
            continue
        res = run_backtest(bars, params)
        print(f"\n{sym} {label} [{tag}]  ({bars[0].ts.date()} -> {bars[-1].ts.date()}, {len(bars):,} bars)")
        for kind in ("sweep", "continuation"):
            n, wr, pnl = stats(res.trades, kind)
            print(line(kind, n, wr, pnl))
            if is_oos:
                oos[kind][0] += n
                oos[kind][2] += pnl
                oos[kind][1] += round(wr * n)

    print("\n--- COMBINED OUT-OF-SAMPLE (excludes May in-sample) ---")
    for kind in ("sweep", "continuation"):
        n, wins, pnl = oos[kind]
        wr = wins / n if n else 0.0
        print(line(kind, n, wr, pnl))


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "DEFAULTS"
    names = list(CONFIGS) if arg == "ALL" else arg.split(",")
    for name in names:
        main(name)
