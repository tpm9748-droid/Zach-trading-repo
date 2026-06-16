"""Run the sweep-strategy backtest on real NQ data.

Usage (from repo root, with the venv activated or via .venv/bin/python):
    .venv/bin/python scripts/run_backtest.py
    .venv/bin/python scripts/run_backtest.py path/to/other.csv.zst
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `strategy.*` and `backtest.*` importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.data_loader import load_databento_ohlcv  # noqa: E402
from backtest.engine import run_backtest  # noqa: E402
from strategy.params import DEFAULT_PARAMS, NQ_POINT_VALUE  # noqa: E402


DEFAULT_DATA = (
    "data/GLBX-20260601-J9M5CGTAY5/glbx-mdp3-20260501-20260531.ohlcv-1m.csv.zst"
)


def fmt_pts(p: float) -> str:
    return f"{p:+.2f} pts (${p * NQ_POINT_VALUE:+,.0f})"


def main(path: str = DEFAULT_DATA) -> None:
    print(f"=== Sweep Backtest ===")
    print(f"Data: {path}")
    print()

    print("Loading bars...")
    bars = load_databento_ohlcv(path, symbol_filter="auto")
    print(f"  Loaded {len(bars):,} bars")
    print(f"  Range: {bars[0].ts} -> {bars[-1].ts}")
    print()

    print("Running backtest...")
    result = run_backtest(bars, DEFAULT_PARAMS, collect_sweep_events=True)
    m = result.metrics

    print()
    print("=== OVERALL ===")
    print(f"  Bars processed:  {result.bar_count:,}")
    print(f"  Sweep events:    {len(result.sweep_events):,}")
    print(f"  Trades:          {m.overall.n_trades}")

    if m.overall.n_trades == 0:
        print()
        print("(No trades fired. Sweep events without a trade:")
        print(f"  {len(result.sweep_events):,})")
        # Show the level-kind distribution of sweep events that didn't trade
        kind_counts: dict = {}
        for ev in result.sweep_events:
            kind_counts[ev.level.kind.value] = kind_counts.get(ev.level.kind.value, 0) + 1
        for k, n in sorted(kind_counts.items(), key=lambda kv: -kv[1]):
            print(f"    {k:20s}  {n:>5d}")
        return

    print(f"  Win rate:        {m.overall.win_rate:.1%}")
    print(f"  Total PnL:       {fmt_pts(m.overall.total_pnl_points)}")
    print(f"  Expectancy:      {m.overall.expectancy_points:+.2f} pts/trade")
    print(f"  Avg win:         {m.overall.avg_win:+.2f} pts")
    print(f"  Avg loss:        {m.overall.avg_loss:+.2f} pts")
    pf = m.overall.profit_factor
    pf_str = "inf" if pf == float("inf") else f"{pf:.2f}"
    print(f"  Profit factor:   {pf_str}")
    print(f"  Max drawdown:    {m.overall.max_drawdown_points:.2f} pts")
    if m.sharpe_per_trade is not None:
        print(f"  Sharpe/trade:    {m.sharpe_per_trade:.2f}")

    print()
    print("=== BY LEVEL KIND ===")
    sorted_kinds = sorted(m.by_level_kind.items(), key=lambda kv: -kv[1].total_pnl_points)
    for kind, stats in sorted_kinds:
        print(f"  {kind.value:20s} "
              f"n={stats.n_trades:>3d} "
              f"wr={stats.win_rate:>5.1%} "
              f"pnl={stats.total_pnl_points:+7.2f}")

    print()
    print("=== BY SETUP KIND ===")
    for kind, stats in sorted(m.by_setup_kind.items()):
        print(f"  {kind:12s} "
              f"n={stats.n_trades:>3d} "
              f"wr={stats.win_rate:>5.1%} "
              f"pnl={stats.total_pnl_points:+7.2f}")

    print()
    print("=== BY DIRECTION ===")
    for direction, stats in sorted(m.by_direction.items()):
        print(f"  {direction:5s} "
              f"n={stats.n_trades:>3d} "
              f"wr={stats.win_rate:>5.1%} "
              f"pnl={stats.total_pnl_points:+7.2f}")

    print()
    print("=== BY EXIT REASON ===")
    for reason, stats in sorted(m.by_exit_reason.items()):
        print(f"  {reason:12s} "
              f"n={stats.n_trades:>3d} "
              f"pnl={stats.total_pnl_points:+7.2f}")

    print()
    print(f"Sweep events without a trade: "
          f"{len(result.sweep_events) - m.overall.n_trades:,} "
          f"(setup invalidated by missing target / no CHoCH / no entry / low R:R)")

    # Optionally dump the trade log
    print()
    print("=== FIRST 10 TRADES ===")
    for t in result.trades[:10]:
        # Continuation trades carry no swept level; show setup kind instead.
        if t.swept_level_kind is not None:
            provenance = f"{t.swept_level_kind.value:18s} @{t.swept_level_price:>8.2f}"
        else:
            provenance = f"{t.setup_kind:18s} {'':>9s}"
        print(f"  {t.direction:5s} "
              f"{provenance} -> "
              f"entry {t.entry_price:>8.2f} "
              f"exit {t.exit_price:>8.2f} "
              f"({t.exit_reason:6s}) "
              f"{t.pnl_points:+6.2f} pts")
    if len(result.trades) > 10:
        print(f"  ... and {len(result.trades) - 10} more")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA
    main(path)
