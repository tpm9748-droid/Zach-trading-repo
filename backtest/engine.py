"""Event-loop backtest engine.

Coordinates one bar at a time through the full pipeline:

    bars cursor advance
    -> levels.on_bar          (update reference-level state)
    -> sweep_detector.on_bar  (consume active levels, emit sweep events)
    -> state_machine.on_bar   (consume sweeps, drive trade lifecycle)
    -> collect any closed Trades

That's it. No vectorization, no lookahead — each module only sees bars at
or before the current cursor.

Continuation setups will plug in alongside the sweep state machine in
module 12. The pattern stays the same: one more on_bar call per loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from strategy.bars import Bar, BarSeries
from strategy.levels import ReferenceLevels
from strategy.params import StrategyParams
from strategy.sweep import SweepDetector, SweepEvent
from strategy.sweep_state_machine import SweepStateMachine, Trade

from backtest.metrics import BacktestMetrics, compute_metrics


@dataclass
class BacktestResult:
    trades: list[Trade]
    metrics: BacktestMetrics
    bar_count: int
    sweep_events: list[SweepEvent] = field(default_factory=list)


def run_backtest(
    bars: BarSeries | list[Bar] | Iterable[Bar],
    params: StrategyParams,
    collect_sweep_events: bool = False,
) -> BacktestResult:
    """Drive the full pipeline. Returns a BacktestResult with trades + metrics.

    If `collect_sweep_events` is True, all sweep events are retained in the
    result for inspection (useful for debugging or for measuring how many
    sweeps fired vs. converted to trades).
    """
    if not isinstance(bars, BarSeries):
        bar_list = list(bars)
        if not bar_list:
            return BacktestResult(trades=[], metrics=compute_metrics([]), bar_count=0)
        bars = BarSeries(bar_list)

    levels = ReferenceLevels(params)
    sweep_detector = SweepDetector(params)
    state_machine = SweepStateMachine(params)

    all_trades: list[Trade] = []
    all_sweeps: list[SweepEvent] = []

    while bars.has_more:
        bars.advance()
        bar = bars.current
        levels.on_bar(bar)
        active_levels = levels.active_levels()
        sweep_events = sweep_detector.on_bar(bar, active_levels)
        if collect_sweep_events and sweep_events:
            all_sweeps.extend(sweep_events)
        new_trades = state_machine.on_bar(bar, active_levels, sweep_events)
        if new_trades:
            all_trades.extend(new_trades)

    return BacktestResult(
        trades=all_trades,
        metrics=compute_metrics(all_trades),
        bar_count=len(bars),
        sweep_events=all_sweeps,
    )
