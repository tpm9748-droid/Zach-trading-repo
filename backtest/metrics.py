"""Backtest metrics — overall and breakdowns.

All metrics are computed in **points** (not dollars). Multiply by
NQ_POINT_VALUE (=$20) for dollar conversion.

Per-trade Sharpe (mean/stdev of trade PnLs) is the v1 risk-adjusted
measure. With infrequent intraday trades it's more honest than a
time-weighted Sharpe; if we later have hundreds of trades per period we'll
add an annualized variant.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Optional

from strategy.levels import LevelKind
from strategy.sweep_state_machine import Trade


@dataclass(frozen=True)
class TradeStats:
    n_trades: int = 0
    n_wins: int = 0
    n_losses: int = 0
    n_scratches: int = 0  # pnl exactly 0
    win_rate: float = 0.0
    total_pnl_points: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy_points: float = 0.0
    max_drawdown_points: float = 0.0


@dataclass(frozen=True)
class BacktestMetrics:
    overall: TradeStats
    by_level_kind: dict[LevelKind, TradeStats] = field(default_factory=dict)
    by_direction: dict[str, TradeStats] = field(default_factory=dict)
    by_exit_reason: dict[str, TradeStats] = field(default_factory=dict)
    by_setup_kind: dict[str, TradeStats] = field(default_factory=dict)
    sharpe_per_trade: Optional[float] = None


def _stats_for(trades: list[Trade]) -> TradeStats:
    if not trades:
        return TradeStats()
    pnls = [t.pnl_points for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    scratches = [p for p in pnls if p == 0]
    n = len(trades)
    total = sum(pnls)
    gross_loss = -sum(losses)  # positive number
    gross_win = sum(wins)

    # Max drawdown over the equity curve formed by sequential trade PnLs.
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        equity += p
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

    if gross_loss == 0:
        # No losses: undefined classically; report inf when there ARE wins,
        # else 0 (no profit either).
        profit_factor = math.inf if gross_win > 0 else 0.0
    else:
        profit_factor = gross_win / gross_loss

    return TradeStats(
        n_trades=n,
        n_wins=len(wins),
        n_losses=len(losses),
        n_scratches=len(scratches),
        win_rate=len(wins) / n,
        total_pnl_points=total,
        avg_win=(gross_win / len(wins)) if wins else 0.0,
        avg_loss=(sum(losses) / len(losses)) if losses else 0.0,
        profit_factor=profit_factor,
        expectancy_points=total / n,
        max_drawdown_points=max_dd,
    )


def compute_metrics(trades: list[Trade]) -> BacktestMetrics:
    overall = _stats_for(trades)

    by_kind: dict[LevelKind, TradeStats] = {}
    by_dir: dict[str, TradeStats] = {}
    by_reason: dict[str, TradeStats] = {}
    by_setup: dict[str, TradeStats] = {}
    if trades:
        kinds: dict[LevelKind, list[Trade]] = {}
        dirs: dict[str, list[Trade]] = {}
        reasons: dict[str, list[Trade]] = {}
        setups: dict[str, list[Trade]] = {}
        for t in trades:
            # by_level_kind is sweep-only; continuation trades carry no level.
            if t.swept_level_kind is not None:
                kinds.setdefault(t.swept_level_kind, []).append(t)
            dirs.setdefault(t.direction, []).append(t)
            reasons.setdefault(t.exit_reason, []).append(t)
            setups.setdefault(t.setup_kind, []).append(t)
        by_kind = {k: _stats_for(v) for k, v in kinds.items()}
        by_dir = {k: _stats_for(v) for k, v in dirs.items()}
        by_reason = {k: _stats_for(v) for k, v in reasons.items()}
        by_setup = {k: _stats_for(v) for k, v in setups.items()}

    sharpe: Optional[float] = None
    if len(trades) > 1:
        pnls = [t.pnl_points for t in trades]
        sd = statistics.pstdev(pnls)  # population stdev — small-N safe
        if sd > 0:
            sharpe = statistics.fmean(pnls) / sd

    return BacktestMetrics(
        overall=overall,
        by_level_kind=by_kind,
        by_direction=by_dir,
        by_exit_reason=by_reason,
        by_setup_kind=by_setup,
        sharpe_per_trade=sharpe,
    )
