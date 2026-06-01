"""Metrics tests."""
from __future__ import annotations

import math
from datetime import date

import pandas as pd
import pytest

from backtest.metrics import BacktestMetrics, TradeStats, compute_metrics
from strategy.levels import LevelKind
from strategy.sweep_state_machine import Trade


def _trade(pnl: float, *, kind: LevelKind = LevelKind.ASIA_HIGH,
           direction: str = "short", reason: str = "target") -> Trade:
    """Construct a Trade with only the fields metrics needs."""
    ts = pd.Timestamp("2026-06-01 09:30", tz="UTC")
    return Trade(
        setup_kind="sweep", direction=direction,
        swept_level_kind=kind, swept_level_price=100.0,
        sweep_bar_idx=0, choch_bar_idx=1, entry_bar_idx=2,
        entry_price=100.0, stop_price=101.0, target_price=90.0,
        exit_bar_idx=3, exit_price=100.0 - (pnl if direction == "short" else -pnl),
        exit_reason=reason, pnl_points=pnl, rr_at_entry=3.0,
    )


def test_empty_trades_produces_zero_stats():
    m = compute_metrics([])
    assert m.overall == TradeStats()
    assert m.by_level_kind == {}
    assert m.by_direction == {}
    assert m.by_exit_reason == {}
    assert m.sharpe_per_trade is None


def test_all_winners():
    trades = [_trade(5), _trade(10), _trade(3)]
    m = compute_metrics(trades)
    assert m.overall.n_trades == 3
    assert m.overall.n_wins == 3
    assert m.overall.n_losses == 0
    assert m.overall.win_rate == 1.0
    assert m.overall.total_pnl_points == 18
    assert m.overall.avg_win == 6.0
    assert m.overall.avg_loss == 0.0
    assert m.overall.profit_factor == math.inf
    assert m.overall.expectancy_points == 6.0
    assert m.overall.max_drawdown_points == 0.0


def test_all_losers():
    trades = [_trade(-2), _trade(-5)]
    m = compute_metrics(trades)
    assert m.overall.n_wins == 0
    assert m.overall.n_losses == 2
    assert m.overall.win_rate == 0.0
    assert m.overall.total_pnl_points == -7
    assert m.overall.avg_loss == -3.5
    assert m.overall.profit_factor == 0.0


def test_mixed():
    trades = [_trade(10), _trade(-3), _trade(5), _trade(-2)]
    m = compute_metrics(trades)
    assert m.overall.n_wins == 2
    assert m.overall.n_losses == 2
    assert m.overall.win_rate == 0.5
    assert m.overall.avg_win == 7.5
    assert m.overall.avg_loss == -2.5
    assert m.overall.profit_factor == pytest.approx(15 / 5)
    assert m.overall.expectancy_points == 2.5


def test_max_drawdown():
    """Equity: 0 -> 10 -> 5 -> 15 -> 3 -> 8.
    Peak after trade 3 is 15; trough is 3. Drawdown = 12."""
    trades = [_trade(10), _trade(-5), _trade(10), _trade(-12), _trade(5)]
    m = compute_metrics(trades)
    assert m.overall.max_drawdown_points == 12.0


def test_breakdown_by_level_kind():
    trades = [
        _trade(10, kind=LevelKind.ASIA_HIGH),
        _trade(-3, kind=LevelKind.ASIA_HIGH),
        _trade(5, kind=LevelKind.LONDON_HIGH),
    ]
    m = compute_metrics(trades)
    assert set(m.by_level_kind.keys()) == {LevelKind.ASIA_HIGH, LevelKind.LONDON_HIGH}
    assert m.by_level_kind[LevelKind.ASIA_HIGH].n_trades == 2
    assert m.by_level_kind[LevelKind.ASIA_HIGH].total_pnl_points == 7
    assert m.by_level_kind[LevelKind.LONDON_HIGH].n_trades == 1
    assert m.by_level_kind[LevelKind.LONDON_HIGH].total_pnl_points == 5


def test_breakdown_by_direction():
    trades = [
        _trade(5, direction="short"),
        _trade(3, direction="long"),
        _trade(-2, direction="short"),
    ]
    m = compute_metrics(trades)
    assert m.by_direction["short"].n_trades == 2
    assert m.by_direction["long"].n_trades == 1


def test_breakdown_by_exit_reason():
    trades = [
        _trade(10, reason="target"),
        _trade(-3, reason="stop"),
        _trade(0, reason="invalidated"),
    ]
    m = compute_metrics(trades)
    assert set(m.by_exit_reason.keys()) == {"target", "stop", "invalidated"}
    assert m.by_exit_reason["target"].n_trades == 1
    assert m.by_exit_reason["stop"].n_trades == 1
    assert m.by_exit_reason["invalidated"].n_trades == 1
    assert m.by_exit_reason["invalidated"].n_scratches == 1


def test_sharpe_with_multiple_trades():
    """Sharpe is None for <2 trades; finite for multiple varied trades."""
    assert compute_metrics([_trade(5)]).sharpe_per_trade is None
    trades = [_trade(5), _trade(10), _trade(-3), _trade(8)]
    m = compute_metrics(trades)
    assert m.sharpe_per_trade is not None
    # All-identical pnls would give zero stdev -> None
    flat = compute_metrics([_trade(5), _trade(5), _trade(5)])
    assert flat.sharpe_per_trade is None
