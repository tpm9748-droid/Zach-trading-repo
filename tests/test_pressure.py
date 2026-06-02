"""Pressure + absorption detector tests (module 11).

Hand-crafted bars with known close-position / body / volume / range
profiles. Trailing-average metrics need a warmup of `window` bars before
they return a value, so each signal test feeds a neutral warmup first.
"""
from __future__ import annotations

import pandas as pd
import pytest

from strategy.bars import Bar
from strategy.params import DEFAULT_PARAMS
from strategy.pressure import (
    AbsorptionDetector,
    PressureDetector,
)

BASE = pd.Timestamp("2026-06-01 18:00").tz_localize("US/Eastern").tz_convert("UTC")


def bar(o: float, h: float, l: float, c: float, v: float = 100.0, i: int = 0) -> Bar:
    # i only needs to keep timestamps tz-aware; the detectors don't enforce ordering.
    return Bar(ts=BASE + pd.Timedelta(minutes=i), open=o, high=h, low=l, close=c, volume=v)


# --- PressureDetector ------------------------------------------------------

def _warm_pressure(det: PressureDetector, n: int = 20, body: float = 1.0) -> None:
    """Feed n bars each with body == `body` so the trailing body avg is known."""
    for k in range(n):
        det.on_bar(bar(100.0, 100.0 + body, 100.0, 100.0 + body, i=k))


def test_pressure_metrics_computed_on_first_bar():
    det = PressureDetector(DEFAULT_PARAMS)
    # range 2, close at 101.8 -> close_pos 0.9 ; body 1.8 -> frac 0.9
    r = det.on_bar(bar(100.0, 102.0, 100.0, 101.8))
    assert r.close_position_in_range == pytest.approx(0.9)
    assert r.body_fraction == pytest.approx(0.9)
    # No history yet -> relative-size unknown -> no confirmed signal.
    assert r.body_vs_trailing_avg is None
    assert r.buying_pressure is False
    assert r.selling_pressure is False


def test_buying_pressure_signal():
    det = PressureDetector(DEFAULT_PARAMS)
    _warm_pressure(det, body=1.0)
    # Strong bullish bar: close upper third, fat body, bigger than avg body.
    r = det.on_bar(bar(100.0, 102.0, 100.0, 101.8, i=99))
    assert r.body_vs_trailing_avg == pytest.approx(1.8)  # body 1.8 vs avg 1.0
    assert r.buying_pressure is True
    assert r.selling_pressure is False


def test_selling_pressure_signal():
    det = PressureDetector(DEFAULT_PARAMS)
    _warm_pressure(det, body=1.0)
    # Strong bearish bar: close lower third, fat body, bigger than avg.
    r = det.on_bar(bar(100.0, 100.0, 98.0, 98.2, i=99))
    assert r.close_position_in_range == pytest.approx(0.1)
    assert r.selling_pressure is True
    assert r.buying_pressure is False


def test_no_pressure_when_body_too_small():
    det = PressureDetector(DEFAULT_PARAMS)
    _warm_pressure(det, body=1.0)
    # Close in upper third but tiny body (indecision) -> no signal.
    r = det.on_bar(bar(101.5, 102.0, 100.0, 101.8, i=99))
    assert r.close_position_in_range == pytest.approx(0.9)
    assert r.body_fraction == pytest.approx(0.15)
    assert r.buying_pressure is False


def test_no_pressure_when_not_bigger_than_average():
    det = PressureDetector(DEFAULT_PARAMS)
    _warm_pressure(det, body=1.0)
    # Clean bullish geometry but body (0.55) smaller than trailing avg (1.0).
    r = det.on_bar(bar(100.0, 100.6, 100.0, 100.55, i=99))
    assert r.body_vs_trailing_avg == pytest.approx(0.55)
    assert r.buying_pressure is False


def test_zero_range_bar_is_neutral():
    det = PressureDetector(DEFAULT_PARAMS)
    _warm_pressure(det, body=1.0)
    r = det.on_bar(bar(100.0, 100.0, 100.0, 100.0, i=99))
    assert r.close_position_in_range == pytest.approx(0.5)
    assert r.body_fraction == pytest.approx(0.0)
    assert r.buying_pressure is False
    assert r.selling_pressure is False


def test_delta_gate_blocks_contradicting_buy():
    det = PressureDetector(DEFAULT_PARAMS)
    _warm_pressure(det, body=1.0)
    # Geometry says buying, but aggressor delta is negative -> gated off.
    r = det.on_bar(bar(100.0, 102.0, 100.0, 101.8, i=99), delta=-50.0)
    assert r.delta == pytest.approx(-50.0)
    assert r.buying_pressure is False


def test_delta_confirms_buy():
    det = PressureDetector(DEFAULT_PARAMS)
    _warm_pressure(det, body=1.0)
    r = det.on_bar(bar(100.0, 102.0, 100.0, 101.8, i=99), delta=50.0)
    assert r.buying_pressure is True


# --- AbsorptionDetector ----------------------------------------------------

def _warm_absorption(det: AbsorptionDetector, n: int = 20, vol: float = 100.0,
                     rng: float = 2.0) -> None:
    """Feed n bars each with volume == vol and range == rng."""
    for k in range(n):
        det.on_bar(bar(100.0, 100.0 + rng, 100.0, 100.0 + rng / 2, v=vol, i=k))


def test_absorption_metrics_none_before_warmup():
    det = AbsorptionDetector(DEFAULT_PARAMS)
    r = det.on_bar(bar(100.0, 101.0, 99.0, 100.0, v=200.0), near_level=100.0)
    assert r.volume_ratio is None
    assert r.range_ratio is None
    assert r.absorption is False


def test_absorption_signal():
    det = AbsorptionDetector(DEFAULT_PARAMS)
    _warm_absorption(det, vol=100.0, rng=2.0)
    # High volume (2x), small range (0.5x), sitting on the level.
    r = det.on_bar(bar(100.0, 100.5, 99.5, 100.0, v=200.0, i=99), near_level=100.0)
    assert r.volume_ratio == pytest.approx(2.0)
    assert r.range_ratio == pytest.approx(0.5)
    assert r.overlaps_level is True
    assert r.absorption is True


def test_no_absorption_without_a_level():
    det = AbsorptionDetector(DEFAULT_PARAMS)
    _warm_absorption(det)
    r = det.on_bar(bar(100.0, 100.5, 99.5, 100.0, v=200.0, i=99), near_level=None)
    assert r.overlaps_level is False
    assert r.absorption is False


def test_no_absorption_when_range_not_small():
    det = AbsorptionDetector(DEFAULT_PARAMS)
    _warm_absorption(det, rng=2.0)
    # High volume but range equals the trailing average (ratio 1.0, not < 0.7).
    r = det.on_bar(bar(100.0, 101.0, 99.0, 100.0, v=200.0, i=99), near_level=100.0)
    assert r.range_ratio == pytest.approx(1.0)
    assert r.absorption is False


def test_no_absorption_when_volume_not_high():
    det = AbsorptionDetector(DEFAULT_PARAMS)
    _warm_absorption(det, vol=100.0)
    r = det.on_bar(bar(100.0, 100.5, 99.5, 100.0, v=100.0, i=99), near_level=100.0)
    assert r.volume_ratio == pytest.approx(1.0)
    assert r.absorption is False


def test_overlaps_level_is_inclusive_at_high():
    det = AbsorptionDetector(DEFAULT_PARAMS)
    _warm_absorption(det)
    r = det.on_bar(bar(100.0, 100.5, 99.5, 100.0, v=200.0, i=99), near_level=100.5)
    assert r.overlaps_level is True
    assert r.absorption is True


def test_absorption_carries_delta_through():
    det = AbsorptionDetector(DEFAULT_PARAMS)
    _warm_absorption(det)
    r = det.on_bar(bar(100.0, 100.5, 99.5, 100.0, v=200.0, i=99),
                   near_level=100.0, delta=-75.0)
    assert r.delta == pytest.approx(-75.0)
