"""Buying/selling pressure and absorption detectors (module 11).

These are OHLCV approximations of order-flow concepts. We don't have
per-bar aggressor delta in the default pipeline (the loader produces
pre-aggregated OHLCV-1m bars), so we proxy "pressure" and "absorption"
from candle geometry and relative volume:

  Buying pressure (per bar)
    - close_position_in_range = (close - low) / (high - low) in [0, 1].
      A close in the upper third (>= pressure_close_upper_third) means
      buyers won the bar. Mirror (lower third) for selling pressure.
    - body_fraction = |close - open| / (high - low) in [0, 1]. A larger
      body relative to range = more conviction, less indecision.
    - body_vs_trailing_avg = body_size / SMA(body_size, n). A relative-size
      proxy: is this a bigger-than-usual move?
    A clean buying-pressure signal needs all three: close upper third AND
    body_fraction >= min AND body_vs_trailing_avg >= mult. Mirror for sells.

  Absorption (per bar at a tested level)
    - High volume (>= mult * trailing SMA, same idea as the sweep filter).
    - Small range relative to recent bars (range < mult * trailing avg range).
    - The bar overlaps the level being tested.
    All three together = the level absorbed the flow (lots of volume traded,
    price barely moved) — evidence the level is holding.

Both detectors accept an optional `delta` (signed aggressor volume) for a
future tick-based upgrade. When supplied, it is carried through on the
reading and, for pressure, used as a sign gate (buying needs delta > 0,
selling needs delta < 0). When omitted, the detectors run on OHLCV proxies
alone — the default.

Pure and incremental, like the rest of the codebase: `on_bar(bar)` per bar.
Trailing averages exclude the current bar (computed before it is appended),
matching the sweep detector's volume-SMA semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from strategy.bars import Bar
from strategy.params import StrategyParams


@dataclass(frozen=True)
class PressureReading:
    close_position_in_range: float          # [0, 1]; 0.5 for a zero-range bar
    body_fraction: float                    # [0, 1]; 0.0 for a zero-range bar
    body_vs_trailing_avg: Optional[float]   # None until enough history
    buying_pressure: bool
    selling_pressure: bool
    delta: Optional[float] = None           # carried through if supplied


@dataclass(frozen=True)
class AbsorptionReading:
    volume_ratio: Optional[float]           # None until enough history
    range_ratio: Optional[float]            # range / trailing avg range; None until history
    overlaps_level: bool
    absorption: bool
    delta: Optional[float] = None           # carried through if supplied


def _close_position_in_range(bar: Bar) -> float:
    rng = bar.high - bar.low
    if rng <= 0:
        return 0.5  # zero-range bar is directionally neutral
    return (bar.close - bar.low) / rng


def _body_fraction(bar: Bar) -> float:
    rng = bar.high - bar.low
    if rng <= 0:
        return 0.0
    return abs(bar.close - bar.open) / rng


class PressureDetector:
    """Per-bar buying/selling pressure from candle geometry + relative body size."""

    def __init__(self, params: StrategyParams) -> None:
        self.params = params
        self._body_history: list[float] = []  # current bar excluded until appended

    def on_bar(self, bar: Bar, delta: Optional[float] = None) -> PressureReading:
        p = self.params
        close_pos = _close_position_in_range(bar)
        body_frac = _body_fraction(bar)
        body_size = abs(bar.close - bar.open)

        body_vs_avg = self._body_vs_trailing_avg(body_size)

        big_enough = body_vs_avg is not None and body_vs_avg >= p.pressure_body_vs_avg_mult
        has_body = body_frac >= p.pressure_min_body_fraction

        buying = (
            close_pos >= p.pressure_close_upper_third
            and has_body
            and big_enough
        )
        selling = (
            close_pos <= p.pressure_close_lower_third
            and has_body
            and big_enough
        )

        # Optional aggressor-delta sign gate (tick-based upgrade hook).
        if delta is not None:
            buying = buying and delta > 0
            selling = selling and delta < 0

        # Append AFTER computing the trailing average (exclude current bar).
        self._body_history.append(body_size)
        if len(self._body_history) > p.pressure_body_window + 1:
            self._body_history.pop(0)

        return PressureReading(
            close_position_in_range=close_pos,
            body_fraction=body_frac,
            body_vs_trailing_avg=body_vs_avg,
            buying_pressure=buying,
            selling_pressure=selling,
            delta=delta,
        )

    def _body_vs_trailing_avg(self, body_size: float) -> Optional[float]:
        window = self.params.pressure_body_window
        history = self._body_history  # current bar not yet appended
        if len(history) < window:
            return None
        avg = sum(history[-window:]) / window
        if avg <= 0:
            return None
        return body_size / avg


class AbsorptionDetector:
    """Per-bar absorption: high volume + small range + overlaps a tested level."""

    def __init__(self, params: StrategyParams) -> None:
        self.params = params
        self._volume_history: list[float] = []
        self._range_history: list[float] = []

    def on_bar(
        self,
        bar: Bar,
        near_level: Optional[float] = None,
        delta: Optional[float] = None,
    ) -> AbsorptionReading:
        p = self.params
        rng = bar.high - bar.low

        volume_ratio = self._volume_ratio(bar.volume)
        range_ratio = self._range_ratio(rng)

        overlaps = near_level is not None and bar.low <= near_level <= bar.high

        absorption = (
            volume_ratio is not None
            and volume_ratio >= p.absorption_volume_mult
            and range_ratio is not None
            and range_ratio < p.absorption_range_mult
            and overlaps
        )

        # Append AFTER computing trailing averages (exclude current bar).
        self._volume_history.append(bar.volume)
        if len(self._volume_history) > p.absorption_volume_window + 1:
            self._volume_history.pop(0)
        self._range_history.append(rng)
        if len(self._range_history) > p.absorption_range_window + 1:
            self._range_history.pop(0)

        return AbsorptionReading(
            volume_ratio=volume_ratio,
            range_ratio=range_ratio,
            overlaps_level=overlaps,
            absorption=absorption,
            delta=delta,
        )

    def _volume_ratio(self, volume: float) -> Optional[float]:
        window = self.params.absorption_volume_window
        history = self._volume_history
        if len(history) < window:
            return None
        avg = sum(history[-window:]) / window
        if avg <= 0:
            return None
        return volume / avg

    def _range_ratio(self, rng: float) -> Optional[float]:
        window = self.params.absorption_range_window
        history = self._range_history
        if len(history) < window:
            return None
        avg = sum(history[-window:]) / window
        if avg <= 0:
            return None
        return rng / avg
