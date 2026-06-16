"""Strategy parameters — single source of truth.

NQ futures contract specs and all tunable strategy thresholds live here.
Changing a number here changes it everywhere downstream.
"""
from dataclasses import dataclass, field


NQ_TICK_SIZE: float = 0.25
NQ_POINT_VALUE: float = 20.0


@dataclass(frozen=True)
class StrategyParams:
    # Reference levels
    eqhl_swing_lookback: int = 5
    eqhl_tolerance_ticks: int = 2
    eqhl_max_age_bars: int = 7 * 24 * 60  # 1 week of 1m bars
    # Round numbers are tiered: majors are stronger (preferred as targets);
    # minors are finer-grained (more candidate setup levels). A level on
    # both grids is classified as major (the higher tier wins).
    round_number_major_step_pts: float = 100.0
    round_number_minor_step_pts: float = 25.0

    # Sweep detection
    penetration_min_ticks: int = 3
    penetration_max_valid_ticks: int = 8
    penetration_broken_ticks: int = 10
    max_rejection_bars: int = 3
    volume_window: int = 20
    volume_mult: float = 1.5

    # LTF structure
    ltf_swing_lookback: int = 3
    max_choch_bars: int = 15

    # Entry / exit
    stop_buffer_ticks: int = 2
    # Skip trades whose target/stop ratio is below this. The strategy asks
    # for "1:3 R:R" so 3.0 is the design target.
    min_rr_ratio: float = 3.0
    # Abandon a setup if no entry trigger fires within this many bars after
    # CHoCH (1m bars; default 30 = ~30 min).
    max_entry_bars: int = 30

    # Pressure / absorption (module 11) — OHLCV approximations.
    # A "buying pressure" bar closes in its upper third, has a body that fills
    # at least half its range, and is at least as large as the trailing body
    # average. Mirror (lower third) for selling pressure.
    pressure_close_upper_third: float = 0.67  # close-position threshold (buy)
    pressure_close_lower_third: float = 0.33  # close-position threshold (sell)
    pressure_min_body_fraction: float = 0.5
    pressure_body_window: int = 20            # trailing SMA window for body size
    pressure_body_vs_avg_mult: float = 1.0
    # Absorption: high volume + unusually small range, sitting on a tested
    # level. Volume threshold mirrors the sweep volume confirmation.
    absorption_volume_window: int = 20
    absorption_volume_mult: float = 1.5
    absorption_range_window: int = 20
    absorption_range_mult: float = 0.7        # range < mult * trailing avg range

    # Continuation setup (module 12). Trend-following: in an HTF uptrend,
    # enter longs on a retest of session VWAP (the intraday "midpoint"),
    # confirmed by buying pressure / absorption / an FVG sitting at VWAP.
    # Mirror for shorts. Each confluence requirement is an independent gate
    # so it can be relaxed against real-data evidence without code changes.
    cont_htf_period: str = "4h"            # HTF trend filter timeframe
    cont_require_pressure: bool = True     # buying/selling pressure on retest bar
    cont_require_absorption: bool = True   # absorption at VWAP on retest bar
    cont_require_fvg_at_vwap: bool = True  # an active FVG straddling VWAP

    # Execution model (v1)
    slippage_ticks: int = 1
    commission_per_contract: float = 0.0  # in points; convert later
    position_size: int = 1

    def ticks_to_pts(self, ticks: int) -> float:
        return ticks * NQ_TICK_SIZE


DEFAULT_PARAMS = StrategyParams()
