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

    # Execution model (v1)
    slippage_ticks: int = 1
    commission_per_contract: float = 0.0  # in points; convert later
    position_size: int = 1

    def ticks_to_pts(self, ticks: int) -> float:
        return ticks * NQ_TICK_SIZE


DEFAULT_PARAMS = StrategyParams()
