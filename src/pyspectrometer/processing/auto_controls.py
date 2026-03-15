"""Auto gain and auto exposure controllers for spectrum peak leveling.

Uses the standard camera-style multiplicative update: value_new = value_old × (target / measured).
See docs/AUTO_EXPOSURE_ALGORITHM.md.
"""

from collections.abc import Callable
import time
from typing import Literal

import numpy as np

from ..core.spectrum import SpectrumData

# Target peak (center of 0.80–0.95 band). Multiplicative update drives toward this.
TARGET_PEAK = 0.875

# Only treat as overexposed ("high") when peak exceeds this. Slightly above 0.95 to avoid pulsing at the boundary.
TARGET_HIGH_HYSTERESIS = 0.96

# When coming down from overexposed, treat this as "ok" low bound so we don't overshoot and oscillate.
TARGET_LOW_AFTER_HIGH = 0.62

# Damp the multiplicative step so we don't overshoot (ratio^damping). 1.0 = full step, 0.4 = conservative.
STEP_DAMPING = 0.4

# When far from target, use larger steps (less damping) so we don't take forever to reach proper exposure.
STEP_DAMPING_WHEN_FAR = 0.75  # ratio far from 1.0 → bigger steps
_RATIO_FAR_DOWN = 0.92   # ratio below this (overexposed) → use STEP_DAMPING_WHEN_FAR
_RATIO_FAR_UP = 1.15     # ratio above this (underexposed) → use STEP_DAMPING_WHEN_FAR

# Cap upward step so smoothing lag cannot cause a single big overshoot (e.g. "into range then up much higher").
MAX_UP_RATIO = 1.2
# Cap downward step when overexposed (peak clipped at 1.0) so we don't overshoot into underexposed and oscillate.
MAX_DOWN_RATIO = 0.88

# Consecutive out-of-band frames before acting (avoids one noisy frame and reduces oscillation).
_SUSTAIN_COUNT = 3

# Minimum meaningful peak to avoid division by zero.
_MIN_PEAK = 1e-6

# Default frame period when exposure_us is unknown (for smoothing alpha).
_DEFAULT_FRAME_PERIOD_SEC = 0.033


def _peak_vs_target(
    current_max: float,
    max_intensity: float = 1.0,
    target_high_frac: float = 0.95,
    target_low_frac: float = 0.80,
) -> Literal["low", "high", "ok"]:
    """Classify spectrum peak vs target band. Only correct outside [low_frac, high_frac]."""
    if current_max < max_intensity * 0.02:
        return "low"
    if current_max > max_intensity * target_high_frac:
        return "high"
    if current_max < max_intensity * target_low_frac:
        return "low"
    return "ok"


def _smooth_peak(
    raw_peak: float,
    frame_period_sec: float,
    smoothing_period_sec: float,
    smoothed: float | None,
) -> float:
    """Exponential moving average of peak. Shorter frame → more smoothing."""
    alpha = min(1.0, frame_period_sec / smoothing_period_sec)
    if smoothed is None:
        return raw_peak
    return alpha * raw_peak + (1.0 - alpha) * smoothed


def _effective_damping(ratio: float) -> float:
    """Use larger steps when far from target so we reach proper exposure in reasonable time."""
    if ratio < _RATIO_FAR_DOWN or ratio > _RATIO_FAR_UP:
        return STEP_DAMPING_WHEN_FAR
    return STEP_DAMPING


class AutoGainController:
    """Adjusts camera gain so spectrum peak stays in target range (80–95%).

    Uses multiplicative update: gain_new = gain_old × (target_peak / smoothed_peak), clamped.
    """

    def __init__(
        self,
        gain_min: float = 1.0,
        gain_max: float = 16.0,
        peak_smoothing_period_sec: float = 0.04,
        max_adjust_rate_hz: float = 10.0,
        verbose: bool = True,
    ):
        self.gain_min = gain_min
        self.gain_max = gain_max
        self.peak_smoothing_period_sec = peak_smoothing_period_sec
        self.max_adjust_rate_hz = max_adjust_rate_hz
        self.verbose = verbose
        self._consecutive_high: int = 0
        self._consecutive_low: int = 0
        self._skip_next_frame: bool = False
        self._smoothed_peak: float | None = None
        self._last_gain_adjust_time: float = 0.0

    def adjust(
        self,
        data: SpectrumData,
        get_gain: Callable[[], float],
        set_gain: Callable[[float], None],
        set_display_gain: Callable[[float], None],
    ) -> bool:
        if data is None or len(data.intensity) == 0:
            return False
        if self._skip_next_frame:
            self._skip_next_frame = False
            return False

        raw_peak = float(np.max(data.intensity))
        frame_period_sec = (
            data.exposure_us / 1e6 if data.exposure_us is not None else _DEFAULT_FRAME_PERIOD_SEC
        )
        self._smoothed_peak = _smooth_peak(
            raw_peak,
            frame_period_sec,
            self.peak_smoothing_period_sec,
            self._smoothed_peak,
        )
        current_max = self._smoothed_peak
        current_gain = get_gain()
        kind = _peak_vs_target(
            current_max, 1.0, target_high_frac=TARGET_HIGH_HYSTERESIS
        )

        if kind == "ok":
            self._consecutive_high = self._consecutive_low = 0
            return False

        if kind == "low":
            self._consecutive_low += 1
            self._consecutive_high = 0
            if self._consecutive_low < _SUSTAIN_COUNT:
                return False
        else:
            self._consecutive_high += 1
            self._consecutive_low = 0
            if self._consecutive_high < _SUSTAIN_COUNT:
                return False

        if kind == "low" and current_gain >= self.gain_max:
            return False
        if kind == "high" and current_gain <= self.gain_min:
            return False

        # Rate-limit gain updates (e.g. 10 Hz) so we don't fight exposure and overshoot.
        now = time.monotonic()
        min_interval = 1.0 / self.max_adjust_rate_hz if self.max_adjust_rate_hz > 0 else 0.0
        if min_interval > 0 and (now - self._last_gain_adjust_time) < min_interval:
            return False

        ratio = TARGET_PEAK / max(current_max, _MIN_PEAK)
        damping = _effective_damping(ratio)
        effective_ratio = ratio ** damping
        if effective_ratio > 1.0:
            effective_ratio = min(effective_ratio, MAX_UP_RATIO)
        new_gain = current_gain * effective_ratio
        new_gain = max(self.gain_min, min(self.gain_max, new_gain))
        set_gain(new_gain)
        set_display_gain(new_gain)
        self._skip_next_frame = True
        self._last_gain_adjust_time = now
        if self.verbose:
            print(f"[AG] Gain: {new_gain:.1f} (peak: {current_max:.3f})")
        return True


class AutoExposureController:
    """Adjusts camera exposure so spectrum peak stays in target range (80–95%).

    Uses multiplicative update: exposure_new = exposure_old × (target_peak / smoothed_peak), clamped.
    Prefers exposure ≤ exposure_preferred_max_us (e.g. 500 ms) before using higher values.
    """

    def __init__(
        self,
        exposure_min_us: int = 100,
        exposure_max_us: int = 1_000_000,
        exposure_preferred_max_us: int = 500_000,
        peak_smoothing_period_sec: float = 0.04,
        max_adjust_rate_hz: float = 10.0,
        verbose: bool = True,
    ):
        self.exposure_min_us = exposure_min_us
        self.exposure_max_us = exposure_max_us
        self.exposure_preferred_max_us = exposure_preferred_max_us
        self.peak_smoothing_period_sec = peak_smoothing_period_sec
        self.verbose = verbose
        self._max_adjust_rate_hz = max_adjust_rate_hz
        self._consecutive_high: int = 0
        self._consecutive_low: int = 0
        self._skip_next_frame: bool = False
        self._smoothed_peak: float | None = None
        self._last_exposure_adjust_time: float = 0.0
        self._recently_overexposed: bool = False

    def adjust(
        self,
        data: SpectrumData,
        get_exposure: Callable[[], int],
        set_exposure: Callable[[int], None],
        set_display_exposure: Callable[[int], None],
    ) -> bool:
        if data is None or len(data.intensity) == 0:
            return False
        if self._skip_next_frame:
            self._skip_next_frame = False
            return False

        raw_peak = float(np.max(data.intensity))
        frame_period_sec = (
            data.exposure_us / 1e6 if data.exposure_us is not None else _DEFAULT_FRAME_PERIOD_SEC
        )
        self._smoothed_peak = _smooth_peak(
            raw_peak,
            frame_period_sec,
            self.peak_smoothing_period_sec,
            self._smoothed_peak,
        )
        current_max = self._smoothed_peak
        current_exposure = get_exposure()
        # When coming down from overexposed, use wider "ok" band so we don't increase again and overshoot back to 1.0.
        # Only clear the flag when solidly in normal band (peak >= 0.78), not on first "ok" at 0.65.
        low_frac = TARGET_LOW_AFTER_HIGH if self._recently_overexposed else 0.80
        kind = _peak_vs_target(
            current_max, 1.0, target_high_frac=TARGET_HIGH_HYSTERESIS, target_low_frac=low_frac
        )
        if kind == "high":
            self._recently_overexposed = True
        elif current_max >= 0.78:
            self._recently_overexposed = False

        if kind == "ok":
            self._consecutive_high = self._consecutive_low = 0
            return False

        if kind == "low":
            self._consecutive_low += 1
            self._consecutive_high = 0
            if self._consecutive_low < _SUSTAIN_COUNT:
                return False
        else:
            self._consecutive_high += 1
            self._consecutive_low = 0
            if self._consecutive_high < _SUSTAIN_COUNT:
                return False

        if kind == "low" and current_exposure >= self.exposure_max_us:
            return False
        if kind == "high" and current_exposure <= self.exposure_min_us:
            return False

        # Rate-limit exposure updates (e.g. 10 Hz) independent of frame rate.
        now = time.monotonic()
        min_interval = 1.0 / self._max_adjust_rate_hz if self._max_adjust_rate_hz > 0 else 0.0
        if min_interval > 0 and (now - self._last_exposure_adjust_time) < min_interval:
            return False

        ratio = TARGET_PEAK / max(current_max, _MIN_PEAK)
        damping = _effective_damping(ratio)
        effective_ratio = ratio ** damping
        if effective_ratio > 1.0:
            effective_ratio = min(effective_ratio, MAX_UP_RATIO)
        else:
            effective_ratio = max(effective_ratio, MAX_DOWN_RATIO)
        new_exposure = int(current_exposure * effective_ratio)
        # Prefer exposure ≤ preferred_max when underexposed; only allow higher once we're there.
        if kind == "low" and current_exposure < self.exposure_preferred_max_us:
            high_cap = self.exposure_preferred_max_us
        else:
            high_cap = self.exposure_max_us
        new_exposure = max(
            self.exposure_min_us,
            min(high_cap, new_exposure),
        )
        set_exposure(new_exposure)
        set_display_exposure(new_exposure)
        self._skip_next_frame = True
        self._last_exposure_adjust_time = now
        if self.verbose:
            print(f"[AE] Exposure: {new_exposure} us (peak: {current_max:.3f})")
        return True


def run_auto_gain_exposure_frame(
    data: SpectrumData,
    auto_exposure_enabled: bool,
    auto_gain_enabled: bool,
    auto_exposure_ctrl: AutoExposureController,
    auto_gain_ctrl: AutoGainController,
    get_exposure: Callable[[], int],
    set_exposure: Callable[[int], None],
    set_exposure_display: Callable[[int], None],
    get_gain: Callable[[], float],
    set_gain: Callable[[float], None],
    set_gain_display: Callable[[float], None],
    gain_cooldown_remaining: int,
) -> int:
    """Run one frame of AE/AG. Shared by desktop and stream for identical behavior.

    Returns:
        New gain_cooldown_remaining (decremented; set to 3 after exposure adjustment).
    """
    if not auto_exposure_enabled and not auto_gain_enabled:
        return gain_cooldown_remaining

    cooldown = gain_cooldown_remaining
    if cooldown > 0:
        cooldown -= 1

    if auto_exposure_enabled:
        if auto_exposure_ctrl.adjust(
            data,
            get_exposure,
            set_exposure,
            set_exposure_display,
        ):
            return 3
    if auto_gain_enabled and cooldown <= 0:
        auto_gain_ctrl.adjust(
            data,
            get_gain,
            set_gain,
            set_gain_display,
        )
    return cooldown
