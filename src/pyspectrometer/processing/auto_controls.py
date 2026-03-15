"""Auto gain and auto exposure: never overexpose; expose correctly.

Simple rules:
- Peak > 0.95: overexposed → reduce exposure (never increase). Reduce gain if needed.
- Peak < 0.90: underexposed → increase exposure, try to decrease gain (prefer exposure over gain).
- 0.90 <= peak <= 0.95: in band → no change.
"""

from collections.abc import Callable
import time

import numpy as np

from ..core.spectrum import SpectrumData

# Thresholds: never overexpose; correct when underexposed.
PEAK_OVER = 0.95   # Above this: only reduce exposure, never increase. Reduce gain.
PEAK_UNDER = 0.90  # Below this: increase exposure, try to decrease gain.

# Multiplicative step: aim for this when adjusting (so we land in band, not overshoot).
TARGET_PEAK = 0.92

# Step caps (per update)
MAX_EXPOSURE_UP_RATIO = 1.2
MAX_EXPOSURE_DOWN_RATIO = 0.88
MAX_GAIN_UP_RATIO = 1.2
MAX_GAIN_DOWN_RATIO = 0.85

# Consecutive out-of-band frames before acting (avoids reacting to one noisy frame).
_SUSTAIN = 2

_MIN_PEAK = 1e-6
_DEFAULT_FRAME_PERIOD_SEC = 0.033


def _smooth_peak(
    raw_peak: float,
    frame_period_sec: float,
    smoothing_period_sec: float,
    smoothed: float | None,
) -> float:
    """Exponential moving average of peak."""
    alpha = min(1.0, frame_period_sec / smoothing_period_sec)
    if smoothed is None:
        return raw_peak
    return alpha * raw_peak + (1.0 - alpha) * smoothed


class AutoGainController:
    """Adjust gain: reduce when overexposed; when underexposed try to decrease gain (exposure takes priority)."""

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
        self._consecutive_high = 0
        self._consecutive_low = 0
        self._skip_next_frame = False
        self._smoothed_peak: float | None = None
        self._last_adjust_time: float = 0.0

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
        frame_period = (
            data.exposure_us / 1e6 if data.exposure_us is not None else _DEFAULT_FRAME_PERIOD_SEC
        )
        self._smoothed_peak = _smooth_peak(
            raw_peak, frame_period, self.peak_smoothing_period_sec, self._smoothed_peak
        )
        peak = self._smoothed_peak
        current_gain = get_gain()

        if peak > PEAK_OVER:
            self._consecutive_low = 0
            self._consecutive_high += 1
            if self._consecutive_high < _SUSTAIN:
                return False
            if current_gain <= self.gain_min:
                return False
            now = time.monotonic()
            if self.max_adjust_rate_hz > 0 and (now - self._last_adjust_time) < 1.0 / self.max_adjust_rate_hz:
                return False
            ratio = TARGET_PEAK / max(peak, _MIN_PEAK)
            ratio = max(ratio, MAX_GAIN_DOWN_RATIO)
            new_gain = max(self.gain_min, min(self.gain_max, current_gain * ratio))
            set_gain(new_gain)
            set_display_gain(new_gain)
            self._skip_next_frame = True
            self._last_adjust_time = now
            if self.verbose:
                print(f"[AG] Gain: {new_gain:.1f} (peak: {peak:.3f})")
            return True

        if peak < PEAK_UNDER:
            self._consecutive_high = 0
            self._consecutive_low += 1
            if self._consecutive_low < _SUSTAIN:
                return False
            # Try to decrease gain (prefer exposure). Only increase gain if at max exposure and still underexposed.
            if current_gain > self.gain_min:
                now = time.monotonic()
                if self.max_adjust_rate_hz > 0 and (now - self._last_adjust_time) < 1.0 / self.max_adjust_rate_hz:
                    return False
                ratio = TARGET_PEAK / max(peak, _MIN_PEAK)
                ratio = min(ratio, MAX_GAIN_DOWN_RATIO)
                new_gain = max(self.gain_min, current_gain * ratio)
                set_gain(new_gain)
                set_display_gain(new_gain)
                self._skip_next_frame = True
                self._last_adjust_time = now
                if self.verbose:
                    print(f"[AG] Gain: {new_gain:.1f} (peak: {peak:.3f}, prefer exposure)")
                return True
            if current_gain >= self.gain_max:
                return False
            now = time.monotonic()
            if self.max_adjust_rate_hz > 0 and (now - self._last_adjust_time) < 1.0 / self.max_adjust_rate_hz:
                return False
            ratio = TARGET_PEAK / max(peak, _MIN_PEAK)
            ratio = min(ratio, MAX_GAIN_UP_RATIO)
            new_gain = min(self.gain_max, current_gain * ratio)
            set_gain(new_gain)
            set_display_gain(new_gain)
            self._skip_next_frame = True
            self._last_adjust_time = now
            if self.verbose:
                print(f"[AG] Gain: {new_gain:.1f} (peak: {peak:.3f})")
            return True

        self._consecutive_high = self._consecutive_low = 0
        return False


class AutoExposureController:
    """Adjust exposure: never increase when peak > 0.95; when peak < 0.90 increase exposure."""

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
        self._consecutive_high = 0
        self._consecutive_low = 0
        self._skip_next_frame = False
        self._smoothed_peak: float | None = None
        self._last_adjust_time: float = 0.0

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
        frame_period = (
            data.exposure_us / 1e6 if data.exposure_us is not None else _DEFAULT_FRAME_PERIOD_SEC
        )
        self._smoothed_peak = _smooth_peak(
            raw_peak, frame_period, self.peak_smoothing_period_sec, self._smoothed_peak
        )
        peak = self._smoothed_peak
        current_exposure = get_exposure()

        # Overexposed: only reduce exposure. NEVER increase when peak > 0.95.
        if peak > PEAK_OVER:
            self._consecutive_low = 0
            self._consecutive_high += 1
            if self._consecutive_high < _SUSTAIN:
                return False
            if current_exposure <= self.exposure_min_us:
                return False
            now = time.monotonic()
            if self._max_adjust_rate_hz > 0 and (now - self._last_adjust_time) < 1.0 / self._max_adjust_rate_hz:
                return False
            ratio = TARGET_PEAK / max(peak, _MIN_PEAK)
            ratio = max(ratio, MAX_EXPOSURE_DOWN_RATIO)
            new_exposure = int(current_exposure * ratio)
            new_exposure = max(self.exposure_min_us, new_exposure)
            set_exposure(new_exposure)
            set_display_exposure(new_exposure)
            self._skip_next_frame = True
            self._last_adjust_time = now
            if self.verbose:
                print(f"[AE] Exposure: {new_exposure} us (peak: {peak:.3f})")
            return True

        # Underexposed: increase exposure (prefer exposure over gain).
        if peak < PEAK_UNDER:
            self._consecutive_high = 0
            self._consecutive_low += 1
            if self._consecutive_low < _SUSTAIN:
                return False
            if current_exposure >= self.exposure_max_us:
                return False
            now = time.monotonic()
            if self._max_adjust_rate_hz > 0 and (now - self._last_adjust_time) < 1.0 / self._max_adjust_rate_hz:
                return False
            ratio = TARGET_PEAK / max(peak, _MIN_PEAK)
            ratio = min(ratio, MAX_EXPOSURE_UP_RATIO)
            new_exposure = int(current_exposure * ratio)
            high_cap = (
                self.exposure_preferred_max_us
                if current_exposure < self.exposure_preferred_max_us
                else self.exposure_max_us
            )
            new_exposure = max(self.exposure_min_us, min(high_cap, new_exposure))
            set_exposure(new_exposure)
            set_display_exposure(new_exposure)
            self._skip_next_frame = True
            self._last_adjust_time = now
            if self.verbose:
                print(f"[AE] Exposure: {new_exposure} us (peak: {peak:.3f})")
            return True

        self._consecutive_high = self._consecutive_low = 0
        return False


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
    """Run one frame: exposure first (never increase if peak > 0.95); then gain with cooldown after exposure change."""
    if not auto_exposure_enabled and not auto_gain_enabled:
        return gain_cooldown_remaining

    cooldown = max(0, gain_cooldown_remaining - 1)

    if auto_exposure_enabled:
        if auto_exposure_ctrl.adjust(
            data, get_exposure, set_exposure, set_exposure_display
        ):
            return 3
    if auto_gain_enabled and cooldown <= 0:
        auto_gain_ctrl.adjust(data, get_gain, set_gain, set_gain_display)
    return cooldown
