"""Auto gain and auto exposure: keep spectrum peak in 0.80–0.95, never overexpose.

Uses golden-ratio bracket search for fast convergence. Peak is smoothed with
exponential averaging at ~10 Hz to filter mains and avoid jumping/pulsing.
"""

from collections.abc import Callable
import math
from typing import Literal

import numpy as np

from ..core.spectrum import SpectrumData

# Target band: peak in [0.80, 0.95]. Never exceed 0.95; aim for 0.8–0.9.
TARGET_HIGH_FRAC = 0.95
TARGET_LOW_FRAC = 0.80

# Smoothing: ~10 Hz so alpha = frame_period / tau, tau = 0.1 s
PEAK_SMOOTHING_TAU_SEC = 0.1

# Golden ratio for bracket search: one evaluation per step, O(log(range)) convergence.
_GOLDEN = (math.sqrt(5) - 1) / 2  # ~0.618
_INV_GOLDEN = 1.0 - _GOLDEN  # ~0.382

_DEFAULT_FRAME_PERIOD_SEC = 0.033


def _peak_vs_target(
    current_max: float,
    max_intensity: float = 1.0,
    target_high_frac: float = TARGET_HIGH_FRAC,
    target_low_frac: float = TARGET_LOW_FRAC,
) -> Literal["low", "high", "ok"]:
    """Classify spectrum peak vs target band. Used by bracket search."""
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
    tau_sec: float,
    smoothed: float | None,
) -> float:
    """Exponential moving average: x = alpha*raw + (1-alpha)*x, alpha = dt/tau (~10 Hz)."""
    alpha = min(1.0, frame_period_sec / tau_sec)
    if smoothed is None:
        return raw_peak
    return alpha * raw_peak + (1.0 - alpha) * smoothed


class AutoGainController:
    """Adjusts camera gain to keep spectrum peak in target range (80–95%).

    Uses golden-ratio bracket search. Peak is smoothed at ~10 Hz to filter mains.
    """

    def __init__(
        self,
        gain_min: float = 1.0,
        gain_max: float = 16.0,
        gain_step_threshold: float = 0.2,
        peak_smoothing_period_sec: float = PEAK_SMOOTHING_TAU_SEC,
        max_adjust_rate_hz: float = 20.0,
        verbose: bool = True,
    ):
        self.gain_min = gain_min
        self.gain_max = gain_max
        self.gain_step_threshold = gain_step_threshold
        self.peak_smoothing_period_sec = peak_smoothing_period_sec
        self._max_adjust_rate_hz = max_adjust_rate_hz
        self.verbose = verbose
        self._search_low: float | None = None
        self._search_high: float | None = None
        self._search_last_gain: float | None = None
        self._smoothed_peak: float | None = None

    def adjust(
        self,
        data: SpectrumData,
        get_gain: Callable[[], float],
        set_gain: Callable[[float], None],
        set_display_gain: Callable[[float], None],
    ) -> bool:
        if data is None or len(data.intensity) == 0:
            return False

        raw_peak = float(np.max(data.intensity))
        frame_period = (
            data.exposure_us / 1e6 if data.exposure_us is not None else _DEFAULT_FRAME_PERIOD_SEC
        )
        self._smoothed_peak = _smooth_peak(
            raw_peak, frame_period, self.peak_smoothing_period_sec, self._smoothed_peak
        )
        current_max = self._smoothed_peak
        current_gain = get_gain()
        kind = _peak_vs_target(current_max, 1.0)

        if (
            self._search_last_gain is not None
            and self._search_high is not None
            and self._search_low is not None
        ):
            last_ok = _peak_vs_target(current_max, 1.0)
            if last_ok == "ok":
                self._search_low = self._search_high = self._search_last_gain = None
                return False
            if last_ok == "low":
                self._search_low = self._search_last_gain
            else:
                self._search_high = self._search_last_gain
            if self._search_high - self._search_low <= self.gain_step_threshold:
                new_gain = (self._search_low + self._search_high) / 2.0
                new_gain = max(self.gain_min, min(self.gain_max, new_gain))
                set_gain(new_gain)
                set_display_gain(new_gain)
                self._search_low = self._search_high = self._search_last_gain = None
                if self.verbose:
                    print(f"[AG] Gain: {new_gain:.1f} (converged, peak: {current_max:.3f})")
                return True
            next_gain = self._search_low + (self._search_high - self._search_low) * _INV_GOLDEN
            next_gain = max(self.gain_min, min(self.gain_max, next_gain))
            set_gain(next_gain)
            set_display_gain(next_gain)
            self._search_last_gain = next_gain
            if self.verbose:
                print(
                    f"[AG] Gain: {next_gain:.1f} (bracket [{self._search_low:.1f},{self._search_high:.1f}], peak: {current_max:.3f})"
                )
            return True

        if kind == "ok":
            return False

        if kind == "low":
            self._search_low = current_gain
            self._search_high = self.gain_max
        else:
            self._search_low = self.gain_min
            self._search_high = current_gain
        next_gain = self._search_low + (self._search_high - self._search_low) * _INV_GOLDEN
        next_gain = max(self.gain_min, min(self.gain_max, next_gain))
        set_gain(next_gain)
        set_display_gain(next_gain)
        self._search_last_gain = next_gain
        if self.verbose:
            print(f"[AG] Gain: {next_gain:.1f} (seek start, peak: {current_max:.3f})")
        return True


class AutoExposureController:
    """Adjusts camera exposure to keep spectrum peak in target range (80–95%).

    Uses golden-ratio bracket search. Never increases exposure when peak > 0.95.
    Peak is smoothed at ~10 Hz to filter mains.
    """

    def __init__(
        self,
        exposure_min_us: int = 100,
        exposure_max_us: int = 1_000_000,
        exposure_preferred_max_us: int | None = 500_000,
        exposure_step_min: int = 50,
        peak_smoothing_period_sec: float = PEAK_SMOOTHING_TAU_SEC,
        max_adjust_rate_hz: float = 20.0,
        verbose: bool = True,
    ):
        self.exposure_min_us = exposure_min_us
        self.exposure_max_us = exposure_max_us
        self.exposure_preferred_max_us = exposure_preferred_max_us
        self.exposure_step_min = exposure_step_min
        self.peak_smoothing_period_sec = peak_smoothing_period_sec
        self._max_adjust_rate_hz = max_adjust_rate_hz
        self.verbose = verbose
        self._search_low_us: int | None = None
        self._search_high_us: int | None = None
        self._search_last_exposure_us: int | None = None
        self._smoothed_peak: float | None = None

    def adjust(
        self,
        data: SpectrumData,
        get_exposure: Callable[[], int],
        set_exposure: Callable[[int], None],
        set_display_exposure: Callable[[int], None],
    ) -> bool:
        if data is None or len(data.intensity) == 0:
            return False

        raw_peak = float(np.max(data.intensity))
        frame_period = (
            data.exposure_us / 1e6 if data.exposure_us is not None else _DEFAULT_FRAME_PERIOD_SEC
        )
        self._smoothed_peak = _smooth_peak(
            raw_peak, frame_period, self.peak_smoothing_period_sec, self._smoothed_peak
        )
        current_max = self._smoothed_peak
        current_exposure = get_exposure()
        kind = _peak_vs_target(current_max, 1.0)

        if (
            self._search_last_exposure_us is not None
            and self._search_high_us is not None
            and self._search_low_us is not None
        ):
            last_ok = _peak_vs_target(current_max, 1.0)
            if last_ok == "ok":
                self._search_low_us = self._search_high_us = None
                self._search_last_exposure_us = None
                return False
            if last_ok == "low":
                self._search_low_us = self._search_last_exposure_us
            else:
                self._search_high_us = self._search_last_exposure_us
            if self._search_high_us - self._search_low_us <= self.exposure_step_min:
                new_exposure = (self._search_low_us + self._search_high_us) // 2
                new_exposure = max(
                    self.exposure_min_us,
                    min(self.exposure_max_us, new_exposure),
                )
                set_exposure(new_exposure)
                set_display_exposure(new_exposure)
                self._search_low_us = self._search_high_us = None
                self._search_last_exposure_us = None
                if self.verbose:
                    print(f"[AE] Exposure: {new_exposure} us (converged, peak: {current_max:.3f})")
                return True
            span = self._search_high_us - self._search_low_us
            next_exposure = self._search_low_us + int(span * _INV_GOLDEN)
            next_exposure = max(
                self.exposure_min_us,
                min(self.exposure_max_us, next_exposure),
            )
            set_exposure(next_exposure)
            set_display_exposure(next_exposure)
            self._search_last_exposure_us = next_exposure
            if self.verbose:
                print(
                    f"[AE] Exposure: {next_exposure} us (bracket "
                    f"[{self._search_low_us},{self._search_high_us}], peak: {current_max:.3f})"
                )
            return True

        if kind == "ok":
            return False

        if kind == "low":
            self._search_low_us = current_exposure
            self._search_high_us = (
                min(self.exposure_max_us, self.exposure_preferred_max_us)
                if self.exposure_preferred_max_us is not None
                else self.exposure_max_us
            )
        else:
            self._search_low_us = self.exposure_min_us
            self._search_high_us = current_exposure
        span = self._search_high_us - self._search_low_us
        next_exposure = self._search_low_us + int(span * _INV_GOLDEN)
        next_exposure = max(
            self.exposure_min_us,
            min(self.exposure_max_us, next_exposure),
        )
        set_exposure(next_exposure)
        set_display_exposure(next_exposure)
        self._search_last_exposure_us = next_exposure
        if self.verbose:
            print(f"[AE] Exposure: {next_exposure} us (seek start, peak: {current_max:.3f})")
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
    """Run one frame: exposure first, then gain with cooldown after exposure change."""
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
