"""Auto gain and auto exposure controllers for spectrum peak leveling."""

from collections.abc import Callable
import math
from typing import Literal

import numpy as np

from ..core.spectrum import SpectrumData

# Golden ratio (1-φ) for bracket search: one evaluation per step, O(log(range)) convergence.
_GOLDEN = (math.sqrt(5) - 1) / 2  # ~0.618
_INV_GOLDEN = 1.0 - _GOLDEN  # ~0.382

# When bracket collapses, bias toward underexposed/low-gain side to avoid blowing highlights.
_CONVERGE_BIAS_LOW = 0.25

# Require this many consecutive out-of-band frames before starting a new search (avoids restart on one noisy frame).
_SUSTAIN_COUNT = 2


def _peak_vs_target(
    current_max: float,
    max_intensity: float = 1.0,
    target_high_frac: float = 0.95,
    target_low_frac: float = 0.80,
) -> Literal["low", "high", "ok"]:
    """Classify spectrum peak vs target band. Hysteresis: only correct outside [low, high].

    - peak > target_high_frac → oversaturated, correct (reduce)
    - peak < target_low_frac → undersaturated, correct (increase)
    - target_low_frac ≤ peak ≤ target_high_frac → ok, no correction (avoids noise-driven nudges)
    """
    if current_max < max_intensity * 0.02:
        return "low"
    if current_max > max_intensity * target_high_frac:
        return "high"
    if current_max < max_intensity * target_low_frac:
        return "low"
    return "ok"


class AutoGainController:
    """Adjusts camera gain to keep spectrum peak in target range (80-95%).

    Hysteresis: correct only when peak < 0.8 (undersaturated) or > 0.95 (oversaturated);
    in 0.8-0.95 no correction to avoid noise-driven oscillation.
    Uses golden-ratio bracket search for fast convergence.
    """

    def __init__(
        self,
        gain_min: float = 1.0,
        gain_max: float = 16.0,
        gain_step_threshold: float = 0.2,
        verbose: bool = True,
    ):
        self.gain_min = gain_min
        self.gain_max = gain_max
        self.gain_step_threshold = gain_step_threshold
        self.verbose = verbose
        self._search_low: float | None = None
        self._search_high: float | None = None
        self._search_last_gain: float | None = None
        self._consecutive_high: int = 0
        self._consecutive_low: int = 0
        self._skip_next_frame: bool = False  # Skip one frame after changing gain (avoids mid-frame artifacts).

    def adjust(
        self,
        data: SpectrumData,
        get_gain: Callable[[], float],
        set_gain: Callable[[float], None],
        set_display_gain: Callable[[float], None],
    ) -> bool:
        """Adjust gain if spectrum peak is outside target range.

        Args:
            data: Processed spectrum data (intensity 0-1)
            get_gain: Callable that returns current gain
            set_gain: Callable to set camera gain
            set_display_gain: Callable to update display slider

        Returns:
            True if gain was adjusted
        """
        if data is None or len(data.intensity) == 0:
            return False
        if self._skip_next_frame:
            self._skip_next_frame = False
            return False

        current_max = float(np.max(data.intensity))
        current_gain = get_gain()
        kind = _peak_vs_target(current_max, 1.0)

        if self._search_last_gain is not None and self._search_high is not None and self._search_low is not None:
            peak_at_last = current_max
            last_ok = _peak_vs_target(peak_at_last, 1.0)
            if last_ok == "ok":
                self._search_low = self._search_high = self._search_last_gain = None
                self._consecutive_high = self._consecutive_low = 0
                return False
            if last_ok == "low":
                self._search_low = self._search_last_gain
            else:
                self._search_high = self._search_last_gain
            if self._search_high - self._search_low <= self.gain_step_threshold:
                new_gain = self._search_low + (self._search_high - self._search_low) * _CONVERGE_BIAS_LOW
                new_gain = max(self.gain_min, min(self.gain_max, new_gain))
                set_gain(new_gain)
                set_display_gain(new_gain)
                self._search_low = self._search_high = self._search_last_gain = None
                self._consecutive_high = self._consecutive_low = 0
                self._skip_next_frame = True
                if self.verbose:
                    print(f"[AG] Gain: {new_gain:.1f} (converged, peak: {current_max:.3f})")
                return True
            next_gain = self._search_low + (self._search_high - self._search_low) * _INV_GOLDEN
            next_gain = max(self.gain_min, min(self.gain_max, next_gain))
            set_gain(next_gain)
            set_display_gain(next_gain)
            self._search_last_gain = next_gain
            self._skip_next_frame = True
            if self.verbose:
                print(f"[AG] Gain: {next_gain:.1f} (bracket [{self._search_low:.1f},{self._search_high:.1f}], peak: {current_max:.3f})")
            return True

        if kind == "ok":
            self._consecutive_high = self._consecutive_low = 0
            return False

        # Sustain: only start a new search after consecutive out-of-band (avoids restart on one noisy frame).
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

        # Do not start a new search if already at limit (never reduce below min or increase above max).
        if kind == "low" and current_gain >= self.gain_max:
            return False
        if kind == "high" and current_gain <= self.gain_min:
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
        self._skip_next_frame = True
        if self.verbose:
            print(f"[AG] Gain: {next_gain:.1f} (seek start, peak: {current_max:.3f})")
        return True


class AutoExposureController:
    """Adjusts camera exposure to keep spectrum peak in target range (80-95%).

    Auto exposure is limited to exposure_max_us (default 1 s per frame). Manual
    control via the capturer can set higher values (e.g. up to 10 s); only auto
    is capped here.
    Hysteresis: correct only when peak < 0.8 or > 0.95; 0.8-0.95 no correction.
    Uses golden-ratio bracket search for fast convergence.
    """

    def __init__(
        self,
        exposure_min_us: int = 100,
        exposure_max_us: int = 1_000_000,  # 1 s max for auto (manual can use capturer up to driver limit)
        exposure_preferred_max_us: int = 500_000,  # Prefer exposure ≤ this to allow lower gain (less noise)
        exposure_step_min: int = 50,
        verbose: bool = True,
    ):
        self.exposure_min_us = exposure_min_us
        self.exposure_max_us = exposure_max_us
        self.exposure_preferred_max_us = exposure_preferred_max_us
        self.exposure_step_min = exposure_step_min
        self.verbose = verbose
        self._search_low_us: int | None = None
        self._search_high_us: int | None = None
        self._search_last_exposure_us: int | None = None
        self._consecutive_high: int = 0
        self._consecutive_low: int = 0
        self._skip_next_frame: bool = False  # Skip one frame after changing exposure (avoids mid-frame artifacts).

    def adjust(
        self,
        data: SpectrumData,
        get_exposure: Callable[[], int],
        set_exposure: Callable[[int], None],
        set_display_exposure: Callable[[int], None],
    ) -> bool:
        """Adjust exposure if spectrum peak is outside target range.

        Args:
            data: Processed spectrum data (intensity 0-1)
            get_exposure: Callable that returns current exposure (microseconds)
            set_exposure: Callable to set camera exposure
            set_display_exposure: Callable to update display slider

        Returns:
            True if exposure was adjusted
        """
        if data is None or len(data.intensity) == 0:
            return False
        if self._skip_next_frame:
            self._skip_next_frame = False
            return False

        current_max = float(np.max(data.intensity))
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
                self._consecutive_high = self._consecutive_low = 0
                return False
            if last_ok == "low":
                self._search_low_us = self._search_last_exposure_us
            else:
                self._search_high_us = self._search_last_exposure_us
            if self._search_high_us - self._search_low_us <= self.exposure_step_min:
                span = self._search_high_us - self._search_low_us
                new_exposure = self._search_low_us + int(span * _CONVERGE_BIAS_LOW)
                new_exposure = max(
                    self.exposure_min_us,
                    min(self.exposure_max_us, new_exposure),
                )
                set_exposure(new_exposure)
                set_display_exposure(new_exposure)
                self._search_low_us = self._search_high_us = None
                self._search_last_exposure_us = None
                self._consecutive_high = self._consecutive_low = 0
                self._skip_next_frame = True
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
            self._skip_next_frame = True
            if self.verbose:
                print(
                    f"[AE] Exposure: {next_exposure} us (bracket "
                    f"[{self._search_low_us},{self._search_high_us}], peak: {current_max:.3f})"
                )
            return True

        if kind == "ok":
            self._consecutive_high = self._consecutive_low = 0
            return False

        # Sustain: only start a new search after consecutive out-of-band (avoids restart on one noisy frame).
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

        # Do not start a new search if already at limit (never reduce below min or increase above max).
        if kind == "low" and current_exposure >= self.exposure_max_us:
            return False
        if kind == "high" and current_exposure <= self.exposure_min_us:
            return False

        if kind == "low":
            self._search_low_us = current_exposure
            # Prefer exposure ≤ preferred_max first (lower gain, less noise); then allow up to full max.
            high_cap = self.exposure_preferred_max_us if current_exposure < self.exposure_preferred_max_us else self.exposure_max_us
            self._search_high_us = min(self.exposure_max_us, high_cap)
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
        self._skip_next_frame = True
        if self.verbose:
            print(f"[AE] Exposure: {next_exposure} us (seek start, peak: {current_max:.3f})")
        return True
