"""Auto gain and auto exposure controllers for spectrum peak leveling."""

from collections.abc import Callable

import numpy as np

from ..core.spectrum import SpectrumData


def _compute_adjustment_ratio(
    current_max: float,
    max_intensity: float,
    threshold_high_frac: float = 0.95,
    threshold_low_frac: float = 0.50,
    ratio_min: float = 0.7,
    ratio_max: float = 1.5,
) -> float | None:
    """Compute gain/exposure adjustment ratio from spectrum peak level.

    Returns None if peak is in acceptable range (no adjustment needed).

    Args:
        current_max: Current maximum intensity (0-1)
        max_intensity: Maximum possible intensity (typically 1.0)
        threshold_high_frac: Above this = saturating, reduce
        threshold_low_frac: Below this = too dark, increase
        ratio_min: Minimum adjustment ratio per step
        ratio_max: Maximum adjustment ratio per step

    Returns:
        Adjustment ratio to apply, or None if no change needed
    """
    threshold_high = max_intensity * threshold_high_frac
    threshold_target = max_intensity * threshold_high_frac
    threshold_low = max_intensity * threshold_low_frac

    if current_max < max_intensity * 0.02:
        ratio = 1.5  # No signal - increase significantly
    elif current_max > threshold_high:
        ratio = threshold_target / current_max
    elif current_max < threshold_low:
        ratio = threshold_target / current_max
    else:
        return None

    return max(ratio_min, min(ratio_max, ratio))


class AutoGainController:
    """Adjusts camera gain to keep spectrum peak in target range (50-95%)."""

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

        current_max = float(np.max(data.intensity))
        max_intensity = 1.0
        ratio = _compute_adjustment_ratio(current_max, max_intensity)
        if ratio is None:
            return False

        current_gain = get_gain()
        new_gain = current_gain * ratio
        new_gain = max(self.gain_min, min(self.gain_max, new_gain))

        if abs(new_gain - current_gain) <= self.gain_step_threshold:
            return False

        set_gain(new_gain)
        set_display_gain(new_gain)
        if self.verbose:
            print(f"[AG] Gain: {new_gain:.1f} (peak: {current_max:.3f})")
        return True


class AutoExposureController:
    """Adjusts camera exposure to keep spectrum peak in target range (50-95%)."""

    def __init__(
        self,
        exposure_min_us: int = 100,
        exposure_max_us: int = 1_000_000,
        exposure_step_min: int = 50,
        verbose: bool = True,
    ):
        self.exposure_min_us = exposure_min_us
        self.exposure_max_us = exposure_max_us
        self.exposure_step_min = exposure_step_min
        self.verbose = verbose

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

        current_max = float(np.max(data.intensity))
        max_intensity = 1.0
        ratio = _compute_adjustment_ratio(current_max, max_intensity)
        if ratio is None:
            return False

        current_exposure = get_exposure()
        new_exposure = int(current_exposure * ratio)
        new_exposure = max(
            self.exposure_min_us,
            min(self.exposure_max_us, new_exposure),
        )

        if abs(new_exposure - current_exposure) <= self.exposure_step_min:
            return False

        set_exposure(new_exposure)
        set_display_exposure(new_exposure)
        if self.verbose:
            print(f"[AE] Exposure: {new_exposure} us (peak: {current_max:.3f})")
        return True
