"""Auto-exposure and auto-gain: keep spectrum peak in [target_low, target_high].

Peak is proportional to E×G (exposure × gain). When underexposed we predict the E (or G)
that would hit target_mid and step there (with a safety factor); we prefer increasing E
over G to keep gain small (better noise). When overexposed we step conservatively.
Skip 2 frames after each change for camera/filter lag.
"""

from collections.abc import Callable

import numpy as np

from ..core.spectrum import SpectrumData
from .sensor_units import counts_to_flux_proxy, peak_to_counts

# Default target band: configurable per-instance.
TARGET_LOW = 0.80
TARGET_HIGH = 0.90

# Underexposed: predicted ratio = target_mid/peak (linear E×G model). Apply safety and cap.
_PREDICT_SAFETY = 0.96   # aim slightly under target; must land in [target_low, target_high]
# Cap up-step so we don't overshoot with Picamera2 pipeline lag (next frame often still old exposure).
_MAX_UP_RATIO = 1.18    # max 18% increase per step when raising E or G (gentler, stays in band)
_MAX_DOWN = 0.5
_SATURATED_RATIO = 0.8
_SATURATED = 0.99
# Frames to skip after changing E or G so the camera returns a frame with the new setting.
_SKIP_FRAMES_AFTER_CHANGE = 4

_DEFAULT_DT = 1.0 / 30.0
_EXPOSURE_EMA_THRESHOLD_US = 33_333  # 1/30 s; only apply EMA when exposure is shorter
_MIN_PEAK = 1e-6


def _ema(raw: float, prev: float | None, dt: float, tau: float) -> float:
    """Exponential moving average. alpha = dt/tau, clamped to [0, 1]."""
    alpha = min(1.0, dt / tau)
    return raw if prev is None else alpha * raw + (1.0 - alpha) * prev


def _frame_dt(exposure_us: int | None) -> float:
    """Inter-frame time: at least 1/30 s; longer exposures extend the frame period."""
    if not exposure_us:
        return _DEFAULT_DT
    return max(_DEFAULT_DT, exposure_us / 1e6)


def _filtered_peak(raw: float, exposure_us: int | None, prev: float | None, tau: float) -> float:
    """Apply EMA only when exposure < 1/30 s; otherwise use raw (long exposure already averages)."""
    if exposure_us is not None and exposure_us >= _EXPOSURE_EMA_THRESHOLD_US:
        return raw
    return _ema(raw, prev, _frame_dt(exposure_us), tau)


class AutoExposureController:
    """Proportional AE: scales exposure so peak converges to target_mid.

    Returns True from adjust() when busy (changed exposure or skipping after change),
    so the coordinator knows to block AG that frame.
    Returns False when in-band or at exposure limit (AG may act).
    """

    def __init__(
        self,
        exposure_min_us: int = 100,
        exposure_max_us: int = 1_000_000,
        target_low: float = TARGET_LOW,
        target_high: float = TARGET_HIGH,
        smoothing_tau: float = 0.05,
        verbose: bool = True,
        bit_depth: int = 10,
        # Legacy params: accepted for caller compatibility, mapped or ignored.
        exposure_preferred_max_us: int | None = None,
        peak_smoothing_period_sec: float = 0.05,
        max_adjust_rate_hz: float = 20.0,
        exposure_step_min: int = 50,
    ):
        self.exposure_min_us = exposure_min_us
        self.exposure_max_us = exposure_max_us
        self.target_low = target_low
        self.target_high = target_high
        self.target_mid = (target_low + target_high) / 2.0
        self.tau = smoothing_tau or peak_smoothing_period_sec
        self.verbose = verbose
        self.bit_depth = bit_depth
        self._ema: float | None = None
        self._skip_remaining: int = 0  # frames to skip after a change (camera/filter lag)
        self.at_max: bool = False

    def _peak_counts_suffix(self, peak: float, data: SpectrumData, steady: bool = False) -> str:
        counts = peak_to_counts(peak, self.bit_depth)
        # flux_proxy only valid when frame was shot with this E×G; omit after we just changed (pipeline lag).
        if steady:
            flux = counts_to_flux_proxy(counts, data.exposure_us, data.gain)
            if flux is not None:
                return f" counts {counts:.0f} flux_proxy {flux:.1f}"
        return f" counts {counts:.0f}"

    @property
    def smoothed_peak(self) -> float | None:
        return self._ema

    def adjust(
        self,
        data: SpectrumData,
        get_exposure: Callable[[], int],
        set_exposure: Callable[[int], None],
        set_display: Callable[[int], None],
    ) -> bool:
        raw = float(np.max(data.intensity))
        self._ema = _filtered_peak(raw, data.exposure_us, self._ema, self.tau)

        if self._skip_remaining > 0:
            self._skip_remaining -= 1
            return True

        peak = self._ema
        current = get_exposure()

        if peak > self.target_high:
            self.at_max = False
            if current <= self.exposure_min_us:
                return False
            ratio = _SATURATED_RATIO if peak >= _SATURATED else max(_MAX_DOWN, self.target_mid / peak)
            new = max(self.exposure_min_us, int(current * ratio))
            if new >= current:
                return False
            set_exposure(new)
            set_display(new)
            self._skip_remaining = _SKIP_FRAMES_AFTER_CHANGE
            if self.verbose:
                print(f"[AE] {new} us (peak {peak:.3f}↓{self._peak_counts_suffix(peak, data, steady=False)})")
            return True

        if peak < self.target_low:
            if self.at_max:
                return False
            if current >= self.exposure_max_us:
                self.at_max = True
                return False
            # Predict: peak ∝ E×G, so E_new = E_current * (target_mid/peak) hits target. Prefer E over G.
            ratio = (self.target_mid / max(peak, _MIN_PEAK)) * _PREDICT_SAFETY
            ratio = min(ratio, _MAX_UP_RATIO)
            new = min(self.exposure_max_us, int(current * ratio))
            if new <= current:
                self.at_max = True
                return False
            set_exposure(new)
            set_display(new)
            self._skip_remaining = _SKIP_FRAMES_AFTER_CHANGE
            if self.verbose:
                print(f"[AE] {new} us (peak {peak:.3f}↑ pred{self._peak_counts_suffix(peak, data, steady=False)})")
            return True

        self.at_max = False
        return False


# When exposure is below this, AG will try to reduce gain (and let AE increase exposure) for better SNR.
PREFER_EXPOSURE_BELOW_US = 500_000
_PREFER_GAIN_DOWN_RATIO = 0.95


class AutoGainController:
    """Proportional AG: scales gain so peak converges to target_mid.

    Secondary actuator: only runs when AE is in-band or at its exposure limit.
    When in band and exposure < prefer_exposure_below_us, reduces gain so AE can
    increase exposure (lower gain → better noise).
    """

    def __init__(
        self,
        gain_min: float = 1.0,
        gain_max: float = 16.0,
        target_low: float = TARGET_LOW,
        target_high: float = TARGET_HIGH,
        smoothing_tau: float = 0.05,
        prefer_exposure_below_us: int = PREFER_EXPOSURE_BELOW_US,
        verbose: bool = True,
        bit_depth: int = 10,
        # Legacy params: accepted for caller compatibility, ignored.
        gain_step_threshold: float = 0.2,
        peak_smoothing_period_sec: float = 0.05,
        max_adjust_rate_hz: float = 20.0,
    ):
        self.gain_min = gain_min
        self.gain_max = gain_max
        self.target_low = target_low
        self.target_high = target_high
        self.target_mid = (target_low + target_high) / 2.0
        self.tau = smoothing_tau or peak_smoothing_period_sec
        self.prefer_exposure_below_us = prefer_exposure_below_us
        self.verbose = verbose
        self.bit_depth = bit_depth
        self._ema: float | None = None
        self._skip_remaining: int = 0

    def _peak_counts_suffix(self, peak: float, data: SpectrumData, steady: bool = False) -> str:
        counts = peak_to_counts(peak, self.bit_depth)
        if steady:
            flux = counts_to_flux_proxy(counts, data.exposure_us, data.gain)
            if flux is not None:
                return f" counts {counts:.0f} flux_proxy {flux:.1f}"
        return f" counts {counts:.0f}"

    @property
    def smoothed_peak(self) -> float | None:
        return self._ema

    def adjust(
        self,
        data: SpectrumData,
        get_gain: Callable[[], float],
        set_gain: Callable[[float], None],
        set_display: Callable[[float], None],
    ) -> bool:
        raw = float(np.max(data.intensity))
        self._ema = _filtered_peak(raw, data.exposure_us, self._ema, self.tau)

        if self._skip_remaining > 0:
            self._skip_remaining -= 1
            return False

        peak = self._ema
        current = get_gain()
        exposure_us = data.exposure_us or 0

        if peak > self.target_high:
            if current <= self.gain_min:
                return False
            ratio = _SATURATED_RATIO if peak >= _SATURATED else max(_MAX_DOWN, self.target_mid / peak)
            new = max(self.gain_min, current * ratio)
            if new >= current - 0.01:
                return False
            set_gain(new)
            set_display(new)
            self._skip_remaining = _SKIP_FRAMES_AFTER_CHANGE
            if self.verbose:
                print(f"[AG] {new:.2f}x (peak {peak:.3f}↓{self._peak_counts_suffix(peak, data, steady=False)})")
            return True

        if peak < self.target_low:
            if current >= self.gain_max:
                return False
            # Predict: peak ∝ E×G. Only increase G when E is at max (strive to keep G small).
            ratio = (self.target_mid / max(peak, _MIN_PEAK)) * _PREDICT_SAFETY
            ratio = min(ratio, _MAX_UP_RATIO)
            new = min(self.gain_max, current * ratio)
            if new <= current + 0.01:
                return False
            set_gain(new)
            set_display(new)
            self._skip_remaining = _SKIP_FRAMES_AFTER_CHANGE
            if self.verbose:
                print(f"[AG] {new:.2f}x (peak {peak:.3f}↑ pred{self._peak_counts_suffix(peak, data, steady=False)})")
            return True

        if (
            exposure_us > 0
            and exposure_us < self.prefer_exposure_below_us
            and current > self.gain_min
        ):
            new = max(self.gain_min, current * _PREFER_GAIN_DOWN_RATIO)
            if new < current - 0.01:
                set_gain(new)
                set_display(new)
                self._skip_remaining = _SKIP_FRAMES_AFTER_CHANGE
                if self.verbose:
                    print(f"[AG] {new:.2f}x (prefer exposure, peak {peak:.3f}{self._peak_counts_suffix(peak, data, steady=False)})")
                return True

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
    gain_cooldown_remaining: int,  # kept for API compat; unused (each controller manages its own skip)
) -> int:
    """Run AE/AG for one frame. AE has priority; AG runs only when AE is idle or at limit."""
    ae_busy = False
    if auto_exposure_enabled:
        ae_busy = auto_exposure_ctrl.adjust(data, get_exposure, set_exposure, set_exposure_display)
    if auto_gain_enabled and not ae_busy:
        auto_gain_ctrl.adjust(data, get_gain, set_gain, set_gain_display)
    return 0
