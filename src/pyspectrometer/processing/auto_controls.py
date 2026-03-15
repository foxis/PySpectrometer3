"""Auto-exposure and auto-gain: keep spectrum peak in [target_low, target_high].

Proportional control: new_setting = current_setting * (target_mid / peak), clamped per step.
When saturated (peak ~1.0): halve the setting immediately for fast recovery.
EMA filter uses dt = max(exposure_sec, 1/30s) so long exposures (which self-average mains)
get full weight, while short exposures are smoothed across frames.

AE (exposure) is the primary actuator. AG (gain) runs only when AE is in-band or at its limit.
One skip frame after each setting change so the camera can settle.
"""

from collections.abc import Callable
import numpy as np
from ..core.spectrum import SpectrumData

# Default target band: configurable per-instance.
TARGET_LOW = 0.80
TARGET_HIGH = 0.90

# Maximum ratio change per step.
_MAX_UP = 2.0    # never more than 2× increase
_MAX_DOWN = 0.5  # never more than 2× decrease (i.e. ratio >= 0.5)

# Saturated: halve immediately instead of gentle proportional step.
_SATURATED = 0.99

_DEFAULT_DT = 1.0 / 30.0
_MIN_PEAK = 1e-6


def _ema(raw: float, prev: float | None, dt: float, tau: float) -> float:
    """Exponential moving average. alpha = dt/tau, clamped to [0, 1].

    dt should be the actual inter-frame time (not exposure_us alone), so the
    filter cutoff stays consistent regardless of exposure setting.
    """
    alpha = min(1.0, dt / tau)
    return raw if prev is None else alpha * raw + (1.0 - alpha) * prev


def _frame_dt(exposure_us: int | None) -> float:
    """Inter-frame time: at least 1/30 s; longer exposures extend the frame period."""
    if not exposure_us:
        return _DEFAULT_DT
    return max(_DEFAULT_DT, exposure_us / 1e6)


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
        self._ema: float | None = None
        self._skip: bool = False
        self.at_max: bool = False  # exposure is at max and peak still low; AG must help

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
        self._ema = _ema(raw, self._ema, _frame_dt(data.exposure_us), self.tau)

        if self._skip:
            self._skip = False
            return True  # still busy: camera hasn't settled, block AG

        peak = self._ema
        current = get_exposure()

        if peak > self.target_high:
            self.at_max = False
            if current <= self.exposure_min_us:
                return False  # at floor; AG must reduce gain
            ratio = _MAX_DOWN if peak >= _SATURATED else max(_MAX_DOWN, self.target_mid / peak)
            new = max(self.exposure_min_us, int(current * ratio))
            if new >= current:
                return False
            set_exposure(new)
            set_display(new)
            self._skip = True
            if self.verbose:
                print(f"[AE] {new} us (peak {peak:.3f}↓)")
            return True

        if peak < self.target_low:
            if self.at_max:
                return False  # exposure can't help; AG must increase gain
            if current >= self.exposure_max_us:
                self.at_max = True
                return False
            ratio = min(_MAX_UP, self.target_mid / max(peak, _MIN_PEAK))
            new = min(self.exposure_max_us, int(current * ratio))
            if new <= current:
                self.at_max = True
                return False
            set_exposure(new)
            set_display(new)
            self._skip = True
            if self.verbose:
                print(f"[AE] {new} us (peak {peak:.3f}↑)")
            return True

        # In band.
        self.at_max = False
        return False


class AutoGainController:
    """Proportional AG: scales gain so peak converges to target_mid.

    Secondary actuator: only runs when AE is in-band or at its exposure limit.
    """

    def __init__(
        self,
        gain_min: float = 1.0,
        gain_max: float = 16.0,
        target_low: float = TARGET_LOW,
        target_high: float = TARGET_HIGH,
        smoothing_tau: float = 0.05,
        verbose: bool = True,
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
        self.verbose = verbose
        self._ema: float | None = None
        self._skip: bool = False

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
        self._ema = _ema(raw, self._ema, _frame_dt(data.exposure_us), self.tau)

        if self._skip:
            self._skip = False
            return False

        peak = self._ema
        current = get_gain()

        if peak > self.target_high:
            if current <= self.gain_min:
                return False
            ratio = _MAX_DOWN if peak >= _SATURATED else max(_MAX_DOWN, self.target_mid / peak)
            new = max(self.gain_min, current * ratio)
            if new >= current - 0.01:
                return False
            set_gain(new)
            set_display(new)
            self._skip = True
            if self.verbose:
                print(f"[AG] {new:.2f}x (peak {peak:.3f}↓)")
            return True

        if peak < self.target_low:
            if current >= self.gain_max:
                return False
            ratio = min(_MAX_UP, self.target_mid / max(peak, _MIN_PEAK))
            new = min(self.gain_max, current * ratio)
            if new <= current + 0.01:
                return False
            set_gain(new)
            set_display(new)
            self._skip = True
            if self.verbose:
                print(f"[AG] {new:.2f}x (peak {peak:.3f}↑)")
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
