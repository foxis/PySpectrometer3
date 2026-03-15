"""Simulation test for auto-exposure and auto-gain with constant light.

Simulates:
- 10-bit sensor: intensity = min(1, raw/1023) with raw = min(1023, exposure_sec * gain * 1023).
- Frame timing: exposure < 1/30 s → 30 fps (frame period 1/30 s); else frame period = exposure.
- Convergence and stability of AE/AG within ~100 steps, no major oscillations, no over/underexposure.

Real-world note: stream_camera.py and mode_context run at most one of AE or AG per frame (exposure
first; if exposure adjusted, skip gain). Even with that, alternating frames (E down, next frame G down)
still over-correct and cause oscillation (saturated -> too dark -> saturated). So after an exposure
adjust we cooldown gain for a few frames so we see the new exposure before changing gain.
"""

import time
from unittest.mock import patch

import numpy as np
import pytest

from ..core.spectrum import SpectrumData
from ..processing.auto_controls import (
    AutoExposureController,
    AutoGainController,
)

# 10-bit sensor max
_SENSOR_MAX = 1023

# Target band (must match auto_controls: 0.90–0.95)
TARGET_LOW = 0.90
TARGET_HIGH = 0.95

# Convergence: expect peak in band within this many steps
CONVERGE_STEPS = 100
# After convergence, check stability for this many steps (no major oscillations)
STABILITY_STEPS = 150

# Allowed peak when "converged": no overexposure, no strong underexposure
PEAK_MAX_OK = 0.97
PEAK_MIN_OK = 0.85

# Frame period when exposure <= 1/30 s (30 fps)
MIN_FRAME_PERIOD_SEC = 1.0 / 30.0


def _sensor_peak_10bit(exposure_us: int, gain: float) -> float:
    """Constant light: linear in exposure*gain, 10-bit quantized, normalized to 0–1."""
    exposure_sec = exposure_us / 1e6
    linear = exposure_sec * gain * _SENSOR_MAX
    raw = min(_SENSOR_MAX, int(linear))
    return raw / float(_SENSOR_MAX)


def _frame_period_sec(exposure_us: int) -> float:
    """Exposure < 1/30 s → 30 fps; else frame period = exposure."""
    exposure_sec = exposure_us / 1e6
    return max(MIN_FRAME_PERIOD_SEC, exposure_sec)


def test_auto_exposure_gain_converges_constant_light():
    """Constant light: AE/AG converge in ~100 steps, then stay in band without major oscillations."""
    # Simulated camera state
    exposure_us = 10_000
    gain = 2.0

    def get_exposure() -> int:
        return exposure_us

    def set_exposure(v: int) -> None:
        nonlocal exposure_us
        exposure_us = v

    def get_gain() -> float:
        return gain

    def set_gain(v: float) -> None:
        nonlocal gain
        gain = v

    # No-op display updaters
    def set_display_exposure(_: int) -> None:
        pass

    def set_display_gain(_: float) -> None:
        pass

    auto_exposure = AutoExposureController(
        exposure_min_us=100,
        exposure_max_us=1_000_000,
        exposure_preferred_max_us=500_000,
        peak_smoothing_period_sec=0.04,
        max_adjust_rate_hz=10.0,
        verbose=False,
    )
    auto_gain = AutoGainController(
        gain_min=1.0,
        gain_max=16.0,
        peak_smoothing_period_sec=0.04,
        max_adjust_rate_hz=10.0,
        verbose=False,
    )

    sim_time = 0.0
    peaks: list[float] = []
    converged_at: int | None = None

    def mock_monotonic() -> float:
        return sim_time

    n_pixels = 64
    wavelengths = np.linspace(380.0, 720.0, n_pixels)

    with patch("pyspectrometer.processing.auto_controls.time.monotonic", side_effect=mock_monotonic):
        total_steps = CONVERGE_STEPS + STABILITY_STEPS
        for step in range(total_steps):
            # Frame period: exposure < 1/30 s → 30 fps
            frame_period = _frame_period_sec(exposure_us)
            sim_time += frame_period

            peak_val = _sensor_peak_10bit(exposure_us, gain)
            peaks.append(peak_val)
            intensity = np.full(n_pixels, peak_val, dtype=np.float64)
            data = SpectrumData(
                intensity=intensity,
                wavelengths=wavelengths,
                exposure_us=exposure_us,
                gain=gain,
            )

            # Same order as mode_context: exposure first, then gain if exposure didn't change
            exposure_adjusted = auto_exposure.adjust(
                data, get_exposure, set_exposure, set_display_exposure
            )
            if not exposure_adjusted:
                auto_gain.adjust(data, get_gain, set_gain, set_display_gain)

            # Check convergence (first time peak in band)
            if converged_at is None and TARGET_LOW <= peak_val <= TARGET_HIGH:
                converged_at = step

    # Must converge within CONVERGE_STEPS
    assert converged_at is not None, (
        f"Peak did not enter band [{TARGET_LOW}, {TARGET_HIGH}] within {CONVERGE_STEPS} steps. "
        f"Last 10 peaks: {peaks[-10:]}"
    )
    assert converged_at <= CONVERGE_STEPS, (
        f"Converged at step {converged_at}, expected within {CONVERGE_STEPS}. "
        f"Peaks near convergence: {peaks[converged_at - 5 : converged_at + 10]}"
    )

    # Stability: after convergence, no major oscillations (peak should stay in/near band)
    post_converge = peaks[converged_at : converged_at + STABILITY_STEPS]
    out_of_band = sum(1 for p in post_converge if p < TARGET_LOW or p > TARGET_HIGH)
    # Allow a few frames outside band (smoothing/quantization), but not many
    assert out_of_band <= max(20, STABILITY_STEPS // 5), (
        f"Too many steps outside band after convergence: {out_of_band}/{len(post_converge)}. "
        f"Peaks: min={min(post_converge):.3f}, max={max(post_converge):.3f}"
    )

    # When converged: no overexposure (peak <= PEAK_MAX_OK)
    final_peaks = peaks[-50:]
    assert max(final_peaks) <= PEAK_MAX_OK, (
        f"Overexposed when converged: max peak {max(final_peaks):.3f} > {PEAK_MAX_OK}"
    )

    # When converged: not severely underexposed (peak >= PEAK_MIN_OK)
    assert min(final_peaks) >= PEAK_MIN_OK, (
        f"Underexposed when converged: min peak {min(final_peaks):.3f} < {PEAK_MIN_OK}"
    )

    # No pulsing: after convergence we must not repeatedly cross into overexposure (normal ↔ over).
    crossings_into_over = 0
    for i in range(1, len(post_converge)):
        if post_converge[i] > TARGET_HIGH and post_converge[i - 1] <= TARGET_HIGH:
            crossings_into_over += 1
    assert crossings_into_over <= 1, (
        f"Exposure pulsing: crossed into overexposure (>{TARGET_HIGH}) {crossings_into_over} times "
        f"after convergence (max 1 allowed). Peaks (first 60): {post_converge[:60]}"
    )


def test_auto_exposure_gain_from_overexposed_no_overshoot():
    """Start overexposed; after converging into band, peak must not jump back up (no overshoot)."""
    exposure_us = 400_000  # 0.4 s
    gain = 12.0
    # peak = min(1, 0.4 * 12) = 1.0 (saturated)

    def get_exposure() -> int:
        return exposure_us

    def set_exposure(v: int) -> None:
        nonlocal exposure_us
        exposure_us = v

    def get_gain() -> float:
        return gain

    def set_gain(v: float) -> None:
        nonlocal gain
        gain = v

    def set_display_exposure(_: int) -> None:
        pass

    def set_display_gain(_: float) -> None:
        pass

    auto_exposure = AutoExposureController(
        exposure_min_us=100,
        exposure_max_us=1_000_000,
        exposure_preferred_max_us=500_000,
        peak_smoothing_period_sec=0.04,
        max_adjust_rate_hz=10.0,
        verbose=False,
    )
    auto_gain = AutoGainController(
        gain_min=1.0,
        gain_max=16.0,
        peak_smoothing_period_sec=0.04,
        max_adjust_rate_hz=10.0,
        verbose=False,
    )

    sim_time = 0.0
    peaks: list[float] = []
    converged_at: int | None = None

    def mock_monotonic() -> float:
        return sim_time

    n_pixels = 64
    wavelengths = np.linspace(380.0, 720.0, n_pixels)

    with patch("pyspectrometer.processing.auto_controls.time.monotonic", side_effect=mock_monotonic):
        total_steps = CONVERGE_STEPS + STABILITY_STEPS
        for step in range(total_steps):
            frame_period = _frame_period_sec(exposure_us)
            sim_time += frame_period

            peak_val = _sensor_peak_10bit(exposure_us, gain)
            peaks.append(peak_val)
            intensity = np.full(n_pixels, peak_val, dtype=np.float64)
            data = SpectrumData(
                intensity=intensity,
                wavelengths=wavelengths,
                exposure_us=exposure_us,
                gain=gain,
            )

            exposure_adjusted = auto_exposure.adjust(
                data, get_exposure, set_exposure, set_display_exposure
            )
            if not exposure_adjusted:
                auto_gain.adjust(data, get_gain, set_gain, set_display_gain)

            if converged_at is None and TARGET_LOW <= peak_val <= TARGET_HIGH:
                converged_at = step

    assert converged_at is not None, (
        f"Did not converge from overexposed within {CONVERGE_STEPS} steps. Last 10 peaks: {peaks[-10:]}"
    )

    # After convergence: must not go back above PEAK_MAX_OK (catches reduce->0.65->increase->1.0 oscillation).
    post = peaks[converged_at:]
    assert max(post) <= PEAK_MAX_OK, (
        f"Overshoot after convergence: max peak {max(post):.3f} > {PEAK_MAX_OK}. "
        f"Peaks after converge: {post[:30]}..."
    )
    # No oscillation: must not cross back into overexposure more than once (no reduce/increase/reduce cycle).
    crossings = sum(1 for i in range(1, len(post)) if post[i] > TARGET_HIGH and post[i - 1] <= TARGET_HIGH)
    assert crossings <= 1, (
        f"Oscillation: crossed into overexposure {crossings} times after convergence (max 1). post[:40]={post[:40]}"
    )
