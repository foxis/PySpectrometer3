"""Simulation tests for auto-exposure and auto-gain.

Sensor model: 10-bit linear with optional light scale.
  peak = min(1.0, light * (exposure_sec * gain * 1023) / 1023)

Frame timing: 30 fps for short exposures; frame_period = exposure_sec when long.

Expected behavior:
- Converges to [0.80, 0.90] within ~80 frames from any starting point.
- Never oscillates (peak does not repeatedly cross into overexposure after convergence).
- With varying light (0.5, 1, 2, ..., 10), convergence and stability hold.
- When in band and exposure < 500ms, AG reduces gain so AE can increase exposure (noise optimization).
"""

import numpy as np
import pytest

from ..core.spectrum import SpectrumData
from ..processing.auto_controls import (
    AutoExposureController,
    AutoGainController,
    TARGET_LOW,
    TARGET_HIGH,
)

_SENSOR_MAX = 1023
_MIN_FRAME_PERIOD = 1.0 / 30.0

# Gentler step (18% max up) + skip frames for pipeline lag; allow more steps to converge.
CONVERGE_STEPS = 200
STABILITY_STEPS = 120

PEAK_MAX_OK = 0.93
PEAK_MIN_OK = 0.76

# Light intensities to test: 0.5, 1, 2, ..., 10.
LIGHT_INTENSITIES = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]


def _sensor_peak(exposure_us: int, gain: float, light: float = 1.0) -> float:
    """10-bit linear sensor with light scaling. Clips at full scale."""
    linear = light * (exposure_us / 1e6) * gain * _SENSOR_MAX
    raw = min(_SENSOR_MAX, int(linear))
    return raw / float(_SENSOR_MAX)


def _frame_period(exposure_us: int) -> float:
    return max(_MIN_FRAME_PERIOD, exposure_us / 1e6)


def _make_controllers() -> tuple[AutoExposureController, AutoGainController]:
    ae = AutoExposureController(
        exposure_min_us=100,
        exposure_max_us=1_000_000,
        smoothing_tau=0.05,
        verbose=False,
    )
    ag = AutoGainController(
        gain_min=1.0,
        gain_max=16.0,
        smoothing_tau=0.05,
        prefer_exposure_below_us=500_000,
        verbose=False,
    )
    return ae, ag


def _run_sim(
    exposure_us: int,
    gain: float,
    steps: int,
    light: float = 1.0,
) -> list[float]:
    """Run AE/AG simulation with given light intensity. Returns raw sensor peaks per step."""
    ae, ag = _make_controllers()

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

    noop_i = lambda _: None
    noop_f = lambda _: None

    peaks: list[float] = []
    wavelengths = np.linspace(380.0, 720.0, 64)

    for _ in range(steps):
        peak_val = _sensor_peak(exposure_us, gain, light)
        peaks.append(peak_val)
        intensity = np.full(64, peak_val, dtype=np.float64)
        data = SpectrumData(
            intensity=intensity,
            wavelengths=wavelengths,
            exposure_us=exposure_us,
            gain=gain,
        )
        ae_busy = ae.adjust(data, get_exposure, set_exposure, noop_i)
        if not ae_busy:
            ag.adjust(data, get_gain, set_gain, noop_f)

    return peaks


def _first_in_band(peaks: list[float]) -> int | None:
    for i, p in enumerate(peaks):
        if TARGET_LOW <= p <= TARGET_HIGH:
            return i
    return None


def test_converges_from_underexposed():
    """Start very underexposed (light=1): converges to band within CONVERGE_STEPS."""
    peaks = _run_sim(exposure_us=5_000, gain=1.0, steps=CONVERGE_STEPS + STABILITY_STEPS)
    converged = _first_in_band(peaks)

    assert converged is not None, (
        f"Never entered band [{TARGET_LOW}, {TARGET_HIGH}]. Last 10: {peaks[-10:]}"
    )
    assert converged <= CONVERGE_STEPS, (
        f"Converged at step {converged}, expected ≤ {CONVERGE_STEPS}. "
        f"Near convergence: {peaks[max(0, converged-3):converged+5]}"
    )

    final = peaks[-50:]
    assert max(final) <= PEAK_MAX_OK, f"Overexposed after convergence: max={max(final):.3f}"
    assert min(final) >= PEAK_MIN_OK, f"Underexposed after convergence: min={min(final):.3f}"


def test_converges_from_overexposed():
    """Start saturated (light=1): reduces to band without oscillation."""
    peaks = _run_sim(exposure_us=400_000, gain=12.0, steps=CONVERGE_STEPS + STABILITY_STEPS)
    converged = _first_in_band(peaks)

    assert converged is not None, (
        f"Never entered band. Last 10: {peaks[-10:]}"
    )
    assert converged <= CONVERGE_STEPS, (
        f"Converged at step {converged}, expected ≤ {CONVERGE_STEPS}."
    )

    post = peaks[converged:]
    assert max(post) <= PEAK_MAX_OK, (
        f"Overshoot after convergence: max={max(post):.3f}. post[:20]={post[:20]}"
    )

    crossings = sum(
        1 for i in range(1, len(post))
        if post[i] > TARGET_HIGH and post[i - 1] <= TARGET_HIGH
    )
    assert crossings <= 1, (
        f"Oscillation: crossed into overexposure {crossings} times after convergence."
    )


def test_stays_stable_in_band():
    """Start with peak already in band: prefer-exposure may nudge slightly; most samples stay near band."""
    peaks = _run_sim(exposure_us=100_000, gain=8.5, steps=200)
    final = peaks[-100:]
    out_of_band = sum(1 for p in final if p < TARGET_LOW or p > TARGET_HIGH)
    assert out_of_band <= 25, (
        f"Drifted out of band {out_of_band}/100 times. min={min(final):.3f} max={max(final):.3f}"
    )
    assert min(final) >= PEAK_MIN_OK and max(final) <= PEAK_MAX_OK, (
        f"Peak left acceptable range. min={min(final):.3f} max={max(final):.3f}"
    )


def _underexposed_start(light: float) -> tuple[int, float]:
    """(exposure_us, gain) that is underexposed for this light."""
    return 5_000, 1.0


def _overexposed_start(light: float) -> tuple[int, float]:
    """(exposure_us, gain) that is overexposed (saturated) for this light."""
    # light * (exposure_us/1e6) * gain >= 1.2 to ensure clipped. exposure_max = 1e6.
    # e.g. light=1: 0.4*12=4.8 ok. light=10: 0.04*12=4.8 ok. light=0.5: 0.8*12=4.8 ok.
    exposure = min(1_000_000, int(500_000 / max(light, 0.1)))
    return exposure, 12.0


@pytest.mark.parametrize("light", LIGHT_INTENSITIES)
def test_converges_from_underexposed_varying_light(light: float):
    """For each light level, start underexposed and converge to band."""
    exp_us, gain = _underexposed_start(light)
    peaks = _run_sim(exposure_us=exp_us, gain=gain, steps=CONVERGE_STEPS + STABILITY_STEPS, light=light)
    converged = _first_in_band(peaks)

    assert converged is not None, (
        f"light={light}: never entered band. Last 10: {peaks[-10:]}"
    )
    assert converged <= CONVERGE_STEPS, (
        f"light={light}: converged at step {converged}, expected ≤ {CONVERGE_STEPS}. "
        f"Near: {peaks[max(0, converged-3):converged+5]}"
    )

    final = peaks[-50:]
    assert max(final) <= PEAK_MAX_OK, (
        f"light={light}: overexposed after convergence: max={max(final):.3f}"
    )
    assert min(final) >= PEAK_MIN_OK, (
        f"light={light}: underexposed after convergence: min={min(final):.3f}"
    )


@pytest.mark.parametrize("light", LIGHT_INTENSITIES)
def test_converges_from_overexposed_varying_light(light: float):
    """For each light level, start overexposed and converge to band without oscillating."""
    exp_us, gain = _overexposed_start(light)
    peaks = _run_sim(exposure_us=exp_us, gain=gain, steps=CONVERGE_STEPS + STABILITY_STEPS, light=light)
    converged = _first_in_band(peaks)

    assert converged is not None, (
        f"light={light}: never entered band. Last 10: {peaks[-10:]}"
    )
    assert converged <= CONVERGE_STEPS, (
        f"light={light}: converged at step {converged}, expected ≤ {CONVERGE_STEPS}."
    )

    post = peaks[converged:]
    assert max(post) <= PEAK_MAX_OK, (
        f"light={light}: overshoot after convergence: max={max(post):.3f}"
    )

    crossings = sum(
        1 for i in range(1, len(post))
        if post[i] > TARGET_HIGH and post[i - 1] <= TARGET_HIGH
    )
    assert crossings <= 1, (
        f"light={light}: oscillation: crossed into overexposure {crossings} times."
    )
