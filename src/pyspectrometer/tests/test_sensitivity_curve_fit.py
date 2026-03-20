"""Sensitivity curve inference: measured / reference ratio vs datasheet CMOS."""

import numpy as np

from ..processing.sensitivity_curve_fit import fit_sensitivity_values


def test_fit_recovers_constant_ratio_on_plateau():
    """On a band where reference is flat and measured is proportional, ratio tracks CMOS scale."""
    wl = np.linspace(400.0, 700.0, 128, dtype=np.float64)
    ref = np.ones_like(wl)
    ref[wl < 420] = 0.01
    ref[wl > 680] = 0.01
    cmos = 0.5 + 0.5 * np.sin((wl - 400) / 300 * np.pi)
    cmos = np.clip(cmos, 0.2, 1.0)
    measured = ref * cmos * 2.0
    out = fit_sensitivity_values(wl, measured, ref, cmos)
    mid = (wl >= 450) & (wl <= 650)
    rel_err = np.abs(out[mid] - cmos[mid]) / (cmos[mid] + 1e-6)
    assert float(np.median(rel_err)) < 0.15
