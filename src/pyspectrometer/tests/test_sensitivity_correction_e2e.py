"""End-to-end sensitivity correction: CRR-style fit from A×CMOS, validate shape on D65×CMOS.

Why: Proves the fitted correction table generalizes beyond the calibration illuminant when the
true instrument response is the datasheet CMOS curve (flat-field goal: corrected ≈ reference SPD).
"""

from __future__ import annotations

import numpy as np
import pytest

from ..data.reference_spectra import ReferenceSource, get_reference_spectrum
from ..processing.sensitivity_correction import SensitivityCorrection


def _skip_without_cmos(eng: SensitivityCorrection) -> None:
    probe = np.linspace(420.0, 680.0, 32, dtype=np.float64)
    if eng.interpolate(probe) is None:
        pytest.skip("Datasheet CMOS sensitivity CSV not found")


def _to_unit_interval(values: np.ndarray) -> np.ndarray:
    peak = float(np.max(values))
    if peak <= 1e-15:
        return values
    return np.clip(values / peak, 0.0, 1.0)


def _metrics(
    corrected: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float]:
    """Pearson r and normalized RMSE on masked samples (shape agreement with D65)."""
    c = corrected[mask].astype(np.float64)
    r = reference[mask].astype(np.float64)
    if c.size < 8 or r.size < 8:
        return 0.0, 1.0
    c0 = c - np.mean(c)
    r0 = r - np.mean(r)
    denom = np.linalg.norm(c0) * np.linalg.norm(r0)
    pearson = float(np.dot(c0, r0) / (denom + 1e-15))
    cn = c / (np.max(np.abs(c)) + 1e-15)
    rn = r / (np.max(np.abs(r)) + 1e-15)
    nrmse = float(np.sqrt(np.mean((cn - rn) ** 2)))
    return pearson, nrmse


def _e2e_shape_error_one(
    eng: SensitivityCorrection,
    wl: np.ndarray,
    ref_a: np.ndarray,
    ref_d65: np.ndarray,
    cmos: np.ndarray,
    noise_std: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Single trial: noisy A×CMOS calibration, then D65×CMOS correction vs reference D65."""
    meas_a = _to_unit_interval(ref_a * cmos)
    if noise_std > 0.0:
        meas_a = np.clip(meas_a + rng.normal(0.0, noise_std, wl.shape), 0.0, 1.0)

    fitted = eng.recalibrate_from_measurement(meas_a, wl, ref_a)
    assert fitted is not None
    wl_fit, sens = fitted
    eng.reset_to_datasheet()
    eng.set_custom_curve(wl_fit, sens)

    meas_d65 = _to_unit_interval(ref_d65 * cmos)
    corrected = np.asarray(eng.apply(meas_d65.astype(np.float32), wl), dtype=np.float64)

    mask = ref_d65 > 0.05 * float(np.max(ref_d65))
    assert int(np.sum(mask)) > 24
    return _metrics(corrected, ref_d65, mask)


def _aggregate_e2e(
    noise_std: float,
    n_trials: int,
    base_seed: int,
) -> tuple[float, float]:
    """Median Pearson / NRMSE over trials (stabilizes single-draw RNG luck)."""
    eng = SensitivityCorrection(config=None)
    _skip_without_cmos(eng)
    wl = np.linspace(400.0, 720.0, 384, dtype=np.float64)
    ref_a = np.maximum(get_reference_spectrum(ReferenceSource.A, wl), 1e-15)
    ref_d65 = np.maximum(get_reference_spectrum(ReferenceSource.D65, wl), 1e-15)
    cmos = np.asarray(eng.interpolate(wl), dtype=np.float64)

    pearsons: list[float] = []
    nrmses: list[float] = []
    for k in range(n_trials):
        rng = np.random.default_rng(base_seed + k * 9973 + int(noise_std * 1_000_000))
        p, n = _e2e_shape_error_one(eng, wl, ref_a, ref_d65, cmos, noise_std, rng)
        pearsons.append(p)
        nrmses.append(n)
    return float(np.median(pearsons)), float(np.median(nrmses))


@pytest.mark.parametrize(
    ("noise_std", "n_trials", "min_pearson", "max_nrmse"),
    [
        (0.0, 1, 0.993, 0.075),
        (0.01, 11, 0.94, 0.095),
        # σ=0.03: Savitzky–Golay + ratio fit degrades; median r stays ~0.85+
        (0.03, 11, 0.84, 0.21),
        # σ=0.06: heavy noise on unit-interval calibration frame; shape still related to D65
        (0.06, 15, 0.72, 0.32),
    ],
)
def test_sensitivity_fit_from_a_restores_d65_shape(
    noise_std: float,
    n_trials: int,
    min_pearson: float,
    max_nrmse: float,
) -> None:
    """CMOS×A → fit → apply to CMOS×D65; corrected spectrum should match D65 (noise on calibration frame)."""
    pearson, nrmse = _aggregate_e2e(noise_std, n_trials, base_seed=20250320)

    assert pearson > min_pearson, (
        f"Pearson r={pearson:.4f} below {min_pearson} (noise_std={noise_std}, n_trials={n_trials})"
    )
    assert nrmse < max_nrmse, (
        f"NRMSE={nrmse:.4f} above {max_nrmse} (noise_std={noise_std}, n_trials={n_trials})"
    )
