"""Tristimulus XYZ and L*a*b* colorimetry.

Uses CIE CMF data from colour-science but performs integration via numpy
to support any wavelength spacing (colour.sd_to_XYZ requires strict 1/5/10/20 nm
intervals per ASTM E308-15, which calibrated spectrometers rarely satisfy).

Illumination (self-luminous sources):
  1. Normalize SPD by total integral → spectral shape only (removes absolute brightness)
     S_norm(λ) = S(λ) / ∫S(λ)dλ
  2. Scale by equal-energy factor k so equal-energy E gives Y = 100
     k = 100 / ∫(1/Δλ)·ȳ(λ)dλ
  3. X = k·∫S_norm·x̄dλ,  Y = k·∫S_norm·ȳdλ,  Z = k·∫S_norm·z̄dλ
  4. Reference white for L*a*b*: equal-energy E → X_E = Y_E = Z_E = 100
     (CIE defines E at x=y=1/3, which yields X=Y=Z for any Y-normalization)

Reflectance/Transmittance (spectrum is already white-corrected ≈ R(λ)):
  X = k · ∫ R(λ)·W(λ)·x̄(λ)dλ,  k = 100 / ∫ W(λ)·ȳ(λ)dλ
  where W(λ) is the raw white-reference capture.
"""

from typing import Literal

import numpy as np

try:
    import colour

    _COLOUR_AVAILABLE = True
except ImportError:
    _COLOUR_AVAILABLE = False

# Lazily cached CMF arrays: (wavelengths, xyz_values)
_CMF_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _cmf_arrays(observer: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (wavelengths, xyz) for the requested observer (cached)."""
    if observer not in _CMF_CACHE:
        key = (
            "CIE 1964 10 Degree Standard Observer"
            if observer == "10deg"
            else "CIE 1931 2 Degree Standard Observer"
        )
        cmfs = colour.MSDS_CMFS[key]
        _CMF_CACHE[observer] = (cmfs.wavelengths.copy(), cmfs.values.copy())
    return _CMF_CACHE[observer]


def _interp_to_cmf(
    values: np.ndarray,
    wavelengths: np.ndarray,
    wl_cmf: np.ndarray,
) -> np.ndarray:
    """Resample a spectrum onto the CMF wavelength grid via linear interpolation."""
    return np.interp(
        wl_cmf,
        np.asarray(wavelengths, dtype=float),
        np.maximum(np.asarray(values, dtype=float), 0.0),
        left=0.0,
        right=0.0,
    )


def calculate_XYZ(
    spectrum: np.ndarray,
    wavelengths: np.ndarray,
    measurement_type: Literal["reflectance", "transmittance", "illumination"],
    illuminant_spectrum: np.ndarray | None = None,
    illuminant_wavelengths: np.ndarray | None = None,
    observer: Literal["2deg", "10deg"] = "10deg",
) -> tuple[float, float, float]:
    """Calculate CIE XYZ tristimulus values.

    Uses numpy trapezoid integration on the CMF grid to support any wavelength
    spacing (unlike colour.sd_to_XYZ which requires 1/5/10/20 nm steps).

    Args:
        spectrum: Measured spectrum (reflectance ≈ R(λ), or raw S for illumination)
        wavelengths: Wavelength array in nm (any spacing)
        measurement_type: "illumination", "reflectance", or "transmittance"
        illuminant_spectrum: White reference W(λ) for refl/trans (raw camera capture)
        illuminant_wavelengths: Wavelengths for illuminant (defaults to spectrum wl)
        observer: "2deg" (CIE 1931) or "10deg" (CIE 1964)

    Returns:
        (X, Y, Z) — normalized so Y=100 for a perfect white/illuminant
    """
    if not _COLOUR_AVAILABLE:
        return (0.0, 0.0, 0.0)

    try:
        wl_cmf, cmf_xyz = _cmf_arrays(observer)

        S = _interp_to_cmf(spectrum, wavelengths, wl_cmf)

        if measurement_type == "illumination":
            return _xyz_illuminant(S, cmf_xyz, wl_cmf)

        if illuminant_spectrum is None:
            return (0.0, 0.0, 0.0)

        wl_ill = illuminant_wavelengths if illuminant_wavelengths is not None else wavelengths
        E = _interp_to_cmf(illuminant_spectrum, wl_ill, wl_cmf)
        return _xyz_reflective(S, E, cmf_xyz, wl_cmf)

    except Exception as e:
        print(f"[XYZ] calculate_XYZ failed: {e}")
        return (0.0, 0.0, 0.0)


def _xyz_illuminant(
    S: np.ndarray,
    cmf_xyz: np.ndarray,
    wl: np.ndarray,
) -> tuple[float, float, float]:
    """Integrate self-luminous SPD with CMFs, scaled to equal-energy reference.

    Normalizes the SPD by its total integral (removes absolute brightness,
    keeps only spectral shape).  The scale factor k ensures that a flat
    equal-energy spectrum E produces Y = 100.  Other sources get Y ≠ 100
    based on their spectral content, enabling meaningful L*a*b* comparisons
    using the equal-energy illuminant (100, 100, 100) as the reference white.
    """
    total = float(np.trapezoid(S, wl))
    if total < 1e-12:
        return (0.0, 0.0, 0.0)
    S_norm = S / total  # spectral shape only

    # Equal-energy scale: flat SPD (integral=1) gives reference Y_E
    dw = float(wl[-1] - wl[0])
    Y_E = float(np.trapezoid(cmf_xyz[:, 1] / dw, wl))
    k = 100.0 / Y_E if Y_E > 1e-12 else 1.0

    X = k * float(np.trapezoid(S_norm * cmf_xyz[:, 0], wl))
    Y = k * float(np.trapezoid(S_norm * cmf_xyz[:, 1], wl))
    Z = k * float(np.trapezoid(S_norm * cmf_xyz[:, 2], wl))
    return (X, Y, Z)


def _xyz_reflective(
    R: np.ndarray,
    E: np.ndarray,
    cmf_xyz: np.ndarray,
    wl: np.ndarray,
) -> tuple[float, float, float]:
    """Integrate reflectance·illuminant with CMFs, normalize by illuminant white."""
    k_denom = float(np.trapezoid(E * cmf_xyz[:, 1], wl))
    if k_denom < 1e-12:
        return (0.0, 0.0, 0.0)
    k = 100.0 / k_denom
    X = k * float(np.trapezoid(R * E * cmf_xyz[:, 0], wl))
    Y = k * float(np.trapezoid(R * E * cmf_xyz[:, 1], wl))
    Z = k * float(np.trapezoid(R * E * cmf_xyz[:, 2], wl))
    return (X, Y, Z)


def xyz_to_lab(
    X: float,
    Y: float,
    Z: float,
    reference_white_XYZ: tuple[float, float, float] | None = None,
) -> tuple[float, float, float]:
    """Convert XYZ to CIE L*a*b* using colour-science.

    Args:
        X, Y, Z: Tristimulus values (0–100 scale)
        reference_white_XYZ: XYZ of reference white. None → use D65.

    Returns:
        (L*, a*, b*)
    """
    if not _COLOUR_AVAILABLE:
        return (0.0, 0.0, 0.0)

    try:
        xyz = np.array([X, Y, Z]) / 100.0
        if reference_white_XYZ is not None:
            rw = np.asarray(reference_white_XYZ, dtype=float) / 100.0
            rw_sum = float(np.sum(rw))
            if rw_sum < 1e-10:
                # Degenerate white — fall back to D65
                lab = colour.XYZ_to_Lab(xyz)
            else:
                illuminant_xy = colour.XYZ_to_xy(rw)
                lab = colour.XYZ_to_Lab(xyz, illuminant=illuminant_xy)
        else:
            lab = colour.XYZ_to_Lab(xyz)
        return (float(lab[0]), float(lab[1]), float(lab[2]))
    except Exception as e:
        print(f"[XYZ] xyz_to_lab failed: {e}")
        return (0.0, 0.0, 0.0)
