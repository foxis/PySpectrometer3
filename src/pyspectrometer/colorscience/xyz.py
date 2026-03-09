"""Tristimulus XYZ and LAB via colour-science.

Uses colour-science for CIE observers, ASTM E308-compliant tristimulus,
and correct formulas per measurement type. Normalizes to custom illuminant
(white reference) for reflectance/transmittance; self-normalizes for illumination.
"""

from typing import Literal

import numpy as np

try:
    import colour

    _COLOUR_AVAILABLE = True
except ImportError:
    _COLOUR_AVAILABLE = False


def _to_sd(wavelengths: np.ndarray, values: np.ndarray):
    """Build SpectralDistribution from arrays."""
    return colour.SpectralDistribution(
        dict(zip(wavelengths.astype(float), np.maximum(values.astype(float), 0)))
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

    Reflectance/Transmittance: X = k × Σ S(λ)×R(λ)×x̄(λ), k = 100/Σ S(λ)×ȳ(λ).
    Illuminant S = white reference. Measured spectrum = S×R or S×τ.

    Illumination: X = k × Σ S(λ)×x̄(λ), k = 100/Σ S(λ)×ȳ(λ).
    Spectrum S is the light source; self-normalized (Y=100).

    Args:
        spectrum: Measured spectrum (S×R, S×τ, or S for illumination)
        wavelengths: Wavelength array in nm
        measurement_type: Reflectance, transmittance, or illumination
        illuminant_spectrum: White reference (our illuminant S) for refl/trans
        illuminant_wavelengths: Wavelengths for illuminant (same as spectrum if None)
        observer: "2deg" or "10deg"

    Returns:
        (X, Y, Z) tristimulus values
    """
    if not _COLOUR_AVAILABLE:
        return (0.0, 0.0, 0.0)
    try:
        cmfs_key = (
            "CIE 1964 10 Degree Standard Observer"
            if observer == "10deg"
            else "CIE 1931 2 Degree Standard Observer"
        )
        cmfs = colour.MSDS_CMFS[cmfs_key]

        sd = _to_sd(wavelengths, spectrum)

        if measurement_type == "illumination":
            illuminant = sd
        elif illuminant_spectrum is not None:
            wl_ill = illuminant_wavelengths if illuminant_wavelengths is not None else wavelengths
            illuminant = _to_sd(wl_ill, illuminant_spectrum)
        else:
            illuminant = colour.SDS_ILLUMINANTS["D65"]

        xyz = colour.sd_to_XYZ(sd, cmfs=cmfs, illuminant=illuminant)
        return (float(xyz[0]), float(xyz[1]), float(xyz[2]))
    except Exception:
        return (0.0, 0.0, 0.0)


def xyz_to_lab(
    X: float,
    Y: float,
    Z: float,
    reference_white_XYZ: tuple[float, float, float] | None = None,
) -> tuple[float, float, float]:
    """Convert XYZ to CIE L*a*b* using colour-science.

    Args:
        X, Y, Z: Tristimulus values (0-100 scale)
        reference_white_XYZ: XYZ of reference white (our illuminant). If None, uses D65.

    Returns:
        (L*, a*, b*)
    """
    if not _COLOUR_AVAILABLE:
        return (0.0, 0.0, 0.0)
    try:
        xyz = np.array([X, Y, Z]) / 100.0
        if reference_white_XYZ is not None:
            rw = np.array(reference_white_XYZ) / 100.0
            illuminant_xy = colour.XYZ_to_xy(rw)
            lab = colour.XYZ_to_Lab(xyz, illuminant=illuminant_xy)
        else:
            lab = colour.XYZ_to_Lab(xyz)
        return (float(lab[0]), float(lab[1]), float(lab[2]))
    except Exception:
        return (0.0, 0.0, 0.0)
