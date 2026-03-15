"""Spectrum data structures for PySpectrometer3."""

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Extremum:
    """Peak or dip in spectrum. Height: positive peak, negative dip."""

    index: int
    position: float
    position_px: int | None
    height: float
    width: float
    is_dip: bool


@dataclass(frozen=True)
class Peak:
    """Represents a detected peak in a spectrum.

    Attributes:
        index: Pixel index of the peak
        wavelength: Wavelength at the peak position (nm)
        intensity: Intensity value at the peak (float32 0-1)
    """

    index: int
    wavelength: float
    intensity: float

    def __str__(self) -> str:
        return f"{self.wavelength:.1f}nm"


@dataclass
class SpectrumData:
    """Container for spectrum measurement data.

    This class holds all data associated with a single spectrum capture,
    including intensity (0-1), calibrated wavelengths, and detected peaks.
    Optional gain/exposure allow standardized intensity (comparable across settings).

    Attributes:
        intensity: Array of intensity values (0-1) for each pixel
        wavelengths: Array of wavelength values (nm) for each pixel
        timestamp: Unix timestamp of when the spectrum was captured
        peaks: List of detected peaks in the spectrum
        raw_frame: Optional raw camera frame for display purposes
        cropped_frame: Optional cropped region used for spectrum extraction
        gain: Camera gain at capture (None if unknown or N/A)
        exposure_us: Camera exposure in µs at capture (None if unknown or N/A)
    """

    intensity: np.ndarray
    wavelengths: np.ndarray
    timestamp: float = field(default_factory=time.time)
    peaks: list[Peak] = field(default_factory=list)
    raw_frame: np.ndarray | None = field(default=None, repr=False)
    cropped_frame: np.ndarray | None = field(default=None, repr=False)
    x_axis_label: str = "Wavelength (nm)"
    gain: float | None = field(default=None, repr=False)
    exposure_us: int | None = field(default=None, repr=False)

    @property
    def width(self) -> int:
        """Number of pixels/data points in the spectrum."""
        return len(self.intensity)

    @property
    def min_wavelength(self) -> float:
        """Minimum wavelength in the spectrum."""
        return float(self.wavelengths[0])

    @property
    def max_wavelength(self) -> float:
        """Maximum wavelength in the spectrum."""
        return float(self.wavelengths[-1])

    @property
    def max_intensity(self) -> float:
        """Maximum intensity value in the spectrum (0-1)."""
        return float(np.max(self.intensity))

    def wavelength_at(self, pixel: int) -> float:
        """Get the wavelength at a specific pixel index."""
        if 0 <= pixel < len(self.wavelengths):
            return float(self.wavelengths[pixel])
        raise IndexError(f"Pixel index {pixel} out of range [0, {len(self.wavelengths)})")

    def intensity_at(self, pixel: int) -> int:
        """Get the intensity at a specific pixel index."""
        if 0 <= pixel < len(self.intensity):
            return int(self.intensity[pixel])
        raise IndexError(f"Pixel index {pixel} out of range [0, {len(self.intensity)})")

    def standardized_intensity(
        self,
        reference_gain: float = 1.0,
        reference_exposure_us: float = 1.0,
    ) -> np.ndarray | None:
        """Intensity scaled so the same irradiance gives the same value across gain/exposure.

        Returns intensity * (reference_gain * reference_exposure_us) / (gain * exposure_us),
        or None if gain or exposure_us is missing. Use for export/comparison across settings.
        """
        if self.gain is None or self.exposure_us is None:
            return None
        g, e = float(self.gain), float(self.exposure_us)
        if g < 1e-12 or e < 1e-12:
            return None
        scale = (reference_gain * reference_exposure_us) / (g * e)
        return (self.intensity.astype(np.float64) * scale).astype(np.float32)

    def with_intensity(self, intensity: np.ndarray) -> "SpectrumData":
        """Return a new SpectrumData with updated intensity values."""
        return SpectrumData(
            intensity=intensity,
            wavelengths=self.wavelengths,
            timestamp=self.timestamp,
            peaks=self.peaks,
            raw_frame=self.raw_frame,
            cropped_frame=self.cropped_frame,
            x_axis_label=self.x_axis_label,
            gain=self.gain,
            exposure_us=self.exposure_us,
        )

    def with_peaks(self, peaks: list[Peak]) -> "SpectrumData":
        """Return a new SpectrumData with updated peaks."""
        return SpectrumData(
            intensity=self.intensity,
            wavelengths=self.wavelengths,
            timestamp=self.timestamp,
            peaks=peaks,
            raw_frame=self.raw_frame,
            cropped_frame=self.cropped_frame,
            x_axis_label=self.x_axis_label,
            gain=self.gain,
            exposure_us=self.exposure_us,
        )

    def to_csv_rows(self) -> list[tuple[float, float]]:
        """Convert spectrum data to CSV-compatible rows."""
        return list(zip(self.wavelengths, self.intensity))
