"""Reference spectrum loading and management."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ReferenceSpectrum:
    """A reference spectrum with wavelength and intensity data."""

    name: str
    wavelengths: np.ndarray
    intensity: np.ndarray
    source: str = ""

    def interpolate_to(self, target_wavelengths: np.ndarray) -> np.ndarray:
        """Interpolate spectrum to match target wavelengths.

        Args:
            target_wavelengths: Target wavelength array to interpolate to

        Returns:
            Interpolated intensity array matching target_wavelengths shape
        """
        return np.interp(
            target_wavelengths,
            self.wavelengths,
            self.intensity,
            left=0.0,
            right=0.0,
        )

    def normalize_to_area(self, target_area: float) -> np.ndarray:
        """Return intensity normalized to match a target area.

        Args:
            target_area: Total area to normalize to

        Returns:
            Normalized intensity array
        """
        current_area = np.trapezoid(self.intensity, self.wavelengths)
        if current_area == 0:
            return self.intensity.copy()
        scale = target_area / current_area
        return self.intensity * scale


class ReferenceSpectrumLoader:
    """Loads reference spectrums from CSV data files."""

    def __init__(self, data_dir: Path | None = None):
        """Initialize loader with data directory.

        Args:
            data_dir: Directory containing spectrum CSV files.
                     If None, uses package data directory.
        """
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent.parent / "data"
        self.data_dir = Path(data_dir)

    def load_blackbody(self) -> ReferenceSpectrum | None:
        """Load blackbody 6500K spectrum."""
        path = self.data_dir / "blackbody_6500K.csv"
        return self._load_two_column_csv(
            path, "Blackbody 6500K", "Theoretical blackbody radiator at 6500K"
        )

    def load_solar(self) -> ReferenceSpectrum | None:
        """Load ASTM G173 solar spectrum (global irradiance)."""
        path = self.data_dir / "ASTM_G173_solar_spectrum.csv"
        if not path.exists():
            return None

        try:
            wavelengths = []
            intensity = []

            lines = self._read_file_lines(path)

            for line in lines[2:]:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) >= 3:
                    try:
                        wl = float(parts[0])
                        # Use global irradiance (column 2)
                        inten = float(parts[2])
                        wavelengths.append(wl)
                        intensity.append(inten)
                    except ValueError:
                        continue

            if not wavelengths:
                return None

            return ReferenceSpectrum(
                name="Solar (ASTM G173)",
                wavelengths=np.array(wavelengths, dtype=np.float64),
                intensity=np.array(intensity, dtype=np.float64),
                source="ASTM G173-03 Global Tilt Irradiance",
            )
        except Exception:
            return None

    def load_mercury_lamp(self) -> ReferenceSpectrum | None:
        """Load low pressure mercury lamp emission spectrum."""
        path = self.data_dir / "low_pressure_mercury_lamp.csv"
        if not path.exists():
            return None

        try:
            wavelengths = []
            intensity = []

            lines = self._read_file_lines(path)

            for line in lines:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("wavelength"):
                    continue

                parts = line.split(",")
                if len(parts) >= 2:
                    try:
                        wl = float(parts[0])
                        inten = float(parts[1])
                        wavelengths.append(wl)
                        intensity.append(inten)
                    except ValueError:
                        continue

            if not wavelengths:
                return None

            # Mercury lamp has discrete lines - expand to continuous spectrum
            # by adding zeros between lines
            expanded_wl, expanded_int = self._expand_line_spectrum(
                np.array(wavelengths),
                np.array(intensity),
                min_wl=380,
                max_wl=780,
                line_width=2.0,  # nm FWHM for each line
            )

            return ReferenceSpectrum(
                name="Mercury Lamp",
                wavelengths=expanded_wl,
                intensity=expanded_int,
                source="Low Pressure Mercury Lamp (Hg I lines)",
            )
        except Exception:
            return None

    def load_fluorescent(self) -> ReferenceSpectrum | None:
        """Load a representative fluorescent lamp spectrum (FL2 - Cool White)."""
        path = self.data_dir / "CIE_illum_FLs_1nm.csv"
        if not path.exists():
            return None

        try:
            wavelengths = []
            intensity = []

            lines = self._read_file_lines(path)

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(",")
                if len(parts) >= 3:
                    try:
                        wl = float(parts[0])
                        # FL2 is column index 2 (0-based: wavelength, FL1, FL2)
                        inten = float(parts[2])
                        wavelengths.append(wl)
                        intensity.append(inten)
                    except ValueError:
                        continue

            if not wavelengths:
                return None

            return ReferenceSpectrum(
                name="Fluorescent (FL2)",
                wavelengths=np.array(wavelengths, dtype=np.float64),
                intensity=np.array(intensity, dtype=np.float64),
                source="CIE FL2 Standard Illuminant - Cool White Fluorescent",
            )
        except Exception:
            return None

    def load_d65_daylight(self) -> ReferenceSpectrum | None:
        """Load CIE D65 standard daylight illuminant."""
        path = self.data_dir / "CIE_std_illum_D65.csv"
        return self._load_cie_illuminant(
            path,
            "D65 Daylight",
            "CIE Standard Illuminant D65 (6500K Daylight)",
        )

    def _read_file_lines(self, path: Path) -> list[str]:
        """Read file lines with encoding detection."""
        # Try UTF-16 first (check BOM)
        with open(path, "rb") as f:
            first_bytes = f.read(2)

        if first_bytes == b"\xff\xfe":
            encoding = "utf-16-le"
        elif first_bytes == b"\xfe\xff":
            encoding = "utf-16-be"
        else:
            encoding = "utf-8"

        with open(path, encoding=encoding) as f:
            return f.readlines()

    def _load_two_column_csv(self, path: Path, name: str, source: str) -> ReferenceSpectrum | None:
        """Load a simple two-column wavelength,intensity CSV."""
        if not path.exists():
            return None

        try:
            wavelengths = []
            intensity = []

            lines = self._read_file_lines(path)

            for line in lines:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "wavelength" in line.lower():
                    continue

                parts = line.split(",")
                if len(parts) >= 2:
                    try:
                        wl = float(parts[0])
                        inten = float(parts[1])
                        wavelengths.append(wl)
                        intensity.append(inten)
                    except ValueError:
                        continue

            if not wavelengths:
                return None

            return ReferenceSpectrum(
                name=name,
                wavelengths=np.array(wavelengths, dtype=np.float64),
                intensity=np.array(intensity, dtype=np.float64),
                source=source,
            )
        except Exception:
            return None

    def _load_cie_illuminant(self, path: Path, name: str, source: str) -> ReferenceSpectrum | None:
        """Load a CIE illuminant CSV (wavelength, value format with possible header)."""
        if not path.exists():
            return None

        try:
            wavelengths = []
            intensity = []

            lines = self._read_file_lines(path)

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(",")
                if len(parts) >= 2:
                    try:
                        wl = float(parts[0])
                        inten = float(parts[1])
                        wavelengths.append(wl)
                        intensity.append(inten)
                    except ValueError:
                        continue

            if not wavelengths:
                return None

            return ReferenceSpectrum(
                name=name,
                wavelengths=np.array(wavelengths, dtype=np.float64),
                intensity=np.array(intensity, dtype=np.float64),
                source=source,
            )
        except Exception:
            return None

    def _expand_line_spectrum(
        self,
        line_wavelengths: np.ndarray,
        line_intensities: np.ndarray,
        min_wl: float = 380,
        max_wl: float = 780,
        line_width: float = 2.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Expand discrete emission lines to a continuous spectrum.

        Creates Gaussian peaks at each line position.

        Args:
            line_wavelengths: Wavelengths of emission lines
            line_intensities: Relative intensities of lines
            min_wl: Minimum wavelength for output
            max_wl: Maximum wavelength for output
            line_width: FWHM of each line in nm

        Returns:
            Tuple of (wavelengths, intensities) arrays
        """
        # Create wavelength array at 1nm resolution
        wavelengths = np.arange(min_wl, max_wl + 1, 1.0)
        intensity = np.zeros_like(wavelengths)

        # Gaussian sigma from FWHM
        sigma = line_width / (2.0 * np.sqrt(2.0 * np.log(2.0)))

        for wl, amp in zip(line_wavelengths, line_intensities):
            if min_wl <= wl <= max_wl:
                # Add Gaussian peak
                gaussian = amp * np.exp(-0.5 * ((wavelengths - wl) / sigma) ** 2)
                intensity += gaussian

        return wavelengths, intensity


class ReferenceSpectrumManager:
    """Manages available reference spectrums and cycling between them."""

    def __init__(self, data_dir: Path | None = None):
        """Initialize manager with data directory.

        Args:
            data_dir: Directory containing spectrum CSV files
        """
        self._loader = ReferenceSpectrumLoader(data_dir)
        self._spectrums: list[ReferenceSpectrum] = []
        self._current_index: int = -1  # -1 means no reference shown

        self._load_all()

    def _load_all(self) -> None:
        """Load all available reference spectrums."""
        loaders = [
            self._loader.load_blackbody,
            self._loader.load_solar,
            self._loader.load_mercury_lamp,
            self._loader.load_fluorescent,
            self._loader.load_d65_daylight,
        ]

        for load_fn in loaders:
            spectrum = load_fn()
            if spectrum is not None:
                self._spectrums.append(spectrum)

        print(f"Loaded {len(self._spectrums)} reference spectrums:")
        for s in self._spectrums:
            print(f"  - {s.name}")

    @property
    def current(self) -> ReferenceSpectrum | None:
        """Get currently selected reference spectrum, or None if disabled."""
        if self._current_index < 0 or self._current_index >= len(self._spectrums):
            return None
        return self._spectrums[self._current_index]

    @property
    def current_name(self) -> str:
        """Get name of current reference spectrum."""
        current = self.current
        return current.name if current else "None"

    def cycle_next(self) -> str:
        """Cycle to the next reference spectrum.

        Returns:
            Name of the new current reference spectrum (or "None")
        """
        if not self._spectrums:
            return "None"

        # Cycle: -1 -> 0 -> 1 -> ... -> N-1 -> -1
        self._current_index += 1
        if self._current_index >= len(self._spectrums):
            self._current_index = -1

        return self.current_name

    def get_interpolated(
        self,
        target_wavelengths: np.ndarray,
        target_intensity: np.ndarray,
    ) -> np.ndarray | None:
        """Get current reference spectrum interpolated and normalized.

        Args:
            target_wavelengths: Wavelength array to interpolate to
            target_intensity: Measured intensity to normalize against

        Returns:
            Interpolated and normalized intensity array, or None if no reference
        """
        current = self.current
        if current is None:
            return None

        # Interpolate to target wavelengths
        interpolated = current.interpolate_to(target_wavelengths)

        # Normalize to match target area
        target_area = np.trapezoid(target_intensity, target_wavelengths)
        if target_area <= 0:
            return interpolated

        ref_area = np.trapezoid(interpolated, target_wavelengths)
        if ref_area <= 0:
            return interpolated

        scale = target_area / ref_area
        return interpolated * scale

    @property
    def available_names(self) -> list[str]:
        """Get list of available reference spectrum names."""
        return [s.name for s in self._spectrums]
