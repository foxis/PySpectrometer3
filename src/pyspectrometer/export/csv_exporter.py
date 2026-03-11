"""CSV export for spectrum data."""

import time
from pathlib import Path

import numpy as np

from ..core.spectrum import SpectrumData
from .base import ExporterInterface


class CSVExporter(ExporterInterface):
    """Exports spectrum data to CSV format.

    The CSV file contains: Pixel (index), Wavelength (nm), Intensity (float32 0-1).
    """

    def __init__(
        self,
        output_dir: Path | str = None,
        timestamp_format: str = "%Y%m%d--%H%M%S",
    ):
        """Initialize CSV exporter.

        Args:
            output_dir: Directory for output files (defaults to "output" relative to CWD)
            timestamp_format: Format string for timestamps in filenames
        """
        self._output_dir = Path(output_dir) if output_dir else Path("output")
        self._timestamp_format = timestamp_format

    @property
    def name(self) -> str:
        return "CSV Exporter"

    @property
    def extension(self) -> str:
        return ".csv"

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    @output_dir.setter
    def output_dir(self, value: Path | str) -> None:
        self._output_dir = Path(value) if value else Path("output")

    def export(
        self,
        data: SpectrumData,
        path: Path = None,
        *,
        reference_intensity: "np.ndarray | None" = None,
    ) -> Path:
        """Export spectrum data to CSV file.

        In calibration mode (reference_intensity provided), writes special format:
        pixel, intensity, reference_wavelength, reference_intensity, calibrated_wavelength, calibrated_intensity

        Args:
            data: Spectrum data to export
            path: Output file path (auto-generated if None)
            reference_intensity: Optional reference illuminant spectrum for calibration CSV

        Returns:
            Path to the created CSV file
        """
        if path is None:
            timestamp = time.strftime(self._timestamp_format)
            path = self._output_dir / f"Spectrum-{timestamp}.csv"

        path.parent.mkdir(parents=True, exist_ok=True)

        if reference_intensity is not None:
            return self._export_calibration(data, path, reference_intensity)

        x_label = getattr(data, "x_axis_label", "Wavelength (nm)")
        col_name = "Wavenumber" if "cm" in x_label else "Wavelength"
        with open(path, "w") as f:
            f.write(f"Pixel,{col_name},Intensity\r\n")
            for pixel_idx, (x_val, intensity) in enumerate(data.to_csv_rows()):
                f.write(f"{pixel_idx},{x_val},{intensity}\r\n")

        return path

    def _export_calibration(
        self,
        data: SpectrumData,
        path: Path,
        reference_intensity: np.ndarray,
    ) -> Path:
        """Export calibration CSV: pixel, intensity, reference_wavelength, reference_intensity, calibrated_wavelength, calibrated_intensity."""
        n = min(
            len(data.intensity),
            len(data.wavelengths),
            len(reference_intensity),
        )
        with open(path, "w") as f:
            f.write(
                "pixel,intensity,reference_wavelength,reference_intensity,"
                "calibrated_wavelength,calibrated_intensity\r\n"
            )
            for i in range(n):
                wl = float(data.wavelengths[i])
                ref_val = float(reference_intensity[i]) if i < len(reference_intensity) else 0.0
                intensity_val = float(data.intensity[i]) if i < len(data.intensity) else 0.0
                f.write(f"{i},{intensity_val},{wl},{ref_val},{wl},{intensity_val}\r\n")
        return path

    def generate_filename(self, prefix: str = "Spectrum") -> Path:
        """Generate a timestamped filename.

        Args:
            prefix: Filename prefix

        Returns:
            Path with timestamped filename
        """
        timestamp = time.strftime(self._timestamp_format)
        return self._output_dir / f"{prefix}-{timestamp}{self.extension}"
