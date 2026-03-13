"""CSV export for spectrum data."""

import csv
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..core.spectrum import SpectrumData
from .base import ExporterInterface


def _align(arr: np.ndarray | None, n: int) -> np.ndarray | None:
    """Pad or trim arr to length n; return None if arr is None."""
    if arr is None:
        return None
    if len(arr) == n:
        return arr
    return arr[:n] if len(arr) > n else np.pad(arr, (0, n - len(arr)))


def _interpolate_to(values: np.ndarray, src_wl: np.ndarray, tgt_wl: np.ndarray) -> np.ndarray:
    """Resample values from src_wl grid to tgt_wl grid via linear interpolation."""
    if len(src_wl) == len(tgt_wl) and np.allclose(src_wl, tgt_wl, atol=0.1):
        return values
    return np.interp(tgt_wl, src_wl, values)


def _write_header_comments(f, metadata: dict[str, Any]) -> None:
    """Write metadata as # key: value comment lines. Extensible for any fields."""
    for key, value in metadata.items():
        if value is None or value == "":
            continue
        f.write(f"# {key}: {value}\n")


class CSVExporter(ExporterInterface):
    """Exports spectrum data to CSV format.

    Supports an abstract header comment section (metadata) before the data.
    Uses csv module for proper escaping and newlines.
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
        dark_intensity: "np.ndarray | None" = None,
        white_intensity: "np.ndarray | None" = None,
        metadata: "dict[str, Any] | None" = None,
        extra_columns: "list[tuple[str, np.ndarray]] | None" = None,
    ) -> Path:
        """Export spectrum data to CSV file.

        Args:
            data: Spectrum data to export
            path: Output file path (auto-generated if None)
            reference_intensity: Optional reference illuminant for calibration CSV
            dark_intensity: Optional dark reference SPD (measurement/color science)
            white_intensity: Optional white/reference SPD (omit for illuminance)
            metadata: Optional dict of key-value pairs for header comments

        Returns:
            Path to the created CSV file
        """
        if path is None:
            timestamp = time.strftime(self._timestamp_format)
            path = self._output_dir / f"Spectrum-{timestamp}.csv"

        path.parent.mkdir(parents=True, exist_ok=True)

        if reference_intensity is not None:
            return self._export_calibration(data, path, reference_intensity, metadata)

        if dark_intensity is not None or white_intensity is not None or extra_columns:
            return self._export_with_references(
                data, path, dark_intensity, white_intensity, metadata,
                extra_columns=extra_columns or [],
            )

        return self._export_standard(data, path, metadata)

    def _export_standard(
        self,
        data: SpectrumData,
        path: Path,
        metadata: "dict[str, Any] | None",
    ) -> Path:
        """Export standard format: Pixel, Wavelength/Wavenumber, Intensity."""
        x_label = getattr(data, "x_axis_label", "Wavelength (nm)")
        col_name = "Wavenumber" if "cm" in x_label else "Wavelength"
        meta = metadata.copy() if metadata else {}
        meta.setdefault("Mode", "Measurement")
        meta.setdefault("Date", time.strftime("%Y-%m-%d %H:%M:%S"))

        with open(path, "w", newline="\n", encoding="utf-8") as f:
            _write_header_comments(f, meta)
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["Pixel", col_name, "Intensity"])
            for pixel_idx, (x_val, intensity) in enumerate(data.to_csv_rows()):
                writer.writerow([pixel_idx, x_val, intensity])

        return path

    def _export_with_references(
        self,
        data: SpectrumData,
        path: Path,
        dark_intensity: np.ndarray | None,
        white_intensity: np.ndarray | None,
        metadata: dict[str, Any] | None,
        extra_columns: list[tuple[str, np.ndarray]] | None = None,
    ) -> Path:
        """Export with Measured, Dark, White/Reference SPD columns, plus optional extras.

        Extra columns are interpolated to the main wavelength grid so the CSV
        rows are all the same length.  Column names are taken from the tuple's
        first element.
        """
        x_label = getattr(data, "x_axis_label", "Wavelength (nm)")
        col_name = "Wavenumber" if "cm" in x_label else "Wavelength"
        meta = metadata.copy() if metadata else {}
        meta.setdefault("Mode", "Measurement")
        meta.setdefault("Date", time.strftime("%Y-%m-%d %H:%M:%S"))

        cols = ["Pixel", col_name, "Measured"]
        if dark_intensity is not None:
            cols.append("Dark")
        if white_intensity is not None:
            cols.append("White")
        extra_cols = extra_columns or []
        for name, _ in extra_cols:
            cols.append(name)

        n = len(data.intensity)
        wl = data.wavelengths

        dark  = _align(dark_intensity,  n)
        white = _align(white_intensity, n)
        extra_aligned = []
        for col_name, col_val in extra_cols:
            if isinstance(col_val, tuple):
                src_wl_col, src_vals = col_val
            else:
                src_wl_col, src_vals = wl, col_val
            extra_aligned.append((col_name, _interpolate_to(src_vals, src_wl_col, wl)))

        with open(path, "w", newline="\n", encoding="utf-8") as f:
            _write_header_comments(f, meta)
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(cols)
            for i in range(n):
                row = [i, float(wl[i]), float(data.intensity[i])]
                if dark  is not None:
                    row.append(float(dark[i]))
                if white is not None:
                    row.append(float(white[i]))
                for _, vals in extra_aligned:
                    row.append(float(vals[i]))
                writer.writerow(row)

        return path

    def _export_calibration(
        self,
        data: SpectrumData,
        path: Path,
        reference_intensity: np.ndarray,
        metadata: "dict[str, Any] | None",
    ) -> Path:
        """Export calibration format with reference columns."""
        n = min(
            len(data.intensity),
            len(data.wavelengths),
            len(reference_intensity),
        )
        meta = metadata.copy() if metadata else {}
        meta.setdefault("Mode", "Calibration")
        meta.setdefault("Date", time.strftime("%Y-%m-%d %H:%M:%S"))

        with open(path, "w", newline="\n", encoding="utf-8") as f:
            _write_header_comments(f, meta)
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow([
                "pixel",
                "intensity",
                "reference_wavelength",
                "reference_intensity",
                "calibrated_wavelength",
                "calibrated_intensity",
            ])
            for i in range(n):
                wl = float(data.wavelengths[i])
                ref_val = float(reference_intensity[i]) if i < len(reference_intensity) else 0.0
                intensity_val = float(data.intensity[i]) if i < len(data.intensity) else 0.0
                writer.writerow([i, intensity_val, wl, ref_val, wl, intensity_val])

        return path

    def generate_filename(self, prefix: str = "Spectrum") -> Path:
        """Generate a timestamped filename."""
        timestamp = time.strftime(self._timestamp_format)
        return self._output_dir / f"{prefix}-{timestamp}{self.extension}"
