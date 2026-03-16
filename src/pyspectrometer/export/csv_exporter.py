"""CSV export for spectrum data."""

import csv
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..core.spectrum import Peak, SpectrumData
from ..processing.peak_detection import peak_widths_nm
from .base import ExporterInterface


def build_markers_peaks_metadata(
    marker_lines: list[int],
    wavelengths: np.ndarray,
    intensity: np.ndarray,
    peaks: list[Peak],
) -> dict[str, Any]:
    """Build metadata dict entries for markers and peaks (with deltas and intensities).

    Markers: wavelength and intensity per marker; Markers_dL_nm and Markers_dI when >=2.
    Peaks: wavelength, intensity, and delta (FWHM in nm) per peak.
    """
    out: dict[str, Any] = {}
    n = len(wavelengths)
    if n == 0:
        return out

    if marker_lines:
        parts = []
        for idx in marker_lines[:10]:
            i = max(0, min(int(idx), n - 1))
            wl = float(wavelengths[i])
            I = float(intensity[i]) if i < len(intensity) else 0.0
            parts.append(f"{wl:.2f}nm {I:.4f}")
        out["Markers"] = "; ".join(parts)
        if len(marker_lines) >= 2:
            i1, i2 = sorted(marker_lines[:2])
            i1 = max(0, min(i1, n - 1))
            i2 = max(0, min(i2, n - 1))
            out["Markers_dL_nm"] = f"{abs(float(wavelengths[i2]) - float(wavelengths[i1])):.2f}"
            out["Markers_dI"] = f"{abs(float(intensity[i2]) - float(intensity[i1])):.4f}"

    if peaks:
        indices = [p.index for p in peaks]
        widths = peak_widths_nm(intensity, wavelengths, indices, rel_height=0.5)
        parts = []
        for p, w_nm in zip(peaks, widths):
            parts.append(f"{p.wavelength:.2f}nm I={p.intensity:.4f} d={w_nm:.3f}nm")
        out["Peaks"] = "; ".join(parts)

    return out


def build_absorption_metadata(
    measured: np.ndarray,
    dark: np.ndarray | None,
    white: np.ndarray,
) -> dict[str, Any]:
    """Build metadata for absorbance and relative power density absorbed (when white ref set).

    Transmission T = (measured - dark) / (white - dark). Absorbance A = -log10(T).
    Fraction absorbed (relative power density absorbed) = 1 - T.
    """
    out: dict[str, Any] = {}
    n = min(len(measured), len(white))
    if n == 0:
        return out
    measured = measured[:n].astype(np.float64)
    white = np.asarray(white[:n], dtype=np.float64)
    if dark is not None:
        dark = np.asarray(dark[:n], dtype=np.float64)
        white = white - dark
        measured = measured - dark
    white = np.maximum(white, 1e-10)
    transmission = np.clip(measured / white, 1e-10, 1.0)
    absorbance = -np.log10(transmission)
    fraction_absorbed = 1.0 - transmission
    out["Absorbance_mean"] = f"{float(np.mean(absorbance)):.4f}"
    out["Absorbance_max"] = f"{float(np.max(absorbance)):.4f}"
    out["Fraction_absorbed_mean"] = f"{float(np.mean(fraction_absorbed)):.4f}"
    out["Fraction_absorbed_max"] = f"{float(np.max(fraction_absorbed)):.4f}"
    return out


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
        measured_raw_intensity: "np.ndarray | None" = None,
    ) -> Path:
        """Export spectrum data to CSV file.

        All SPD columns (Measured, Dark, White, extra) are written as raw values:
        no normalization and no dark subtraction.

        Args:
            data: Spectrum data (wavelengths; intensity used only if measured_raw_intensity omitted)
            path: Output file path (auto-generated if None)
            reference_intensity: Optional reference illuminant for calibration CSV
            dark_intensity: Optional dark reference SPD (raw)
            white_intensity: Optional white/reference SPD (raw)
            metadata: Optional dict of key-value pairs for header comments
            measured_raw_intensity: Raw measured SPD for Measured/Intensity column (default: data.intensity)

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
                measured_raw_intensity=measured_raw_intensity,
            )

        return self._export_standard(data, path, metadata, measured_raw_intensity=measured_raw_intensity)

    def _export_standard(
        self,
        data: SpectrumData,
        path: Path,
        metadata: "dict[str, Any] | None",
        measured_raw_intensity: "np.ndarray | None" = None,
    ) -> Path:
        """Export standard format: Pixel, Wavelength/Wavenumber, Intensity (raw SPD)."""
        x_label = getattr(data, "x_axis_label", "Wavelength (nm)")
        col_name = "Wavenumber" if "cm" in x_label else "Wavelength"
        meta = metadata.copy() if metadata else {}
        meta.setdefault("Mode", "Measurement")
        meta.setdefault("Date", time.strftime("%Y-%m-%d %H:%M:%S"))
        intensity = measured_raw_intensity if measured_raw_intensity is not None else data.intensity
        n = min(len(data.wavelengths), len(intensity))
        wl = data.wavelengths[:n]
        intensity = _align(intensity, n)

        with open(path, "w", newline="\n", encoding="utf-8") as f:
            _write_header_comments(f, meta)
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["Pixel", col_name, "Intensity"])
            for i in range(n):
                writer.writerow([i, float(wl[i]), float(intensity[i])])

        return path

    def _export_with_references(
        self,
        data: SpectrumData,
        path: Path,
        dark_intensity: np.ndarray | None,
        white_intensity: np.ndarray | None,
        metadata: dict[str, Any] | None,
        extra_columns: list[tuple[str, np.ndarray]] | None = None,
        measured_raw_intensity: np.ndarray | None = None,
    ) -> Path:
        """Export with Measured, Dark, White SPD columns (all raw), plus optional extras.

        Measured column uses measured_raw_intensity when provided; otherwise data.intensity.
        Extra columns are interpolated to the main wavelength grid.
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
        measured = _align(
            measured_raw_intensity if measured_raw_intensity is not None else data.intensity,
            n,
        )
        dark = _align(dark_intensity, n)
        white = _align(white_intensity, n)
        extra_aligned = []
        for ec_name, col_val in extra_cols:
            if isinstance(col_val, tuple):
                src_wl_col, src_vals = col_val
            else:
                src_wl_col, src_vals = wl, col_val
            extra_aligned.append((ec_name, _interpolate_to(src_vals, src_wl_col, wl)))

        with open(path, "w", newline="\n", encoding="utf-8") as f:
            _write_header_comments(f, meta)
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(cols)
            for i in range(n):
                row = [i, float(wl[i]), float(measured[i])]
                if dark is not None:
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

    def export_waterfall(
        self,
        rows: list[np.ndarray],
        wavelengths: np.ndarray,
        path: Path,
        *,
        dark_intensity: np.ndarray | None = None,
        white_intensity: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Export waterfall: one row per measured SPD; dark/white in header comments.

        rows: list of raw measured intensity arrays (one per time step).
        Each row is aligned to wavelength length; rows are written in order (row 0 = earliest).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = metadata.copy() if metadata else {}
        meta.setdefault("Mode", "Waterfall")
        meta.setdefault("Date", time.strftime("%Y-%m-%d %H:%M:%S"))
        n_wl = len(wavelengths)

        with open(path, "w", newline="\n", encoding="utf-8") as f:
            _write_header_comments(f, meta)
            if dark_intensity is not None and len(dark_intensity) > 0:
                dark_aligned = _align(dark_intensity, n_wl)
                f.write("# Dark: " + ",".join(str(float(x)) for x in dark_aligned) + "\n")
            if white_intensity is not None and len(white_intensity) > 0:
                white_aligned = _align(white_intensity, n_wl)
                f.write("# White: " + ",".join(str(float(x)) for x in white_aligned) + "\n")
            writer = csv.writer(f, lineterminator="\n")
            header = ["Row"] + [f"{float(w):.2f}" for w in wavelengths[:n_wl]]
            writer.writerow(header)
            for row_idx, intensity in enumerate(rows):
                aligned = _align(intensity, n_wl)
                writer.writerow([row_idx] + [float(x) for x in aligned])

        return path

    def start_waterfall_rec(
        self,
        path: Path,
        wavelengths: np.ndarray,
        *,
        dark_intensity: np.ndarray | None = None,
        white_intensity: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "WaterfallRecWriter":
        """Open a CSV for continuous waterfall recording; dark/white in comments. Caller must close."""
        path.parent.mkdir(parents=True, exist_ok=True)
        return WaterfallRecWriter(
            path, wavelengths,
            dark_intensity=dark_intensity,
            white_intensity=white_intensity,
            metadata=metadata,
        )

    def generate_filename(self, prefix: str = "Spectrum") -> Path:
        """Generate a timestamped filename."""
        timestamp = time.strftime(self._timestamp_format)
        return self._output_dir / f"{prefix}-{timestamp}{self.extension}"


class WaterfallRecWriter:
    """Appends one row per write_row() for indefinite waterfall recording."""

    def __init__(
        self,
        path: Path,
        wavelengths: np.ndarray,
        *,
        dark_intensity: np.ndarray | None = None,
        white_intensity: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self._path = path
        self._n_wl = len(wavelengths)
        self._wl = wavelengths
        self._file = open(path, "w", newline="\n", encoding="utf-8")
        meta = metadata.copy() if metadata else {}
        meta.setdefault("Mode", "Waterfall-Rec")
        meta.setdefault("Date", time.strftime("%Y-%m-%d %H:%M:%S"))
        _write_header_comments(self._file, meta)
        if dark_intensity is not None and len(dark_intensity) > 0:
            dark_aligned = _align(dark_intensity, self._n_wl)
            self._file.write("# Dark: " + ",".join(str(float(x)) for x in dark_aligned) + "\n")
        if white_intensity is not None and len(white_intensity) > 0:
            white_aligned = _align(white_intensity, self._n_wl)
            self._file.write("# White: " + ",".join(str(float(x)) for x in white_aligned) + "\n")
        self._writer = csv.writer(self._file, lineterminator="\n")
        header = ["Row"] + [f"{float(w):.2f}" for w in self._wl[: self._n_wl]]
        self._writer.writerow(header)
        self._row_count = 0

    def write_row(self, intensity: np.ndarray) -> None:
        """Append one measured SPD row (raw)."""
        aligned = _align(intensity, self._n_wl)
        self._writer.writerow([self._row_count] + [float(x) for x in aligned])
        self._row_count += 1

    def close(self) -> None:
        """Close the file."""
        if self._file is not None:
            self._file.close()
            self._file = None
