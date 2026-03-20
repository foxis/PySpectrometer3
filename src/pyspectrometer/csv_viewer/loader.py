"""CSV type detection and parsing for the CSV viewer.

Supports the four formats produced by CSVExporter:
  SPECTRUM     — Pixel, Wavelength, Intensity [, Dark] [, White] [, extras] [, Sensitivity]
  WATERFALL    — Row, <wl_0>, <wl_1>, ... (wavelengths as column headers)
  COLORSCIENCE — SPECTRUM layout + # Swatch_N: comments
  CALIBRATION  — pixel, intensity, reference_wavelength, ...
"""

import csv
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

import numpy as np

_WATERFALL_MODES = {"waterfall", "waterfall-rec"}
_CALIBRATION_MODES = {"calibration"}
_COLORSCIENCE_MODES = {"color science", "colorscience"}

_CALIBRATION_COLS = {"pixel", "intensity", "reference_wavelength"}


class CsvType(Enum):
    SPECTRUM = auto()
    WATERFALL = auto()
    COLORSCIENCE = auto()
    CALIBRATION = auto()


@dataclass
class LoadedCsv:
    """Parsed representation of an exported spectrum CSV."""

    csv_type: CsvType
    intensity: np.ndarray
    wavelengths: np.ndarray
    dark: np.ndarray | None = None
    white: np.ndarray | None = None
    sensitivity: np.ndarray | None = None
    # Named extra columns (e.g. swatches): name → intensity array
    extra_columns: dict[str, np.ndarray] = field(default_factory=dict)
    # All comment-line key/value pairs
    metadata: dict[str, str] = field(default_factory=dict)


def load_csv(path: Path) -> LoadedCsv:
    """Parse a CSV file exported by CSVExporter and return a LoadedCsv.

    Detection uses the ``# Mode:`` comment line first; falls back to
    column-header fingerprinting when the comment is absent.
    """
    raw_comments, header, rows = _read_raw(path)
    metadata = _parse_comments(raw_comments)
    csv_type = _detect_type(metadata, header, raw_comments)

    match csv_type:
        case CsvType.WATERFALL:
            return _load_waterfall(header, rows, metadata)
        case CsvType.CALIBRATION:
            return _load_calibration(header, rows, metadata)
        case CsvType.SPECTRUM | CsvType.COLORSCIENCE:
            return _load_spectrum(header, rows, metadata, csv_type)


# ---------------------------------------------------------------------------
# Low-level parsing helpers
# ---------------------------------------------------------------------------


def _read_raw(
    path: Path,
) -> tuple[list[str], list[str], list[list[str]]]:
    """Read the file once; split comment lines, header row, and data rows."""
    comments: list[str] = []
    header: list[str] = []
    rows: list[list[str]] = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            first = row[0].strip()
            if first.startswith("#"):
                # Rejoin: csv.reader splits comma-separated comment payloads (e.g. # Dark: 0.1,0.2,...)
                comments.append(",".join(row).strip())
            elif not header:
                header = [c.strip() for c in row]
            else:
                rows.append([c.strip() for c in row])

    return comments, header, rows


def _parse_comments(comment_lines: list[str]) -> dict[str, str]:
    """Extract ``# Key: value`` pairs from comment lines."""
    result: dict[str, str] = {}
    for line in comment_lines:
        # Strip leading "# " and split on first ":"
        body = line.lstrip("#").strip()
        if ":" in body:
            key, _, val = body.partition(":")
            result[key.strip()] = val.strip()
    return result


def _detect_type(
    metadata: dict[str, str],
    header: list[str],
    comments: list[str],
) -> CsvType:
    """Determine CSV type: Mode comment → column fingerprint fallback."""
    mode_val = metadata.get("Mode", "").lower().strip()

    if mode_val in _WATERFALL_MODES:
        return CsvType.WATERFALL
    if mode_val in _CALIBRATION_MODES:
        return CsvType.CALIBRATION
    if mode_val in _COLORSCIENCE_MODES:
        return CsvType.COLORSCIENCE
    if mode_val:
        # Any other non-empty Mode comment → treat as spectrum
        return CsvType.SPECTRUM

    # Fallback: column-header fingerprinting
    if not header:
        return CsvType.SPECTRUM

    lower = [h.lower() for h in header]
    first = lower[0] if lower else ""

    if first == "row" and _all_float_headers(header[1:]):
        return CsvType.WATERFALL

    if _CALIBRATION_COLS.issubset(set(lower)):
        return CsvType.CALIBRATION

    swatch_in_comments = any("swatch_" in c.lower() for c in comments)
    if swatch_in_comments:
        return CsvType.COLORSCIENCE

    return CsvType.SPECTRUM


def _all_float_headers(headers: list[str]) -> bool:
    """True if every header string can be parsed as a float (waterfall wavelength columns)."""
    if not headers:
        return False
    for h in headers:
        try:
            float(h)
        except ValueError:
            return False
    return True


# ---------------------------------------------------------------------------
# Format-specific loaders
# ---------------------------------------------------------------------------


def _load_spectrum(
    header: list[str],
    rows: list[list[str]],
    metadata: dict[str, str],
    csv_type: CsvType,
) -> LoadedCsv:
    """Load SPECTRUM or COLORSCIENCE CSV.

    Column layout:  Pixel, Wavelength, Intensity|Measured [, Dark] [, White]
                    [, <extra>...] [, Sensitivity]
    """
    lower = [h.lower() for h in header]

    # Required: wavelength column (index 1) and intensity column (index 2)
    if len(header) < 3:
        raise ValueError(f"Expected at least 3 columns, got {len(header)}: {header}")

    wl_col = 1  # always "Wavelength" or "Wavenumber"

    # Figure out what each column beyond Pixel/Wavelength contains
    known = {"pixel", "wavelength", "wavenumber"}
    intensity_names = {"intensity", "measured"}
    optional_names = {"dark", "white", "sensitivity"}

    # Map column index → role
    col_roles: dict[int, str] = {}
    extra_names: list[tuple[int, str]] = []

    for i, name in enumerate(lower):
        if i in {0, 1}:
            continue  # Pixel and Wavelength
        if name in intensity_names and "intensity" not in col_roles.values() and "measured" not in col_roles.values():
            col_roles[i] = name
        elif name in optional_names:
            col_roles[i] = name
        elif name not in known:
            extra_names.append((i, header[i]))

    n = len(rows)
    wavelengths = np.zeros(n, dtype=np.float64)
    intensity = np.zeros(n, dtype=np.float32)
    dark_arr: list[float] | None = None
    white_arr: list[float] | None = None
    sens_arr: list[float] | None = None
    extras: dict[str, list[float]] = {name: [] for _, name in extra_names}

    # Allocate optional arrays lazily
    has_dark = any(r == "dark" for r in col_roles.values())
    has_white = any(r == "white" for r in col_roles.values())
    has_sens = any(r == "sensitivity" for r in col_roles.values())
    if has_dark:
        dark_arr = []
    if has_white:
        white_arr = []
    if has_sens:
        sens_arr = []

    # Intensity column index
    int_col = next(
        (i for i, r in col_roles.items() if r in intensity_names),
        2,  # default to column 2
    )

    def _cell(row: list[str], col: int) -> float:
        return float(row[col]) if col < len(row) else 0.0

    for row_idx, row in enumerate(rows):
        wavelengths[row_idx] = _cell(row, wl_col)
        intensity[row_idx] = _cell(row, int_col)

        for col_i, role in col_roles.items():
            if role == "dark" and dark_arr is not None:
                dark_arr.append(_cell(row, col_i))
            elif role == "white" and white_arr is not None:
                white_arr.append(_cell(row, col_i))
            elif role == "sensitivity" and sens_arr is not None:
                sens_arr.append(_cell(row, col_i))

        for col_i, name in extra_names:
            extras[name].append(_cell(row, col_i))

    return LoadedCsv(
        csv_type=csv_type,
        intensity=intensity,
        wavelengths=wavelengths,
        dark=np.array(dark_arr, dtype=np.float32) if dark_arr is not None else None,
        white=np.array(white_arr, dtype=np.float32) if white_arr is not None else None,
        sensitivity=np.array(sens_arr, dtype=np.float32) if sens_arr is not None else None,
        extra_columns={name: np.array(vals, dtype=np.float32) for name, vals in extras.items()},
        metadata=metadata,
    )


def _load_waterfall(
    header: list[str],
    rows: list[list[str]],
    metadata: dict[str, str],
) -> LoadedCsv:
    """Load WATERFALL CSV.

    Column layout: Row, <wl_0>, <wl_1>, ...
    Dark/White/Sensitivity are embedded in # comment lines.
    Returns the last row as the primary intensity (most recent spectrum).
    """
    wavelengths = np.array([float(h) for h in header[1:]], dtype=np.float64)
    n_wl = len(wavelengths)

    if not rows:
        return LoadedCsv(
            csv_type=CsvType.WATERFALL,
            intensity=np.zeros(n_wl, dtype=np.float32),
            wavelengths=wavelengths,
            metadata=metadata,
        )

    # Parse the row matrix; the last row is the primary spectrum
    last_row = rows[-1]
    intensity = np.array(
        [float(last_row[i + 1]) if i + 1 < len(last_row) else 0.0 for i in range(n_wl)],
        dtype=np.float32,
    )

    # Dark/White/Sensitivity were written as comma-separated # comment lines
    dark = _parse_array_comment(metadata, "Dark", n_wl)
    white = _parse_array_comment(metadata, "White", n_wl)
    sensitivity = _parse_array_comment(metadata, "Sensitivity", n_wl)

    return LoadedCsv(
        csv_type=CsvType.WATERFALL,
        intensity=intensity,
        wavelengths=wavelengths,
        dark=dark,
        white=white,
        sensitivity=sensitivity,
        metadata=metadata,
    )


def _load_calibration(
    header: list[str],
    rows: list[list[str]],
    metadata: dict[str, str],
) -> LoadedCsv:
    """Load CALIBRATION CSV.

    Column layout: pixel, intensity, reference_wavelength, reference_intensity,
                   calibrated_wavelength, calibrated_intensity [, sensitivity]
    Uses calibrated_wavelength + calibrated_intensity columns.
    """
    lower = [h.lower() for h in header]
    wl_col = _col_index(lower, "calibrated_wavelength", fallback=4)
    int_col = _col_index(lower, "calibrated_intensity", fallback=5)
    sens_col = _col_index(lower, "sensitivity", fallback=-1)

    n = len(rows)
    wavelengths = np.zeros(n, dtype=np.float64)
    intensity = np.zeros(n, dtype=np.float32)
    sens_arr: list[float] = []

    def _cell(row: list[str], col: int) -> float:
        return float(row[col]) if col < len(row) else 0.0

    for row_idx, row in enumerate(rows):
        wavelengths[row_idx] = _cell(row, wl_col)
        intensity[row_idx] = _cell(row, int_col)
        if sens_col >= 0:
            sens_arr.append(_cell(row, sens_col))

    return LoadedCsv(
        csv_type=CsvType.CALIBRATION,
        intensity=intensity,
        wavelengths=wavelengths,
        sensitivity=np.array(sens_arr, dtype=np.float32) if sens_arr else None,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _col_index(lower_headers: list[str], name: str, fallback: int = -1) -> int:
    try:
        return lower_headers.index(name)
    except ValueError:
        return fallback


def _parse_array_comment(
    metadata: dict[str, str],
    key: str,
    length: int,
) -> np.ndarray | None:
    """Parse a comma-separated float list stored in a metadata comment (e.g. Dark/White)."""
    val = metadata.get(key)
    if not val:
        return None
    try:
        arr = np.array([float(x) for x in val.split(",")], dtype=np.float32)
        if len(arr) == 0:
            return None
        # Pad or trim to match wavelength count
        if len(arr) < length:
            arr = np.pad(arr, (0, length - len(arr)))
        elif len(arr) > length:
            arr = arr[:length]
        return arr
    except ValueError:
        return None
