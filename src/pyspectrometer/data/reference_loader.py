"""Load reference spectra from data/reference folder.

Scans data/reference/*.csv and loads spectra. Falls back to colour-science
for sources not found in files. Supports adding custom references via UI later.
"""

from pathlib import Path

import numpy as np

from .reference_spectra import ReferenceSource


def _reference_dir() -> Path:
    """Return path to data/reference. Tries CWD first (project root)."""
    for base in [Path.cwd(), Path(__file__).resolve().parents[2]]:
        ref = base / "data" / "reference"
        if ref.is_dir():
            return ref
    return Path.cwd() / "data" / "reference"


# File name patterns and column mapping: (filename_pattern, column_index or None for single-col)
# Column index 0 = wavelength, 1 = first data column
_FILE_SOURCE_MAP: list[tuple[str, ReferenceSource, int | None]] = [
    ("CIE_D65", ReferenceSource.D65, None),
    ("CIE_illum_FLs", ReferenceSource.FL1, 1),
    ("CIE_illum_FLs", ReferenceSource.FL2, 2),
    ("CIE_illum_FLs", ReferenceSource.FL3, 3),
    ("CIE_illum_FLs", ReferenceSource.FL12, 12),
]


def _load_csv_spectrum(path: Path, col_idx: int | None = None) -> tuple[np.ndarray, np.ndarray] | None:
    """Load wavelength and intensity from CSV. Returns (wavelengths, values) or None."""
    if not path.exists():
        return None
    wl_list: list[float] = []
    val_list: list[float] = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                w = float(parts[0])
            except ValueError:
                continue  # Skip header row
            try:
                if col_idx is not None:
                    if col_idx >= len(parts):
                        continue
                    v = float(parts[col_idx])
                else:
                    v = float(parts[1])
                wl_list.append(w)
                val_list.append(v)
            except (ValueError, IndexError):
                continue

    if len(wl_list) < 2:
        return None
    return (np.array(wl_list, dtype=np.float64), np.array(val_list, dtype=np.float64))


def _interpolate_to_wavelengths(
    wl_src: np.ndarray, val_src: np.ndarray, wl_target: np.ndarray
) -> np.ndarray:
    """Interpolate (wl_src, val_src) to wl_target. Extrapolates with edge values."""
    out = np.interp(wl_target, wl_src, val_src)
    if out.max() > 0:
        out = out / out.max()
    return out.astype(np.float64)


# Cache: source -> (wl_src, val_src) for interpolation
_spectrum_cache: dict[ReferenceSource, tuple[np.ndarray, np.ndarray] | None] = {}


def _load_from_files(source: ReferenceSource) -> tuple[np.ndarray, np.ndarray] | None:
    """Load spectrum for source from reference folder. Returns (wl, val) or None."""
    if source in _spectrum_cache:
        return _spectrum_cache[source]

    ref_dir = _reference_dir()

    for pattern, src, col_idx in _FILE_SOURCE_MAP:
        if src != source:
            continue
        for p in ref_dir.glob("*.csv"):
            if pattern in p.name:
                data = _load_csv_spectrum(p, col_idx)
                if data is not None:
                    _spectrum_cache[source] = data
                    return data

    _spectrum_cache[source] = None
    return None


def get_reference_spectrum_from_files(
    source: ReferenceSource, wavelengths: np.ndarray
) -> np.ndarray | None:
    """Get reference spectrum from data/reference files, interpolated to wavelengths.

    Returns None if not found in files (caller should fall back to colour-science).
    """
    data = _load_from_files(source)
    if data is None:
        return None
    wl_src, val_src = data
    return _interpolate_to_wavelengths(wl_src, val_src, wavelengths)


def list_available_reference_files() -> list[tuple[str, str]]:
    """List CSV files in reference folder. Returns [(filename, display_name), ...]."""
    ref_dir = _reference_dir()
    if not ref_dir.is_dir():
        return []
    result: list[tuple[str, str]] = []
    for p in sorted(ref_dir.glob("*.csv")):
        name = p.stem.replace("_", " ").replace("-", " ")
        result.append((p.name, name))
    return result


def clear_spectrum_cache() -> None:
    """Clear loaded spectrum cache (e.g. after adding new files)."""
    _spectrum_cache.clear()
