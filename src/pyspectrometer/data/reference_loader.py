"""Load reference spectra from configurable directories.

Searches ``ReferenceSearchPaths`` for CSV files. Falls back to colour-science for
sources not found in files. No mutable process-wide search path — use
:class:`ReferenceFileLoader` per application or test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .reference_paths import ReferenceSearchPaths
from .reference_spectra import ReferenceSource

# File name patterns and column mapping: (filename_pattern, column_index or None for single-col)
_FILE_SOURCE_MAP: list[tuple[str, ReferenceSource, int | None]] = [
    ("CIE_D65", ReferenceSource.D65, None),
    ("CIE_illum_FLs", ReferenceSource.FL1, 1),
    ("CIE_illum_FLs", ReferenceSource.FL2, 2),
    ("CIE_illum_FLs", ReferenceSource.FL3, 3),
    ("CIE_illum_FLs", ReferenceSource.FL12, 12),
]


def _detect_csv_columns(header_parts: list[str]) -> tuple[int, int] | None:
    """Detect wavelength and value column indices from header. Returns (wl_col, val_col) or None."""
    header_lower = [p.strip().lower() for p in header_parts]
    if "reference_wavelength" in header_lower and "reference_intensity" in header_lower:
        return (header_lower.index("reference_wavelength"), header_lower.index("reference_intensity"))
    for wl in ("wavelength_nm", "wavelength"):
        if wl in header_lower:
            wl_col = header_lower.index(wl)
            if wl_col + 1 < len(header_parts):
                return (wl_col, wl_col + 1)
    return None


def _load_csv_spectrum(
    path: Path,
    col_idx: int | None = None,
    wl_col: int | None = None,
    val_col: int | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load wavelength and intensity from CSV. Returns (wavelengths, values) or None."""
    if not path.exists():
        return None
    wl_list: list[float] = []
    val_list: list[float] = []
    header_cols: tuple[int, int] | None = None

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            if header_cols is None and wl_col is None and val_col is None and col_idx is None:
                detected = _detect_csv_columns(parts)
                if detected is not None:
                    header_cols = detected
            try:
                if header_cols is not None:
                    w, v = float(parts[header_cols[0]]), float(parts[header_cols[1]])
                elif col_idx is not None:
                    w = float(parts[0])
                    if col_idx >= len(parts):
                        continue
                    v = float(parts[col_idx])
                elif wl_col is not None and val_col is not None:
                    w, v = float(parts[wl_col]), float(parts[val_col])
                else:
                    w, v = float(parts[0]), float(parts[1])
                wl_list.append(w)
                val_list.append(v)
            except (ValueError, IndexError):
                if header_cols is None:
                    continue
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


class ReferenceFileLoader:
    """Loads reference SPD snippets from CSV under fixed search paths (instance-scoped cache)."""

    def __init__(self, paths: ReferenceSearchPaths) -> None:
        self._paths = paths
        self._spectrum_cache: dict[ReferenceSource, tuple[np.ndarray, np.ndarray] | None] = {}

    def resolved_dirs(self) -> list[Path]:
        return self._paths.resolved()

    def clear_cache(self) -> None:
        """Clear loaded spectrum cache (e.g. after adding new files)."""
        self._spectrum_cache.clear()

    def _load_from_files(self, source: ReferenceSource) -> tuple[np.ndarray, np.ndarray] | None:
        if source in self._spectrum_cache:
            return self._spectrum_cache[source]

        for ref_dir in self.resolved_dirs():
            if not ref_dir.is_dir():
                continue
            for pattern, src, col_idx in _FILE_SOURCE_MAP:
                if src != source:
                    continue
                for p in ref_dir.glob("*.csv"):
                    if pattern in p.name:
                        data = _load_csv_spectrum(p, col_idx)
                        if data is not None:
                            self._spectrum_cache[source] = data
                            return data

        self._spectrum_cache[source] = None
        return None

    def get_interpolated(self, source: ReferenceSource, wavelengths: np.ndarray) -> np.ndarray | None:
        """Return reference spectrum from CSV files, interpolated to *wavelengths*, or None."""
        data = self._load_from_files(source)
        if data is None:
            return None
        wl_src, val_src = data
        return _interpolate_to_wavelengths(wl_src, val_src, wavelengths)

    def list_csv_files(self) -> list[tuple[str, str]]:
        """List CSV files in all reference dirs. Returns [(filename, display_name), ...]."""
        seen: set[str] = set()
        result: list[tuple[str, str]] = []
        for ref_dir in self.resolved_dirs():
            if not ref_dir.is_dir():
                continue
            for p in sorted(ref_dir.glob("*.csv")):
                key = str(p.resolve())
                if key in seen:
                    continue
                seen.add(key)
                name = p.stem.replace("_", " ").replace("-", " ")
                result.append((p.name, name))
        return result
