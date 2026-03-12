"""Load spectrum data from CSV. Calibrator does not handle I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_spectrum_csv(csv_path: str | Path) -> tuple[np.ndarray, int, np.ndarray]:
    """Load intensity and wavelengths from spectrum CSV.

    Handles Pixel,Wavelength,Intensity and Pixel,Intensity,reference_wavelength formats.
    Returns (intensity, n_pixels, wavelengths). Wavelengths default to 380–750 nm if missing.

    Returns:
        (intensity, n_pixels, wavelengths) or (empty array, 0, empty array) on failure
    """
    path = Path(csv_path)
    if not path.exists():
        return np.array([]), 0, np.array([])

    pixels_list: list[int] = []
    intensity_list: list[float] = []
    wl_list: list[float] = []
    intensity_col = 1
    wl_col = 2

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if line.lower().startswith("pixel"):
                if "reference_wavelength" in line.lower():
                    wl_col = 2
                elif "calibrated_wavelength" in line.lower():
                    wl_col = 4
                continue
            if len(parts) >= max(2, wl_col + 1):
                try:
                    px = int(parts[0])
                    intensity = float(parts[intensity_col])
                    wl = float(parts[wl_col]) if wl_col < len(parts) else 0.0
                    pixels_list.append(px)
                    intensity_list.append(intensity)
                    wl_list.append(wl)
                except (ValueError, IndexError):
                    continue

    if not pixels_list:
        return np.array([]), 0, np.array([])

    pairs = sorted(zip(pixels_list, intensity_list, wl_list), key=lambda x: x[0])
    n = pairs[-1][0] + 1
    measured = np.zeros(n, dtype=np.float64)
    wavelengths = np.zeros(n, dtype=np.float64)
    for px, intensity, wl in pairs:
        measured[px] = intensity
        wavelengths[px] = wl
    if not np.any(wavelengths > 0):
        wavelengths = np.linspace(380, 750, n)
    return measured, n, wavelengths
