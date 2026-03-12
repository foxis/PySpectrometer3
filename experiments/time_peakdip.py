"""Timing harness for calibrate_spectrum_anchors (peak/dip SPD matcher)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pyspectrometer.data.reference_spectra import ReferenceSource, get_reference_spectrum
from pyspectrometer.processing.calibration.extremum import extract
from pyspectrometer.processing.calibration.hough_matching import calibrate_spectrum_anchors


def load_csv(path: Path) -> tuple[np.ndarray, int]:
    pixels: list[int] = []
    intensity: list[float] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split(",")
            if s.lower().startswith("pixel"):
                continue
            if len(parts) < 2:
                continue
            try:
                px = int(parts[0])
                it = float(parts[1])
            except ValueError:
                continue
            pixels.append(px)
            intensity.append(it)
    if not pixels:
        return np.array([]), 0
    pairs = sorted(zip(pixels, intensity), key=lambda x: x[0])
    n = pairs[-1][0] + 1
    arr = np.zeros(n, dtype=np.float64)
    for px, it in pairs:
        arr[px] = it
    return arr, n


def time_one(csv_rel: str, source: ReferenceSource, repeats: int = 10) -> None:
    csv_path = ROOT / csv_rel
    measured, n = load_csv(csv_path)
    if n == 0:
        print(f"{csv_rel}: no data")
        return

    wl_ref = np.linspace(380.0, 750.0, 500)
    ref_spd = get_reference_spectrum(source, wl_ref)
    ref_spd = ref_spd / ref_spd.max() if ref_spd.max() > 0 else ref_spd

    position_px = np.arange(n, dtype=np.intp)
    pos = np.linspace(380.0, 750.0, n)

    meas_exts = extract(measured, pos, position_px=position_px, max_count=40)
    meas_px = np.array([e.position_px for e in meas_exts if e.position_px is not None], dtype=np.float64)
    meas_int = np.array([abs(e.height) for e in meas_exts if e.position_px is not None], dtype=np.float64)
    meas_is_dip = np.array([e.is_dip for e in meas_exts if e.position_px is not None], dtype=bool)

    ref_exts = extract(ref_spd, wl_ref, position_px=None, max_count=40)
    ref_wl = np.array([e.position for e in ref_exts], dtype=np.float64)
    ref_int = np.array([abs(e.height) for e in ref_exts], dtype=np.float64)
    ref_is_dip = np.array([e.is_dip for e in ref_exts], dtype=bool)

    # Warm-up
    calibrate_spectrum_anchors(
        n,
        meas_pixels=meas_px,
        meas_is_dip=meas_is_dip,
        meas_intensities=meas_int,
        ref_feat_wl=ref_wl,
        ref_feat_is_dip=ref_is_dip,
        ref_feat_intensities=ref_int,
    )

    t0 = time.perf_counter()
    for _ in range(repeats):
        calibrate_spectrum_anchors(
            n,
            meas_pixels=meas_px,
            meas_is_dip=meas_is_dip,
            meas_intensities=meas_int,
            ref_feat_wl=ref_wl,
            ref_feat_is_dip=ref_is_dip,
            ref_feat_intensities=ref_int,
        )
    dt = (time.perf_counter() - t0) / repeats
    print(f"{csv_rel}: {dt*1000:.1f} ms per calibration (avg over {repeats})")


if __name__ == "__main__":
    from pyspectrometer.data.reference_spectra import ReferenceSource

    time_one("output/Spectrum-20260311--193856.csv", ReferenceSource.FL12)
    time_one("output/Spectrum-20260311--214345.csv", ReferenceSource.FL12)

