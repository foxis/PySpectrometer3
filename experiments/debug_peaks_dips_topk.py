"""Draw peaks and dips together using top-k filtering per SPD.

Uses extract_extremums from processing (max_count=k).
Each subplot: one SPD (measured or reference) with top-k peaks (green) and dips (red).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
OUT_DIR = ROOT / ".tmp"
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt

from pyspectrometer.data.reference_spectra import ReferenceSource, get_reference_spectrum
from pyspectrometer.processing.peak_detection import extract_extremums


def load_csv(csv_path: Path) -> tuple[np.ndarray, int, np.ndarray, ReferenceSource | None]:
    """Load intensity and wavelengths from CSV. Returns (intensity, n, wavelengths, ref_source)."""
    pixels_list: list[int] = []
    intensity_list: list[float] = []
    wl_list: list[float] = []
    intensity_col = 1
    wl_col = 2
    ref_source: ReferenceSource | None = None

    with open(csv_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                if "Reference source:" in line:
                    m = re.search(r"Reference source:\s*(\w+)", line, re.I)
                    if m:
                        try:
                            ref_source = ReferenceSource[m.group(1).upper()]
                        except KeyError:
                            pass
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
        return np.array([]), 0, np.array([]), ref_source

    pairs = sorted(zip(pixels_list, intensity_list, wl_list), key=lambda x: x[0])
    n = pairs[-1][0] + 1
    measured = np.zeros(n, dtype=np.float64)
    wavelengths = np.zeros(n, dtype=np.float64)
    for px, intensity, wl in pairs:
        measured[px] = intensity
        wavelengths[px] = wl
    if not np.any(wavelengths > 0):
        wavelengths = np.linspace(380, 750, n)
    return measured, n, wavelengths, ref_source


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Draw peaks and dips together with top-k filtering per SPD"
    )
    parser.add_argument(
        "csv_paths",
        nargs="*",
        default=[str(p) for p in sorted(OUTPUT_DIR.glob("*.csv"))],
    )
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=20,
        help="Top-k extremums per SPD (max_count). Default: 20",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = [Path(p) for p in args.csv_paths if Path(p).exists()]
    if not csv_paths:
        print("No CSV files found")
        return 1

    k = args.top_k

    datasets: list[tuple[str, np.ndarray, np.ndarray, list, list, ReferenceSource | None]] = []
    for csv_path in csv_paths:
        measured, n, wavelengths, ref_source = load_csv(csv_path)
        if n < 10:
            continue
        wl = wavelengths if np.any(wavelengths > 0) else np.linspace(380, 750, n)
        meas_norm = measured.copy()
        if meas_norm.max() > 0:
            meas_norm = meas_norm / meas_norm.max()

        extremums = extract_extremums(
            meas_norm,
            wl,
            position_px=np.arange(n, dtype=np.intp),
            max_count=k,
        )
        peaks = [e for e in extremums if not e.is_dip]
        dips = [e for e in extremums if e.is_dip]
        datasets.append((csv_path.stem, wl, meas_norm, peaks, dips, ref_source))

    if not datasets:
        print("No valid datasets")
        return 1

    # Reference SPD (use first CSV's wavelengths and source)
    ref_source = datasets[0][5] or ReferenceSource.FL12
    wl_ref = datasets[0][1]
    ref_intensity = get_reference_spectrum(ref_source, wl_ref)
    ref_norm = np.asarray(ref_intensity, dtype=np.float64)
    if ref_norm.max() > 0:
        ref_norm = ref_norm / ref_norm.max()
    ref_extremums = extract_extremums(
        ref_norm,
        wl_ref,
        position_px=np.arange(len(ref_norm), dtype=np.intp),
        max_count=k,
    )
    ref_peaks = [e for e in ref_extremums if not e.is_dip]
    ref_dips = [e for e in ref_extremums if e.is_dip]

    # Layout: each measured CSV + one reference. 2 rows per measured (meas, ref) or 1 row per?
    # Simpler: n_plots = len(datasets) + 1 (all measured + reference)
    n_plots = len(datasets) + 1
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    def plot_spd(
        ax,
        stem: str,
        wl: np.ndarray,
        intensity: np.ndarray,
        peaks: list,
        dips: list,
        color: str = "b",
    ) -> None:
        mask = (wl >= 350) & (wl <= 750)
        ax.plot(wl[mask], intensity[mask], "-", color=color, alpha=0.7, linewidth=1)
        for e in peaks:
            if not (350 <= e.position <= 750):
                continue
            y = float(intensity[e.index]) if 0 <= e.index < len(intensity) else 0
            ax.scatter([e.position], [y], c="green", s=50, zorder=5, edgecolors="black")
        for e in dips:
            if not (350 <= e.position <= 750):
                continue
            y = float(intensity[e.index]) if 0 <= e.index < len(intensity) else 0
            ax.scatter([e.position], [y], c="red", s=50, zorder=5, edgecolors="black", marker="v")
        ax.set_ylabel("Intensity")
        n_peaks, n_dips = len(peaks), len(dips)
        ax.set_title(f"{stem} — top-k: {n_peaks} peaks, {n_dips} dips")
        ax.grid(True, alpha=0.3)

    for ax, (stem, wl, intensity, peaks, dips, _) in zip(axes, datasets):
        plot_spd(ax, stem, wl, intensity, peaks, dips, color="b")

    # Reference subplot
    plot_spd(
        axes[-1],
        f"Reference ({ref_source.name})",
        wl_ref,
        ref_norm,
        ref_peaks,
        ref_dips,
        color="orange",
    )

    axes[-1].set_xlabel("Wavelength (nm)")
    fig.suptitle(
        f"Peaks (green) and dips (red) with top-k={k} per SPD",
        fontsize=11,
        y=1.01,
    )
    plt.tight_layout()
    out_path = out_dir / "peaks_dips_topk.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
