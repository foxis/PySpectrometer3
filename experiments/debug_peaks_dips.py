"""Debug peak/dip detection: all CSVs, all peaks and dips with height/width labels.

Output:
- peaks_all_csvs.png: All measured SPDs, ALL detected peaks with h=, w= labels
- dips_all_csvs.png: All measured SPDs, ALL detected dips with h=, w= labels
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
OUT_DIR = ROOT / ".tmp"
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt

from pyspectrometer.processing.peak_detection import debug_raw_peaks_and_dips


def load_csv(csv_path: Path) -> tuple[np.ndarray, int, np.ndarray]:
    """Load intensity and wavelengths from CSV."""
    pixels_list: list[int] = []
    intensity_list: list[float] = []
    wl_list: list[float] = []
    intensity_col = 1
    wl_col = 2

    with open(csv_path, encoding="utf-8") as f:
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug peaks/dips with height and width")
    parser.add_argument(
        "csv_paths",
        nargs="*",
        default=[str(p) for p in sorted(OUTPUT_DIR.glob("*.csv"))],
    )
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--no-valley-filter", action="store_true", help="Show all dips, not just valleys")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = [Path(p) for p in args.csv_paths if Path(p).exists()]
    if not csv_paths:
        print("No CSV files found")
        return 1

    datasets: list[tuple[str, np.ndarray, np.ndarray, list, list]] = []
    for csv_path in csv_paths:
        measured, n, wavelengths = load_csv(csv_path)
        if n < 10:
            continue
        wl = wavelengths if np.any(wavelengths > 0) else np.linspace(380, 750, n)
        peaks, dips = debug_raw_peaks_and_dips(
            measured,
            wl,
            peak_prominence=0.002,
            dip_prominence=0.01,
            no_valley_filter=True,
        )
        meas_norm = measured.copy()
        if meas_norm.max() > 0:
            meas_norm = meas_norm / meas_norm.max()
        datasets.append((csv_path.stem, wl, meas_norm, peaks, dips))

    if not datasets:
        print("No valid datasets")
        return 1

    n_plots = len(datasets)
    fig_peaks, axes_peaks = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots), sharex=True)
    if n_plots == 1:
        axes_peaks = [axes_peaks]

    for ax, (stem, wl, intensity, peaks, _) in zip(axes_peaks, datasets):
        mask = (wl >= 350) & (wl <= 750)
        ax.plot(wl[mask], intensity[mask], "b-", alpha=0.7, linewidth=1)
        for idx, pos, h, w in peaks:
            if not (350 <= pos <= 750):
                continue
            y = float(intensity[idx]) if 0 <= idx < len(intensity) else 0
            ax.scatter([pos], [y], c="green", s=40, zorder=5, edgecolors="black")
            ax.annotate(
                f"h={h:.3f}\nw={w:.1f}nm",
                (pos, y),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
                ha="left",
                va="bottom",
            )
        ax.set_ylabel("Intensity")
        ax.set_title(f"{stem} — {len(peaks)} peaks")
        ax.grid(True, alpha=0.3)

    axes_peaks[-1].set_xlabel("Wavelength (nm)")
    fig_peaks.suptitle("ALL detected peaks with height (h) and width (w)", fontsize=11, y=1.01)
    plt.tight_layout()
    out_peaks = out_dir / "peaks_all_csvs.png"
    fig_peaks.savefig(out_peaks, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_peaks}")

    fig_dips, axes_dips = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots), sharex=True)
    if n_plots == 1:
        axes_dips = [axes_dips]

    for ax, (stem, wl, intensity, _, dips) in zip(axes_dips, datasets):
        mask = (wl >= 350) & (wl <= 750)
        ax.plot(wl[mask], intensity[mask], "b-", alpha=0.7, linewidth=1)
        for idx, pos, h, w in dips:
            if not (350 <= pos <= 750):
                continue
            y = float(intensity[idx]) if 0 <= idx < len(intensity) else 0
            ax.scatter([pos], [y], c="red", s=40, zorder=5, edgecolors="black")
            ax.annotate(
                f"h={h:.3f}\nw={w:.1f}nm",
                (pos, y),
                textcoords="offset points",
                xytext=(5, -5),
                fontsize=7,
                ha="left",
                va="top",
            )
        ax.set_ylabel("Intensity")
        ax.set_title(f"{stem} — {len(dips)} dips")
        ax.grid(True, alpha=0.3)

    axes_dips[-1].set_xlabel("Wavelength (nm)")
    fig_dips.suptitle("ALL detected dips with height (h) and width (w)", fontsize=11, y=1.01)
    plt.tight_layout()
    out_dips = out_dir / "dips_all_csvs.png"
    fig_dips.savefig(out_dips, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_dips}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
