"""Experiment: top 6 extremums → generator triplets → best match vs all ref triplets.

Takes top 6 extremums from measured, generates proper triplets via generator,
finds best match for each against all ref triplets. Displays scores and descriptors.
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

from pyspectrometer.data.reference_spectra import ReferenceSource, get_reference_spectrum
from pyspectrometer.processing.calibration.calibrate import _valid_positions
from pyspectrometer.processing.calibration.extremum import extract, from_known_lines
from pyspectrometer.processing.calibration.detect_peaks import get_reference_peaks
from pyspectrometer.processing.calibration.triplet import (
    DEFAULT_WEIGHTS,
    DESC_LABELS,
    Triplet,
    build as build_triplets,
    generate as generate_triplets,
    score as triplet_score,
)


def load_csv(csv_path: Path) -> tuple[np.ndarray, int, np.ndarray | None]:
    """Load intensity and wavelengths from CSV."""
    pixels_list: list[int] = []
    intensity_list: list[float] = []
    wl_list: list[float] = []
    intensity_col = 1
    wl_col = -1

    with open(csv_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if line.lower().startswith("pixel"):
                intensity_col = 1
                if "reference_wavelength" in line.lower():
                    wl_col = 2
                elif "calibrated_wavelength" in line.lower():
                    wl_col = 4
                continue
            if len(parts) >= max(2, intensity_col + 1):
                try:
                    px = int(parts[0])
                    intensity_list.append(float(parts[intensity_col]))
                    pixels_list.append(px)
                    wl_list.append(
                        float(parts[wl_col]) if wl_col >= 0 and len(parts) > wl_col else 0.0
                    )
                except (ValueError, IndexError):
                    continue

    if not pixels_list:
        return np.array([]), 0, None

    pairs = sorted(zip(pixels_list, intensity_list, wl_list), key=lambda x: x[0])
    n = pairs[-1][0] + 1 if pairs else 0
    measured = np.zeros(n, dtype=np.float64)
    wavelengths = np.zeros(n, dtype=np.float64) if wl_list else None
    for px, intensity, wl in pairs:
        measured[px] = intensity
        if wavelengths is not None:
            wavelengths[px] = wl
    if wavelengths is not None and np.any(wavelengths > 0):
        px_vals = np.array([p[0] for p in pairs])
        wl_vals = np.array([p[2] for p in pairs])
        for i in range(n):
            if wavelengths[i] == 0:
                wavelengths[i] = float(np.interp(i, px_vals, wl_vals))
    else:
        wavelengths = None
    return measured, n, wavelengths


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Top 6 extremums → generator triplets → best match vs all ref"
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=str(OUTPUT_DIR / "Spectrum-20260311--214345.csv"),
    )
    parser.add_argument("--source", default="FL12")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--n-extremums", type=int, default=6, help="Max extremums to use")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return 1

    measured, n, wl_csv = load_csv(csv_path)
    if n < 10:
        print("Need at least 10 pixels")
        return 1

    positions = (
        wl_csv if wl_csv is not None and np.any(wl_csv > 0)
        else np.linspace(380, 750, n)
    )
    pos = _valid_positions(positions, n)
    position_px = np.arange(n, dtype=np.intp)

    # Top N extremums
    meas = extract(
        measured,
        pos,
        position_px=position_px,
        max_count=args.n_extremums,
    )
    if len(meas) < 2:
        print("Need at least 2 extremums")
        return 1

    # Generate triplets from those extremums
    triplets_meas = list(generate_triplets(meas))
    if not triplets_meas:
        print("No proper triplets generated (rel_pos 0.30–0.70)")
        return 1

    try:
        src = ReferenceSource[args.source.upper()]
    except KeyError:
        src = ReferenceSource.FL12
    ref_peaks = get_reference_peaks(src)
    ref = from_known_lines([p.wavelength for p in ref_peaks], [p.intensity for p in ref_peaks])
    triplets_ref = build_triplets(ref)

    # For each measured triplet, find best matching ref triplet
    matches: list[tuple[Triplet, Triplet, float]] = []
    for ta in triplets_meas:
        best_tb, best_s = None, -1e9
        for tb in triplets_ref:
            if meas[ta.center_idx].is_dip != ref[tb.center_idx].is_dip:
                continue
            s = triplet_score(ta, tb, meas, ref, DEFAULT_WEIGHTS, "euclidean")
            if s > best_s:
                best_s, best_tb = s, tb
        if best_tb is not None and best_s > -1e8:
            matches.append((ta, best_tb, best_s))

    matches.sort(key=lambda m: -m[2])

    # Reference SPD for plotting
    wl_ref = np.linspace(350, 750, 500)
    ref_spd = get_reference_spectrum(src, wl_ref)
    if ref_spd.max() > 0:
        ref_spd = ref_spd / ref_spd.max()
    meas_norm = measured.copy()
    if meas_norm.max() > 0:
        meas_norm = meas_norm / meas_norm.max()

    n_rows = len(matches)
    if n_rows == 0:
        print("No matches")
        return 1

    fig = plt.figure(figsize=(16, 4 * n_rows), constrained_layout=True)
    gs = fig.add_gridspec(n_rows, 2, width_ratios=[2, 1])
    colors = ["#e41a1c", "#377eb8", "#4daf4a"]
    marker_sz = 28

    for row_idx, (ta, tb, score_val) in enumerate(matches):
        ax = fig.add_subplot(gs[row_idx, 0])
        ax_bar = fig.add_subplot(gs[row_idx, 1])

        mask = (pos >= 350) & (pos <= 750)
        ax.plot(pos[mask], meas_norm[mask], "b-", alpha=0.7, linewidth=1.5, label="Measured")
        ax.plot(wl_ref, ref_spd, color="orange", alpha=0.7, linewidth=1.5, label="Ref SPD")

        m_left, m_center, m_right = meas[ta.left_idx], meas[ta.center_idx], meas[ta.right_idx]
        r_left, r_center, r_right = ref[tb.left_idx], ref[tb.center_idx], ref[tb.right_idx]
        m_pos = [m_left.position, m_center.position, m_right.position]
        r_pos = [r_left.position, r_center.position, r_right.position]
        m_ys = [
            float(meas_norm[e.position_px]) if e.position_px is not None and 0 <= e.position_px < n else 0.0
            for e in [m_left, m_center, m_right]
        ]
        r_ys = [float(np.interp(w, wl_ref, ref_spd)) for w in r_pos]

        for i, c in enumerate(colors):
            ax.plot([m_pos[i], r_pos[i]], [m_ys[i], r_ys[i]], color=c, alpha=0.8, linewidth=1.5, zorder=5)
            ax.scatter([m_pos[i]], [m_ys[i]], c=[c], s=marker_sz, zorder=6, edgecolors="black")
            ax.scatter([r_pos[i]], [r_ys[i]], c=[c], s=marker_sz, zorder=6, edgecolors="black", marker="s")

        ax.set_xlim(350, 750)
        ax.set_ylabel("Intensity")
        ax.set_title(
            f"Triplet #{row_idx + 1} (from top {args.n_extremums} extremums) — score={score_val:.4f} "
            f"C={m_center.position:.0f}↔{r_center.position:.0f}nm"
        )
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        x = np.arange(len(DESC_LABELS))
        w = 0.35
        ax_bar.bar(x - w / 2, ta.descriptor, w, label="meas", color="steelblue", alpha=0.8)
        ax_bar.bar(x + w / 2, tb.descriptor, w, label="ref", color="darkorange", alpha=0.8)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(DESC_LABELS, rotation=45, ha="right", fontsize=8)
        ax_bar.set_ylabel("value")
        ax_bar.legend(loc="upper right", fontsize=7)
        ax_bar.set_title(f"score={score_val:.4f}", fontsize=10)

    fig.axes[2 * (n_rows - 1)].set_xlabel("Wavelength (nm)")
    fig.suptitle(
        f"Top {args.n_extremums} extremums → generator triplets → best match vs all ref ({args.source})",
        fontsize=11,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "experiment_top6_triplets.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path} ({len(triplets_meas)} triplets from {len(meas)} extremums, {n_rows} matches)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
