"""Draw Nth measured triplet vs ref triplets 1, 2, 3, … for two CSV files.

For each CSV: take the Nth measured triplet (by position), match to ref triplets
in order (1st, 2nd, 3rd by position). Generates separate images for 1st, 2nd, 3rd.
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
from pyspectrometer.processing.calibration.extremum import extract
from pyspectrometer.processing.calibration.triplet import (
    DEFAULT_WEIGHTS,
    DESC_LABELS,
    Triplet,
    build as build_triplets,
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


def _process_csv(
    csv_path: Path,
    src: ReferenceSource,
    triplet_index: int = 0,
) -> tuple[Triplet | None, list, list, list[tuple[Triplet, float]], np.ndarray, np.ndarray, int]:
    """Return (ta, meas, ref, [(tb, score), ...], pos, meas_norm, n). triplet_index 0=1st, 1=2nd, 2=3rd."""
    measured, n, wl_csv = load_csv(csv_path)
    if n < 10:
        return None, [], [], [], np.array([]), np.array([]), n

    positions = (
        wl_csv
        if wl_csv is not None and np.any(wl_csv > 0)
        else np.linspace(380, 750, n)
    )
    pos = _valid_positions(positions, n)
    position_px = np.arange(n, dtype=np.intp)

    meas = extract(measured, pos, position_px=position_px, max_count=15)

    # Ref from SPD (spectrum curve), not known lines
    ref_wl = np.linspace(350, 750, n)
    ref_spd_arr = get_reference_spectrum(src, ref_wl)
    ref_spd_arr = np.asarray(ref_spd_arr, dtype=np.float64)
    if ref_spd_arr.max() > 0:
        ref_spd_arr = ref_spd_arr / ref_spd_arr.max()
    ref = extract(
        ref_spd_arr,
        ref_wl,
        position_px=np.arange(n, dtype=np.intp),
        max_count=15,
    )

    if len(meas) < 2 or len(ref) < 2:
        return None, meas, ref, [], pos, np.array([]), n

    triplets_meas = build_triplets(meas)
    triplets_ref = build_triplets(ref)

    # Measured triplets by position: 1st, 2nd, 3rd, …
    triplets_meas_by_pos = sorted(
        triplets_meas,
        key=lambda t: (meas[t.center_idx].position, t.left_idx, t.right_idx),
    )
    if triplet_index >= len(triplets_meas_by_pos):
        return None, meas, ref, [], pos, np.array([]), n
    ta = triplets_meas_by_pos[triplet_index]

    # Ref triplets by position: 1st, 2nd, 3rd, …
    ref_by_pos = sorted(triplets_ref, key=lambda t: ref[t.center_idx].position)

    candidates: list[tuple[Triplet, float]] = []
    for tb in ref_by_pos:
        s = triplet_score(ta, tb, meas, ref, DEFAULT_WEIGHTS, "euclidean")
        if s > -1e8:
            candidates.append((tb, s))

    meas_norm = measured.copy()
    if meas_norm.max() > 0:
        meas_norm = meas_norm / meas_norm.max()

    return ta, meas, ref, candidates, pos, meas_norm, n


def main() -> int:
    parser = argparse.ArgumentParser(
        description="First triplet vs other ref triplets for two CSVs"
    )
    parser.add_argument(
        "csv_paths",
        nargs="*",
        default=[
            str(OUTPUT_DIR / "Spectrum-20260311--214345.csv"),
            str(OUTPUT_DIR / "Spectrum-20260311--193723.csv"),
        ],
    )
    parser.add_argument("--source", default="FL12")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--max-refs", type=int, default=6, help="Max ref triplets per CSV")
    parser.add_argument(
        "--triplet-index",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="0=1st, 1=2nd, 2=3rd measured triplet",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all 3 images (1st, 2nd, 3rd triplet vs others)",
    )
    args = parser.parse_args()

    try:
        src = ReferenceSource[args.source.upper()]
    except KeyError:
        src = ReferenceSource.FL12

    csv_paths = [Path(p) for p in args.csv_paths if Path(p).exists()]
    if len(csv_paths) < 2:
        csv_paths = list(Path(OUTPUT_DIR).glob("*.csv"))[:2]
    if not csv_paths:
        print("No CSV files found")
        return 1

    indices = [0, 1, 2] if args.all else [args.triplet_index]
    labels = ["first", "second", "third"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx in indices:
        datasets: list[tuple[Path, Triplet, list, list, list[tuple[Triplet, float]], np.ndarray, np.ndarray, int]] = []
        for csv_path in csv_paths[:2]:
            result = _process_csv(csv_path, src, triplet_index=idx)
            if result[0] is None:
                continue
            ta, meas, ref, candidates, pos, meas_norm, n = result
            datasets.append((csv_path, ta, meas, ref, candidates, pos, meas_norm, n))

        if not datasets:
            continue

        wl_ref = np.linspace(350, 750, 500)
        ref_spd = get_reference_spectrum(src, wl_ref)
        if ref_spd.max() > 0:
            ref_spd = ref_spd / ref_spd.max()

        n_rows = sum(min(args.max_refs, len(d[4])) for d in datasets)
        if n_rows == 0:
            continue

        fig = plt.figure(figsize=(16, 3.5 * n_rows), constrained_layout=True)
        gs = fig.add_gridspec(n_rows, 2, width_ratios=[2, 1])
        row_idx = 0

        for csv_path, ta, meas, ref, candidates, pos, meas_norm, n in datasets:
            n_show = min(args.max_refs, len(candidates))
            for k in range(n_show):
                tb, score_val = candidates[k]
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
                colors = ["#e41a1c", "#377eb8", "#4daf4a"]
                for i, c in enumerate(colors):
                    ax.plot([m_pos[i], r_pos[i]], [m_ys[i], r_ys[i]], color=c, alpha=0.9, linewidth=2, zorder=5)
                    ax.scatter([m_pos[i]], [m_ys[i]], c=[c], s=80, zorder=6, edgecolors="black")
                    ax.scatter([r_pos[i]], [r_ys[i]], c=[c], s=80, zorder=6, edgecolors="black", marker="s")
                ax.set_xlim(350, 750)
                ax.set_ylabel("Intensity")
                ax.legend(loc="upper right", fontsize=7)
                ax.set_title(
                    f"{csv_path.stem}: {labels[idx]} meas triplet vs ref #{k + 1} "
                    f"(C={m_center.position:.0f}↔{r_center.position:.0f}nm) score={score_val:.4f}"
                )
                ax.grid(True, alpha=0.3)

                x = np.arange(len(DESC_LABELS))
                w = 0.35
                ax_bar.bar(x - w / 2, ta.descriptor, w, label="meas", color="steelblue", alpha=0.8)
                ax_bar.bar(x + w / 2, tb.descriptor, w, label="ref", color="darkorange", alpha=0.8)
                ax_bar.set_xticks(x)
                ax_bar.set_xticklabels(DESC_LABELS, rotation=45, ha="right", fontsize=8)
                ax_bar.set_ylabel("value")
                ax_bar.legend(loc="upper right", fontsize=7)
                ax_bar.set_title(f"score={score_val:.4f}", fontsize=9)

                if row_idx == n_rows - 1:
                    ax.set_xlabel("Wavelength (nm)")
                row_idx += 1

        fig.suptitle(
            f"{labels[idx].capitalize()} measured triplet vs ref triplets 1st, 2nd, 3rd, … — {args.source}",
            fontsize=11,
        )

        out_path = out_dir / f"debug_{labels[idx]}_triplet_vs_others.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
