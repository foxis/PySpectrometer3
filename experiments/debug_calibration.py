"""Debug calibration: 5 image types for understanding triplet matching and peak detection.

Uses output/*.csv as inputs. Algorithm lives in pyspectrometer.processing.calibration.

Output images:
1. peak_detection_<csv_stem>.png - measured SPD vs reference SPD, peaks=green, dips=red (per CSV)
2. triplets_with_descriptors.png - two SPDs per row, each with triplet and descriptor
3. best_20_triplet_matches.png - two SPDs, triplet pairs with matching score, top 20
4. triplet_first_vs_ref.png - first measured triplet vs all ref triplets
5. triplet_third_vs_ref.png - third measured triplet vs all ref triplets
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
from pyspectrometer.processing.calibration import (
    DEFAULT_WEIGHTS,
    Triplet,
    build_triplets,
    extract,
    from_known_lines,
    triplet_score,
)
from pyspectrometer.processing.calibration.detect_peaks import get_reference_peaks

DESC_LABELS = ["height", "width", "A-h/h", "B-h/h", "A-w/w", "B-w/w", "rel_pos"]
SOURCE_MAP = {"FL12": ReferenceSource.FL12, "FL1": ReferenceSource.FL1, "HG": ReferenceSource.HG}


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


def _y_at_extremum(ext, intensity: np.ndarray, wavelengths: np.ndarray, n: int) -> float:
    """Get y value for extremum (position is wavelength)."""
    if ext.position_px is not None and 0 <= ext.position_px < n:
        return float(intensity[ext.position_px])
    return float(np.interp(ext.position, wavelengths, intensity)) if len(wavelengths) == n else 0.0


def image1_peak_detection(
    out_dir: Path,
    csv_path: Path,
    measured: np.ndarray,
    n: int,
    wavelengths: np.ndarray,
    source: ReferenceSource,
) -> None:
    """One image per CSV: measured SPD + reference SPD, peaks=green, dips=red."""
    position_px = np.arange(n, dtype=np.intp)
    initial_wl = wavelengths if np.any(wavelengths > 0) else np.linspace(380, 750, n)

    all_meas = extract(
        measured,
        initial_wl,
        position_px=position_px,
        max_count=15,
    )
    peaks_meas = [p for p in all_meas if not p.is_dip]
    dips_meas = [p for p in all_meas if p.is_dip]

    wl_ref = np.linspace(350, 750, n)
    ref_spd = get_reference_spectrum(source, wl_ref)
    if ref_spd.max() > 0:
        ref_spd = ref_spd / ref_spd.max()

    all_ref = extract(ref_spd, wl_ref, position_px=None, max_count=20)
    peaks_ref = [p for p in all_ref if not p.is_dip]
    dips_ref = [p for p in all_ref if p.is_dip]
    if len(peaks_ref) < 4:
        ref_peaks = get_reference_peaks(source)
        peaks_ref = from_known_lines(
            [p.wavelength for p in ref_peaks],
            [p.intensity for p in ref_peaks],
        )
        dips_ref = []

    meas_norm = measured.copy()
    if meas_norm.max() > 0:
        meas_norm = meas_norm / meas_norm.max()

    fig, (ax_meas, ax_ref) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    mask = (initial_wl >= 350) & (initial_wl <= 750)
    ax_meas.plot(initial_wl[mask], meas_norm[mask], "b-", alpha=0.8, label="Measured", linewidth=1)
    ax_meas.scatter(
        [p.position for p in peaks_meas],
        [_y_at_extremum(p, meas_norm, initial_wl, n) for p in peaks_meas],
        c="green",
        s=30,
        zorder=5,
        label="Peaks",
    )
    ax_meas.scatter(
        [p.position for p in dips_meas],
        [_y_at_extremum(p, meas_norm, initial_wl, n) for p in dips_meas],
        c="red",
        s=30,
        zorder=5,
        label="Dips",
    )
    ax_meas.set_ylabel("Intensity")
    ax_meas.set_title(f"Measured SPD — peaks (green), dips (red) — {csv_path.name}")
    ax_meas.legend()
    ax_meas.grid(True, alpha=0.3)

    ax_ref.plot(wl_ref, ref_spd, color="orange", alpha=0.8, label="Reference", linewidth=1)
    ax_ref.scatter(
        [p.position for p in peaks_ref],
        [float(np.interp(p.position, wl_ref, ref_spd)) for p in peaks_ref],
        c="green",
        s=30,
        zorder=5,
        label="Peaks",
    )
    ax_ref.scatter(
        [p.position for p in dips_ref],
        [float(np.interp(p.position, wl_ref, ref_spd)) for p in dips_ref],
        c="red",
        s=30,
        zorder=5,
        label="Dips",
    )
    ax_ref.set_xlabel("Wavelength (nm)")
    ax_ref.set_ylabel("Intensity")
    ax_ref.set_title("Reference SPD — peaks (green), dips (red)")
    ax_ref.legend()
    ax_ref.grid(True, alpha=0.3)
    ax_ref.set_xlim(350, 750)

    plt.tight_layout()
    out_path = out_dir / f"peak_detection_{csv_path.stem}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def image2_triplets_with_descriptors(
    out_dir: Path,
    measured: np.ndarray,
    n: int,
    initial_wl: np.ndarray,
    ref_spd: np.ndarray,
    wl_ref: np.ndarray,
    peaks_meas: list,
    dips_meas: list,
    peaks_ref: list,
    dips_ref: list,
    triplets_meas: list[Triplet],
    triplets_ref: list[Triplet],
    csv_stem: str,
) -> None:
    """Two SPDs per row, each with triplet and descriptor."""
    if not triplets_meas or not triplets_ref:
        return
    n_rows = min(6, max(len(triplets_meas), len(triplets_ref)))
    if n_rows == 0:
        return
    fig, axes = plt.subplots(n_rows, 3, figsize=(14, 3 * n_rows), gridspec_kw={"width_ratios": [1.5, 1.5, 0.8]})
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    meas_norm = measured.copy()
    if meas_norm.max() > 0:
        meas_norm = meas_norm / meas_norm.max()

    def _y_meas(p):
        return _y_at_extremum(p, meas_norm, initial_wl, n)

    def _y_ref(p):
        return float(np.interp(p.position, wl_ref, ref_spd))

    for row in range(n_rows):
        ax_meas, ax_ref, ax_vec = axes[row, 0], axes[row, 1], axes[row, 2]
        mask = (initial_wl >= 350) & (initial_wl <= 750)
        ax_meas.plot(initial_wl[mask], meas_norm[mask], "b-", alpha=0.6, linewidth=1)
        ax_meas.scatter(
            [p.position for p in peaks_meas],
            [_y_meas(p) for p in peaks_meas],
            c="green",
            s=20,
            zorder=3,
        )
        ax_meas.scatter(
            [p.position for p in dips_meas],
            [_y_meas(p) for p in dips_meas],
            c="red",
            s=20,
            zorder=3,
        )

        ax_ref.plot(wl_ref, ref_spd, color="orange", alpha=0.6, linewidth=1)
        ax_ref.scatter(
            [p.position for p in peaks_ref],
            [_y_ref(p) for p in peaks_ref],
            c="green",
            s=20,
            zorder=3,
        )
        ax_ref.scatter(
            [p.position for p in dips_ref],
            [_y_ref(p) for p in dips_ref],
            c="red",
            s=20,
            zorder=3,
        )

        ta = triplets_meas[row % len(triplets_meas)] if triplets_meas else None
        tb = triplets_ref[row % len(triplets_ref)] if triplets_ref else None
        if ta and tb:
            pa, ca, pb = peaks_meas[ta.left_idx], peaks_meas[ta.center_idx], peaks_meas[ta.right_idx]
            qa, cb, qb = peaks_ref[tb.left_idx], peaks_ref[tb.center_idx], peaks_ref[tb.right_idx]
            ax_meas.scatter(
                [pa.position, ca.position, pb.position],
                [_y_meas(pa), _y_meas(ca), _y_meas(pb)],
                c=["green", "red", "green"],
                s=60,
                zorder=5,
                edgecolors="black",
            )
            ax_ref.scatter(
                [qa.position, cb.position, qb.position],
                [_y_ref(qa), _y_ref(cb), _y_ref(qb)],
                c=["green", "red", "green"],
                s=60,
                zorder=5,
                edgecolors="black",
            )
            x = np.arange(len(DESC_LABELS))
            width = 0.35
            ax_vec.bar(x - width / 2, ta.descriptor, width, label="Meas", color="blue", alpha=0.7)
            ax_vec.bar(x + width / 2, tb.descriptor, width, label="Ref", color="orange", alpha=0.7)
            ax_vec.set_xticks(x)
            ax_vec.set_xticklabels(DESC_LABELS, rotation=45, ha="right")
            ax_vec.legend()
        ax_meas.set_xlim(350, 750)
        ax_ref.set_xlim(350, 750)
        ax_meas.set_ylabel("Intensity")
        ax_ref.set_ylabel("Intensity")
        ax_vec.grid(True, alpha=0.3)

    for ax in axes[:, 0]:
        ax.set_xlabel("Wavelength (nm)")
    for ax in axes[:, 1]:
        ax.set_xlabel("Wavelength (nm)")

    fig.suptitle(f"Triplets with descriptors — {csv_stem}", fontsize=10, y=1.02)
    plt.tight_layout()
    out_path = out_dir / f"triplets_with_descriptors_{csv_stem}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def image3_best_20_triplet_matches(
    out_dir: Path,
    measured: np.ndarray,
    n: int,
    initial_wl: np.ndarray,
    ref_spd: np.ndarray,
    wl_ref: np.ndarray,
    peaks_meas: list,
    peaks_ref: list,
    triplets_meas: list[Triplet],
    triplets_ref: list[Triplet],
    csv_stem: str,
) -> None:
    """Two SPDs with triplet in one and corresponding triplet in another, matching score. Top 20."""
    pairs: list[tuple[Triplet, Triplet, float, float]] = []
    for ta in triplets_meas:
        for tb in triplets_ref:
            s_euc = triplet_score(ta, tb, peaks_meas, peaks_ref, DEFAULT_WEIGHTS, "euclidean")
            s_cos = triplet_score(ta, tb, peaks_meas, peaks_ref, DEFAULT_WEIGHTS, "cosine")
            if s_euc > -1e8:
                pairs.append((ta, tb, s_euc, s_cos))
    pairs.sort(key=lambda x: -x[2])
    pairs = pairs[:20]
    if not pairs:
        return

    meas_norm = measured.copy()
    if meas_norm.max() > 0:
        meas_norm = meas_norm / meas_norm.max()

    n_rows = len(pairs)
    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(14, 2.5 * n_rows),
        gridspec_kw={"width_ratios": [1.5, 1.5, 0.8]},
    )
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    def _y_meas(p):
        return _y_at_extremum(p, meas_norm, initial_wl, n)

    def _y_ref(p):
        return float(np.interp(p.position, wl_ref, ref_spd))

    for row, (ta, tb, s_euc, s_cos) in enumerate(pairs):
        ax_meas, ax_ref, ax_vec = axes[row, 0], axes[row, 1], axes[row, 2]
        mask = (initial_wl >= 350) & (initial_wl <= 750)
        ax_meas.plot(initial_wl[mask], meas_norm[mask], "b-", alpha=0.6, linewidth=1)
        pa, ca, pb = peaks_meas[ta.left_idx], peaks_meas[ta.center_idx], peaks_meas[ta.right_idx]
        ax_meas.scatter(
            [pa.position, ca.position, pb.position],
            [_y_meas(pa), _y_meas(ca), _y_meas(pb)],
            c=["green", "red", "green"],
            s=60,
            zorder=5,
            edgecolors="black",
        )
        ax_meas.set_xlim(350, 750)
        ax_meas.set_ylabel("Intensity")

        ax_ref.plot(wl_ref, ref_spd, color="orange", alpha=0.6, linewidth=1)
        qa, cb, qb = peaks_ref[tb.left_idx], peaks_ref[tb.center_idx], peaks_ref[tb.right_idx]
        ax_ref.scatter(
            [qa.position, cb.position, qb.position],
            [_y_ref(qa), _y_ref(cb), _y_ref(qb)],
            c=["green", "red", "green"],
            s=60,
            zorder=5,
            edgecolors="black",
        )
        ax_ref.set_xlim(350, 750)
        ax_ref.set_ylabel("Intensity")

        x = np.arange(len(DESC_LABELS))
        width = 0.35
        ax_vec.bar(x - width / 2, ta.descriptor, width, label="Meas", color="blue", alpha=0.7)
        ax_vec.bar(x + width / 2, tb.descriptor, width, label="Ref", color="orange", alpha=0.7)
        ax_vec.set_xticks(x)
        ax_vec.set_xticklabels(DESC_LABELS, rotation=45, ha="right")
        ax_vec.set_title(f"Euc={s_euc:.4f} Cos={s_cos:.4f}")
        ax_vec.legend()
        ax_vec.grid(True, alpha=0.3)

    for ax in axes[:, 0]:
        ax.set_xlabel("Wavelength (nm)")
    for ax in axes[:, 1]:
        ax.set_xlabel("Wavelength (nm)")

    fig.suptitle(f"20 best matching triplets — {csv_stem}", fontsize=10, y=1.02)
    plt.tight_layout()
    out_path = out_dir / f"best_20_triplet_matches_{csv_stem}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def image4_first_triplet_vs_ref(
    out_dir: Path,
    measured: np.ndarray,
    n: int,
    initial_wl: np.ndarray,
    ref_spd: np.ndarray,
    wl_ref: np.ndarray,
    peaks_meas: list,
    dips_meas: list,
    peaks_ref: list,
    dips_ref: list,
    triplets_meas: list[Triplet],
    triplets_ref: list[Triplet],
    csv_stem: str,
) -> None:
    """First measured triplet vs all ref triplets."""
    _image_triplet_vs_ref(
        out_dir,
        measured,
        n,
        initial_wl,
        ref_spd,
        wl_ref,
        peaks_meas,
        dips_meas,
        peaks_ref,
        dips_ref,
        triplets_meas,
        triplets_ref,
        center_idx=0,
        csv_stem=csv_stem,
        suffix="first",
    )


def image5_third_triplet_vs_ref(
    out_dir: Path,
    measured: np.ndarray,
    n: int,
    initial_wl: np.ndarray,
    ref_spd: np.ndarray,
    wl_ref: np.ndarray,
    peaks_meas: list,
    dips_meas: list,
    peaks_ref: list,
    dips_ref: list,
    triplets_meas: list[Triplet],
    triplets_ref: list[Triplet],
    csv_stem: str,
) -> None:
    """Third measured triplet vs all ref triplets."""
    _image_triplet_vs_ref(
        out_dir,
        measured,
        n,
        initial_wl,
        ref_spd,
        wl_ref,
        peaks_meas,
        dips_meas,
        peaks_ref,
        dips_ref,
        triplets_meas,
        triplets_ref,
        center_idx=min(2, len(peaks_meas) - 1) if peaks_meas else 0,
        csv_stem=csv_stem,
        suffix="third",
    )


def _image_triplet_vs_ref(
    out_dir: Path,
    measured: np.ndarray,
    n: int,
    initial_wl: np.ndarray,
    ref_spd: np.ndarray,
    wl_ref: np.ndarray,
    peaks_meas: list,
    dips_meas: list,
    peaks_ref: list,
    dips_ref: list,
    triplets_meas: list[Triplet],
    triplets_ref: list[Triplet],
    center_idx: int,
    csv_stem: str,
    suffix: str,
) -> None:
    """One measured triplet (by center index) vs all ref triplets."""
    ta_list = [t for t in triplets_meas if t.center_idx == center_idx]
    ta = ta_list[0] if ta_list else (triplets_meas[0] if triplets_meas else None)
    if ta is None:
        return

    pairs: list[tuple[Triplet, Triplet, float, float]] = []
    for tb in triplets_ref:
        s_euc = triplet_score(ta, tb, peaks_meas, peaks_ref, DEFAULT_WEIGHTS, "euclidean")
        s_cos = triplet_score(ta, tb, peaks_meas, peaks_ref, DEFAULT_WEIGHTS, "cosine")
        pairs.append((ta, tb, s_euc, s_cos))
    pairs.sort(key=lambda x: -x[2])
    pairs = pairs[:20]

    meas_norm = measured.copy()
    if meas_norm.max() > 0:
        meas_norm = meas_norm / meas_norm.max()

    def _y_meas(p):
        return _y_at_extremum(p, meas_norm, initial_wl, n)

    def _y_ref(p):
        return float(np.interp(p.position, wl_ref, ref_spd))

    n_rows = len(pairs)
    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(14, 2.5 * n_rows),
        gridspec_kw={"width_ratios": [1.5, 1.5, 0.8]},
    )
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for row, (ta, tb, s_euc, s_cos) in enumerate(pairs):
        ax_meas, ax_ref, ax_vec = axes[row, 0], axes[row, 1], axes[row, 2]
        mask = (initial_wl >= 350) & (initial_wl <= 750)
        ax_meas.plot(initial_wl[mask], meas_norm[mask], "b-", alpha=0.6, linewidth=1)
        triplet_idx = {ta.left_idx, ta.center_idx, ta.right_idx}
        other_peaks = [p for i, p in enumerate(peaks_meas) if i not in triplet_idx]
        if other_peaks:
            ax_meas.scatter(
                [p.position for p in other_peaks],
                [_y_meas(p) for p in other_peaks],
                c="green",
                s=20,
                zorder=3,
            )
        if dips_meas:
            ax_meas.scatter(
                [p.position for p in dips_meas],
                [_y_meas(p) for p in dips_meas],
                c="red",
                s=20,
                zorder=3,
            )
        pa, ca, pb = peaks_meas[ta.left_idx], peaks_meas[ta.center_idx], peaks_meas[ta.right_idx]
        ax_meas.scatter(
            [pa.position, ca.position, pb.position],
            [_y_meas(pa), _y_meas(ca), _y_meas(pb)],
            c=["green", "red", "green"],
            s=80,
            zorder=5,
            edgecolors="black",
        )
        ax_meas.set_title(f"Meas (L{ta.left_idx},C{ta.center_idx},R{ta.right_idx})")
        ax_meas.set_xlim(350, 750)
        ax_meas.set_ylabel("Intensity")

        ax_ref.plot(wl_ref, ref_spd, color="orange", alpha=0.6, linewidth=1)
        ref_triplet_idx = {tb.left_idx, tb.center_idx, tb.right_idx}
        other_ref = [p for i, p in enumerate(peaks_ref) if i not in ref_triplet_idx]
        if other_ref:
            ax_ref.scatter(
                [p.position for p in other_ref],
                [_y_ref(p) for p in other_ref],
                c="green",
                s=20,
                zorder=3,
            )
        if dips_ref:
            ax_ref.scatter(
                [p.position for p in dips_ref],
                [_y_ref(p) for p in dips_ref],
                c="red",
                s=20,
                zorder=3,
            )
        qa, cb, qb = peaks_ref[tb.left_idx], peaks_ref[tb.center_idx], peaks_ref[tb.right_idx]
        ax_ref.scatter(
            [qa.position, cb.position, qb.position],
            [_y_ref(qa), _y_ref(cb), _y_ref(qb)],
            c=["green", "red", "green"],
            s=80,
            zorder=5,
            edgecolors="black",
        )
        ax_ref.set_title(f"Ref (L{tb.left_idx},C{tb.center_idx},R{tb.right_idx})")
        ax_ref.set_xlim(350, 750)
        ax_ref.set_ylabel("Intensity")

        x = np.arange(len(DESC_LABELS))
        width = 0.35
        ax_vec.bar(x - width / 2, ta.descriptor, width, label="Meas", color="blue", alpha=0.7)
        ax_vec.bar(x + width / 2, tb.descriptor, width, label="Ref", color="orange", alpha=0.7)
        ax_vec.set_xticks(x)
        ax_vec.set_xticklabels(DESC_LABELS, rotation=45, ha="right")
        ax_vec.set_title(f"Euc={s_euc:.4f} Cos={s_cos:.4f}")
        ax_vec.legend()
        ax_vec.grid(True, alpha=0.3)

    for ax in axes[:, 0]:
        ax.set_xlabel("Wavelength (nm)")
    for ax in axes[:, 1]:
        ax.set_xlabel("Wavelength (nm)")

    fig.suptitle(f"Triplet {suffix} vs all ref triplets — {csv_stem}", fontsize=10, y=1.02)
    plt.tight_layout()
    out_path = out_dir / f"triplet_{suffix}_vs_ref_{csv_stem}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug calibration with 5 image types")
    parser.add_argument(
        "csv_paths",
        nargs="*",
        default=[str(p) for p in sorted(OUTPUT_DIR.glob("*.csv"))],
        help="CSV files (default: output/*.csv)",
    )
    parser.add_argument("--source", default="FL12")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source = SOURCE_MAP.get(args.source.upper(), ReferenceSource.FL12)

    csv_paths = [Path(p) for p in args.csv_paths if Path(p).exists()]
    if not csv_paths:
        print("No CSV files found")
        return 1

    for csv_path in csv_paths:
        measured, n, wavelengths = load_csv(csv_path)
        if n < 10:
            print(f"Skipping {csv_path.name}: need at least 10 pixels")
            continue

        initial_wl = wavelengths if np.any(wavelengths > 0) else np.linspace(380, 750, n)
        position_px = np.arange(n, dtype=np.intp)

        all_meas = extract(
            measured,
            initial_wl,
            position_px=position_px,
            max_count=15,
        )
        peaks_meas = [p for p in all_meas if not p.is_dip]
        dips_meas = [p for p in all_meas if p.is_dip]

        wl_ref = np.linspace(350, 750, n)
        ref_spd = get_reference_spectrum(source, wl_ref)
        if ref_spd.max() > 0:
            ref_spd = ref_spd / ref_spd.max()

        all_ref = extract(ref_spd, wl_ref, position_px=None, max_count=20)
        peaks_ref = [p for p in all_ref if not p.is_dip]
        dips_ref = [p for p in all_ref if p.is_dip]
        if len(peaks_ref) < 4:
            ref_peaks = get_reference_peaks(source)
            peaks_ref = from_known_lines(
                [p.wavelength for p in ref_peaks],
                [p.intensity for p in ref_peaks],
            )
            dips_ref = []

        triplets_meas = build_triplets(peaks_meas)
        triplets_ref = build_triplets(peaks_ref)

        image1_peak_detection(out_dir, csv_path, measured, n, initial_wl, source)
        image2_triplets_with_descriptors(
            out_dir,
            measured,
            n,
            initial_wl,
            ref_spd,
            wl_ref,
            peaks_meas,
            dips_meas,
            peaks_ref,
            dips_ref,
            triplets_meas,
            triplets_ref,
            csv_path.stem,
        )
        image3_best_20_triplet_matches(
            out_dir,
            measured,
            n,
            initial_wl,
            ref_spd,
            wl_ref,
            peaks_meas,
            peaks_ref,
            triplets_meas,
            triplets_ref,
            csv_path.stem,
        )
        image4_first_triplet_vs_ref(
            out_dir,
            measured,
            n,
            initial_wl,
            ref_spd,
            wl_ref,
            peaks_meas,
            dips_meas,
            peaks_ref,
            dips_ref,
            triplets_meas,
            triplets_ref,
            csv_path.stem,
        )
        image5_third_triplet_vs_ref(
            out_dir,
            measured,
            n,
            initial_wl,
            ref_spd,
            wl_ref,
            peaks_meas,
            dips_meas,
            peaks_ref,
            dips_ref,
            triplets_meas,
            triplets_ref,
            csv_path.stem,
        )

    print(f"All images saved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
