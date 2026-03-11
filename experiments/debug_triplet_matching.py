"""Debug triplet matching: ONE measured triplet vs ALL reference triplets.

Output: multi-row PNG with SPDs, vectors, and score. Sorted by Euc, highest on top.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / ".tmp"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(TMP))

import matplotlib.pyplot as plt

from calibration_srp.detect import extract, from_known_lines, Extremum
from calibration_srp.descriptor import DEFAULT_WEIGHTS, Triplet, build as build_triplets, score as triplet_score
from calibration_srp.detect_peaks import get_reference_peaks
from pyspectrometer.data.reference_spectra import ReferenceSource, get_reference_spectrum


DESC_LABELS = ["height", "width", "A-h/h", "B-h/h", "A-w/w", "B-w/w", "rel_pos"]


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
                continue
            if len(parts) >= max(2, wl_col + 1):
                try:
                    px = int(parts[0])
                    intensity = float(parts[intensity_col])
                    wl = float(parts[wl_col])
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
    import argparse
    parser = argparse.ArgumentParser(description="Debug triplet matching")
    parser.add_argument("csv_path", nargs="?", default=str(ROOT / "output" / "Spectrum-20260311--214345.csv"))
    parser.add_argument("--source", default="FL12")
    parser.add_argument("--center", type=int, default=1, help="Measured triplet center index (default 1)")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"Not found: {csv_path}")
        return 1

    measured, n, wavelengths = load_csv(csv_path)
    if n < 10:
        print("Need at least 10 pixels")
        return 1

    initial_wl = np.linspace(380, 750, n) if not np.any(wavelengths > 0) else wavelengths
    position_px = np.arange(n, dtype=np.intp)

    all_meas = extract(
        measured,
        initial_wl,
        position_px=position_px,
        max_count=15,
    )
    peaks_meas = [p for p in all_meas if not p.is_dip]
    dips_meas = [p for p in all_meas if p.is_dip]

    src_map = {"FL12": ReferenceSource.FL12, "FL1": ReferenceSource.FL1, "HG": ReferenceSource.HG}
    source = src_map.get(args.source.upper(), ReferenceSource.FL12)
    wl_ref = np.linspace(350, 750, n)
    ref_spd = get_reference_spectrum(source, wl_ref)
    if ref_spd.max() > 0:
        ref_spd = ref_spd / ref_spd.max()

    all_ref = extract(ref_spd, wl_ref, position_px=None, max_count=20)
    peaks_ref = [p for p in all_ref if not p.is_dip]
    dips_ref = [p for p in all_ref if p.is_dip]
    if len(peaks_ref) < 4:
        # TODO: fallback to pure convolution optimization autocalib (correlation-based)
        ref_peaks = get_reference_peaks(source)
        peaks_ref = from_known_lines(
            [p.wavelength for p in ref_peaks],
            [p.intensity for p in ref_peaks],
        )
        dips_ref = []

    triplets_meas = build_triplets(peaks_meas)
    triplets_ref = build_triplets(peaks_ref)

    ta_list = [t for t in triplets_meas if t.center_idx == args.center]
    ta = ta_list[0] if ta_list else triplets_meas[0]
    pairs: list[tuple[Triplet, Triplet, float, float]] = []
    for tb in triplets_ref:
        s_euc = triplet_score(ta, tb, peaks_meas, peaks_ref, DEFAULT_WEIGHTS, "euclidean")
        s_cos = triplet_score(ta, tb, peaks_meas, peaks_ref, DEFAULT_WEIGHTS, "cosine")
        pairs.append((ta, tb, s_euc, s_cos))
    pairs.sort(key=lambda x: -x[2])

    n_rows = len(pairs)
    if n_rows == 0:
        print("No triplet pairs")
        return 1
    meas_norm = measured.copy()
    if meas_norm.max() > 0:
        meas_norm = meas_norm / meas_norm.max()

    fig, axes = plt.subplots(
        n_rows, 3,
        figsize=(14, 3 * n_rows),
        gridspec_kw={"width_ratios": [1.5, 1.5, 0.8]},
    )
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for row, (ta, tb, s_euc, s_cos) in enumerate(pairs):
        ax_meas, ax_ref, ax_vec = axes[row, 0], axes[row, 1], axes[row, 2]

        pa = peaks_meas[ta.left_idx]
        ca = peaks_meas[ta.center_idx]
        pb = peaks_meas[ta.right_idx]
        qa = peaks_ref[tb.left_idx]
        cb = peaks_ref[tb.center_idx]
        qb = peaks_ref[tb.right_idx]

        wl_meas = np.interp(np.arange(n), np.arange(len(initial_wl)), initial_wl)
        mask = (wl_meas >= 350) & (wl_meas <= 750)
        ax_meas.plot(wl_meas[mask], meas_norm[mask], "b-", alpha=0.6, linewidth=1)
        def _y_meas(p: Extremum) -> float:
            if p.position_px is None or p.position_px < 0 or p.position_px >= n:
                return 0.0
            return float(meas_norm[p.position_px])

        triplet_idx = {ta.left_idx, ta.center_idx, ta.right_idx}
        other_peaks = [p for i, p in enumerate(peaks_meas) if i not in triplet_idx]
        if other_peaks:
            ox = [p.position for p in other_peaks]
            oy = [_y_meas(p) for p in other_peaks]
            ax_meas.scatter(ox, oy, c="green", s=20, zorder=3, alpha=0.8)
        if dips_meas:
            dx = [p.position for p in dips_meas]
            dy = [_y_meas(p) for p in dips_meas]
            ax_meas.scatter(dx, dy, c="red", s=20, zorder=3, alpha=0.8)

        ax_meas.scatter(
            [pa.position, ca.position, pb.position],
            [_y_meas(pa), _y_meas(ca), _y_meas(pb)],
            c=["green", "red", "green"], s=80, zorder=5, edgecolors="black",
        )
        meas_wl_str = f"{ca.position:.1f}nm h={ca.height:.2f}"
        ax_meas.set_title(f"Meas triplet (L{ta.left_idx},C{ta.center_idx},R{ta.right_idx}) {meas_wl_str}")
        ax_meas.set_xlim(350, 750)
        ax_meas.set_ylabel("Intensity")
        ax_meas.grid(True, alpha=0.3)

        ax_ref.plot(wl_ref, ref_spd, color="orange", alpha=0.6, linewidth=1)

        def _y_ref(p: Extremum) -> float:
            return float(np.interp(p.position, wl_ref, ref_spd))

        ref_triplet_idx = {tb.left_idx, tb.center_idx, tb.right_idx}
        other_ref = [p for i, p in enumerate(peaks_ref) if i not in ref_triplet_idx]
        if other_ref:
            rx = [p.position for p in other_ref]
            ry = [_y_ref(p) for p in other_ref]
            ax_ref.scatter(rx, ry, c="green", s=20, zorder=3, alpha=0.8)
        if dips_ref:
            dx = [p.position for p in dips_ref]
            dy = [_y_ref(p) for p in dips_ref]
            ax_ref.scatter(dx, dy, c="red", s=20, zorder=3, alpha=0.8)
        y_qa = float(np.interp(qa.position, wl_ref, ref_spd))
        y_cb = float(np.interp(cb.position, wl_ref, ref_spd))
        y_qb = float(np.interp(qb.position, wl_ref, ref_spd))
        ax_ref.scatter(
            [qa.position, cb.position, qb.position],
            [y_qa, y_cb, y_qb],
            c=["green", "red", "green"], s=80, zorder=5, edgecolors="black",
        )
        ref_wl_str = f"{cb.position:.1f}nm h={cb.height:.2f}"
        ax_ref.set_title(f"Ref triplet (L{tb.left_idx},C{tb.center_idx},R{tb.right_idx}) {ref_wl_str}")
        ax_ref.set_xlim(350, 750)
        ax_ref.set_ylabel("Intensity")
        ax_ref.grid(True, alpha=0.3)

        x = np.arange(len(DESC_LABELS))
        width = 0.35
        va = ta.descriptor
        vb = tb.descriptor
        ax_vec.bar(x - width/2, va, width, label="Meas", color="blue", alpha=0.7)
        ax_vec.bar(x + width/2, vb, width, label="Ref", color="orange", alpha=0.7)
        diff_a = abs(va[4] - vb[4])
        diff_b = abs(va[5] - vb[5])
        ax_vec.text(0.02, 0.98, f"ΔA-w/w={diff_a:.2f} ΔB-w/w={diff_b:.2f}", transform=ax_vec.transAxes, fontsize=8, va="top")
        ax_vec.set_xticks(x)
        ax_vec.set_xticklabels(DESC_LABELS, rotation=45, ha="right")
        ax_vec.set_title(f"Meas C{ta.center_idx}→Ref C{tb.center_idx}  Euc={s_euc:.4f} Cos={s_cos:.4f}")
        ax_vec.legend()
        ax_vec.grid(True, alpha=0.3)

    for ax in axes[:, 0]:
        ax.set_xlabel("Wavelength (nm)")
    for ax in axes[:, 1]:
        ax.set_xlabel("Wavelength (nm)")
    axes[-1, 2].set_xlabel("Descriptor")

    fig.suptitle(
        f"Single measured triplet vs all ref triplets. Sorted by Euc, highest on top.",
        fontsize=10, y=1.02,
    )
    plt.tight_layout()
    out_path = TMP / "debug_triplet_matching.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    ca = peaks_meas[ta.center_idx]
    print(f"Measured triplet: (L{ta.left_idx},C{ta.center_idx},R{ta.right_idx}) px={ca.position_px} wl~{ca.position:.1f}nm h={ca.height:.2f}")
    print(f"Reference triplets: {len(triplets_ref)}")
    print("Matches (Euc desc):")
    for i, (_, tb, s_euc, s_cos) in enumerate(pairs):
        cb = peaks_ref[tb.center_idx]
        print(f"  {i+1}. Ref C{tb.center_idx} ({cb.position:.1f}nm h={cb.height:.2f}) Euc={s_euc:.4f} Cos={s_cos:.4f}")
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
