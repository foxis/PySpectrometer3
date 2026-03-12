"""Debug: compare _feature_pair_score at GT vs false calibration."""
import sys
sys.path.insert(0, 'src')
import numpy as np
from pyspectrometer.data.reference_spectra import ReferenceSource, get_reference_spectrum
from pyspectrometer.processing.calibration.hough_matching import _best_match_score
from pyspectrometer.processing.calibration.extremum import extract


def load(f):
    ints, wls = [], []
    with open(f) as fh:
        for line in fh:
            s = line.strip()
            if s.startswith('#') or s.lower().startswith('pixel'):
                continue
            p = s.split(',')
            if len(p) >= 5:
                try:
                    ints.append(float(p[1]))
                    wls.append(float(p[4]))
                except Exception:
                    pass
    return np.array(ints), np.array(wls)


for csv_path, src_name in [
    ('output/Spectrum-20260311--213319.csv', 'HG'),
    ('output/Spectrum-20260311--214345.csv', 'HG'),
    ('output/Spectrum-20260311--193856.csv', 'FL12'),
]:
    intensity, wl_gt = load(csv_path)
    n = len(intensity)
    src = ReferenceSource[src_name]
    wl_ref = np.linspace(380, 750, 500)
    ref_spd = get_reference_spectrum(src, wl_ref)

    ref_exts = extract(ref_spd, wl_ref, position_px=None, max_count=40)
    rw = np.array([e.position for e in ref_exts])
    rd = np.array([e.is_dip for e in ref_exts])
    ri = np.array([abs(e.height) for e in ref_exts])
    ri_n = ri / ri.max()

    px_arr_gt = np.arange(n, dtype=np.intp)
    meas_exts = extract(intensity, wl_gt, position_px=px_arr_gt, max_count=40)
    meas_px = np.array([e.position_px for e in meas_exts if e.position_px is not None], dtype=float)
    meas_dip = np.array([e.is_dip for e in meas_exts if e.position_px is not None])
    meas_h = np.array([abs(e.height) for e in meas_exts if e.position_px is not None])
    mi_n = meas_h / meas_h.max()

    slope_gt = (wl_gt[-1] - wl_gt[0]) / (n - 1)
    c_gt = float(wl_gt[0])

    px = np.arange(n, dtype=np.float64)

    print(f"\n=== {csv_path.split('/')[-1]} ({src_name}) ===")
    print(f"GT: slope={slope_gt:.4f} c={c_gt:.2f}")
    print(f"Features: {len(meas_px)} meas, {len(rw)} ref")

    candidates = [
        (slope_gt, c_gt, 'GT'),
        (0.494, 270.26, 'FALSE-FL12(0.494,270)'),
        (0.238, 274.56, 'FALSE-HG(0.238,274)'),
    ]
    for m, c, lbl in candidates:
        wl_a = m * px + c
        for sig in [10.0, 5.0, 4.0, 2.5]:
            s = _best_match_score(meas_px, meas_dip, mi_n, rw, rd, ri_n, wl_a, sig, 0.3)
            print(f"  {lbl:<26} sigma={sig:4.1f}  score={s:.3f}")
