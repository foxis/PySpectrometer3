"""Debug: which reference (FL12 vs HG) matches each CSV at GT calibration."""
import sys
sys.path.insert(0, 'src')
import numpy as np
from pyspectrometer.data.reference_spectra import ReferenceSource, get_reference_spectrum
from pyspectrometer.processing.calibration.hough_matching import _feature_pair_score
from pyspectrometer.processing.calibration.extremum import extract


def load(path):
    ints, wls = [], []
    with open(path) as fh:
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


for csv_path in [
    'output/Spectrum-20260311--213319.csv',
    'output/Spectrum-20260311--214345.csv',
    'output/Spectrum-20260311--193856.csv',
]:
    intensity, wl_gt = load(csv_path)
    n = len(intensity)
    slope_gt = (wl_gt[-1] - wl_gt[0]) / (n - 1)
    c_gt = float(wl_gt[0])
    px = np.arange(n, dtype=np.float64)
    px_idx = px.astype(np.intp)
    meas_exts = extract(intensity, wl_gt, position_px=px_idx, max_count=40)
    meas_px = np.array([e.position_px for e in meas_exts if e.position_px is not None], dtype=float)
    meas_dip = np.array([e.is_dip for e in meas_exts if e.position_px is not None])
    meas_h = np.array([abs(e.height) for e in meas_exts if e.position_px is not None])
    mi_n = meas_h / meas_h.max()
    wl_arr = slope_gt * px + c_gt

    print(f"\n{csv_path.split('/')[-1]}  slope={slope_gt:.3f}  range={wl_gt[0]:.0f}-{wl_gt[-1]:.0f}")
    for src_name in ['FL12', 'HG']:
        src = ReferenceSource[src_name]
        wl_ref = np.linspace(330, 830, 600)
        ref_spd = get_reference_spectrum(src, wl_ref)
        ref_exts = extract(ref_spd, wl_ref, position_px=None, max_count=40)
        rw = np.array([e.position for e in ref_exts])
        rd = np.array([e.is_dip for e in ref_exts])
        ri = np.array([abs(e.height) for e in ref_exts])
        ri_n = ri / ri.max()
        s = _feature_pair_score(meas_px, meas_dip, mi_n, rw, rd, ri_n, wl_arr, sigma_nm=8.0, sigma_int=0.3)
        print(f"  GT vs {src_name:4s}: score={s:.3f}  ({len(rw)} ref feats, {len(meas_px)} meas feats)")
