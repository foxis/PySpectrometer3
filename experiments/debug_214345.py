"""Debug: inspect measured features in --214345 and match vs HG reference."""
import sys
sys.path.insert(0, 'src')
import numpy as np
from pyspectrometer.data.reference_spectra import ReferenceSource, get_reference_spectrum
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


intensity, wl_gt = load('output/Spectrum-20260311--214345.csv')
n = len(intensity)
slope_gt = (wl_gt[-1] - wl_gt[0]) / (n - 1)
c_gt = float(wl_gt[0])
px_idx = np.arange(n, dtype=np.intp)
exts = extract(intensity, wl_gt, position_px=px_idx, max_count=40)

print(f"GT: slope={slope_gt:.4f}  c={c_gt:.2f}  range={wl_gt[0]:.0f}-{wl_gt[-1]:.0f} nm")
print(f"\n=== --214345 measured features ({len(exts)}) ===")
for e in exts:
    kind = 'DIP ' if e.is_dip else 'PEAK'
    pxi = e.position_px if e.position_px is not None else -1
    print(f"  {kind}  wl={e.position:7.2f} nm  px={pxi:5d}  height={abs(e.height):.3f}")

print("\n=== HG reference features ===")
wl_ref = np.linspace(330, 800, 600)
ref_spd = get_reference_spectrum(ReferenceSource.HG, wl_ref)
exts_r = extract(ref_spd, wl_ref, position_px=None, max_count=40)
for e in exts_r:
    kind = 'DIP ' if e.is_dip else 'PEAK'
    pxi_gt = (e.position - c_gt) / slope_gt if slope_gt > 0 else 0
    print(f"  {kind}  ref={e.position:7.2f} nm  expected_px={pxi_gt:.0f}  h={abs(e.height):.3f}")
