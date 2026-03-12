"""Debug: print extracted features from HG and FL12 reference spectra."""
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


for src_name in ['HG', 'FL12']:
    src = ReferenceSource[src_name]
    wl_ref = np.linspace(330, 800, 600)
    ref_spd = get_reference_spectrum(src, wl_ref)
    exts = extract(ref_spd, wl_ref, position_px=None, max_count=40)
    print(f"\n=== {src_name} reference features ({len(exts)}) ===")
    for e in exts:
        kind = 'DIP ' if e.is_dip else 'PEAK'
        print(f"  {kind}  wl={e.position:7.2f} nm  height={abs(e.height):.3f}")

print("\n=== --213319 measured features (GT calibrated) ===")
intensity, wl_gt = load('output/Spectrum-20260311--213319.csv')
n = len(intensity)
px_idx = np.arange(n, dtype=np.intp)
exts = extract(intensity, wl_gt, position_px=px_idx, max_count=40)
for e in exts:
    kind = 'DIP ' if e.is_dip else 'PEAK'
    pxi = e.position_px if e.position_px is not None else -1
    print(f"  {kind}  wl={e.position:7.2f} nm  px={pxi:5d}  height={abs(e.height):.3f}")
