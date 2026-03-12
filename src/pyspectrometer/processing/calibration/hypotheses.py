"""Generate sequential peak-match hypotheses."""

from __future__ import annotations

from dataclasses import dataclass

from .detect_peaks import MeasuredPeak, ReferencePeak


@dataclass
class Match:
    """Single peak match: measured peak index -> reference peak index."""

    idx_measured: int
    idx_reference: int


@dataclass
class Hypothesis:
    """A hypothesis: list of (measured_idx, reference_idx) matches."""

    matches: list[Match]

    def to_cal_points(
        self,
        measured: list[MeasuredPeak],
        reference: list[ReferencePeak],
    ) -> list[tuple[int, float]]:
        """Convert to (pixel, wavelength) calibration points."""
        return [
            (measured[m.idx_measured].pixel, reference[m.idx_reference].wavelength)
            for m in self.matches
        ]


def generate_sequential_hypotheses(
    measured: list[MeasuredPeak],
    reference: list[ReferencePeak],
    *,
    min_matches: int = 4,
    tolerance_nm: float = 35.0,
    max_hypotheses: int = 200,
    initial_wavelengths: list | None = None,
) -> list[Hypothesis]:
    """Generate sequential match hypotheses."""
    import numpy as np

    n_meas = len(measured)
    n_ref = len(reference)
    if n_meas < min_matches or n_ref < min_matches:
        return []

    wl_arr = np.asarray(initial_wavelengths) if initial_wavelengths is not None else None
    n = len(wl_arr) if wl_arr is not None else 0
    if n > 1:

        def pixel_to_approx_wl(px: int) -> float:
            return float(np.interp(px, np.arange(n), wl_arr))
    else:
        px_lo = measured[0].pixel
        px_hi = measured[-1].pixel
        wl_lo = reference[0].wavelength
        wl_hi = reference[-1].wavelength
        span_px = max(px_hi - px_lo, 1)
        span_wl = wl_hi - wl_lo

        def pixel_to_approx_wl(px: int) -> float:
            return wl_lo + (px - px_lo) / span_px * span_wl

    hypotheses: list[Hypothesis] = []

    def recurse(meas_idx: int, ref_idx: int, path: list[Match]) -> None:
        if len(hypotheses) >= max_hypotheses:
            return
        if meas_idx >= n_meas:
            if len(path) >= min_matches:
                hypotheses.append(Hypothesis(matches=path.copy()))
            return
        if ref_idx >= n_ref:
            return

        approx_wl = pixel_to_approx_wl(measured[meas_idx].pixel)
        candidates = [
            (j, abs(reference[j].wavelength - approx_wl))
            for j in range(ref_idx, n_ref)
            if abs(reference[j].wavelength - approx_wl) <= tolerance_nm
        ]
        candidates.sort(key=lambda x: x[1])
        for j, _ in candidates:
            path.append(Match(meas_idx, j))
            recurse(meas_idx + 1, j + 1, path)
            path.pop()

        recurse(meas_idx + 1, ref_idx, path)
        if ref_idx + 1 < n_ref:
            recurse(meas_idx, ref_idx + 1, path)

    recurse(0, 0, [])

    seen: set[tuple[tuple[int, int], ...]] = set()
    unique: list[Hypothesis] = []
    for h in hypotheses:
        key = tuple((m.idx_measured, m.idx_reference) for m in h.matches)
        if key not in seen:
            seen.add(key)
            unique.append(h)

    return unique
