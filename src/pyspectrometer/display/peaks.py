"""Peak labels and vertical-line renderer for spectrum graphs.

Non-overlapping label placement strategy
-----------------------------------------
For each visible peak, candidate positions are generated above and below the
peak tip, left and right of the vertical line.  Each candidate is scored by:

1. Does it obscure another peak tip? (disqualified)
2. Does it overlap an already-placed label? (disqualified)
3. How many other peaks' vertical lines does it cross? (lower is better)
4. How much does the spectrum curve pass through the label box? (lower is better)

The best surviving candidate is selected.  When no valid position exists the
label is clamped to the nearest image edge.
"""

from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from ..core.spectrum import Peak, SpectrumData
from .peak_width import fwhm_nm
from .viewport import Viewport

_GAP = 2        # px gap between peak tip and label box
_PAD = 2        # inner padding inside label box
_ALPHA = 0.65   # label background opacity
_DELTA_SCALE = 0.75  # font scale for delta line relative to main label


def compute_placements(
    peaks: list[Peak],
    viewport: Viewport,
    width: int,
    height: int,
    spectrum_y: np.ndarray,
    font: int,
    font_scale: float,
    show_width: bool = False,
) -> list["_Placement"]:
    """Compute non-overlapping label placements for a list of peaks (or peak-like points).

    Used by both PeaksRenderer and MarkersRenderer for intelligent label placement.
    """
    return _place_all(
        peaks, viewport, width, height, spectrum_y,
        font, font_scale, show_width,
    )


@dataclass
class PeaksRenderer:
    """Renders peak wavelength labels and vertical indicator lines.

    Non-overlapping label placement; yellow box + black text; thick vertical lines.
    """

    show_labels: bool = True
    show_lines: bool = True
    show_width: bool = False
    font: int = cv2.FONT_HERSHEY_SIMPLEX
    font_scale: float = 0.45

    def render(self, image: np.ndarray, data: SpectrumData, viewport: Viewport) -> np.ndarray:
        """Render peaks onto *image* in-place.  Returns image for chaining."""
        if not data.peaks:
            return image
        height, width = image.shape[:2]
        spectrum_y = _spectrum_screen_y(data, viewport, width, height)
        placements = _place_all(
            data.peaks, viewport, width, height, spectrum_y,
            self.font, self.font_scale, self.show_width,
        )
        def get_fwhm(p: Peak) -> float | None:
            return fwhm_nm(p.index, data.intensity, data.wavelengths) if self.show_width else None
        if self.show_labels:
            draw_labels(image, placements, self.font, self.font_scale, get_fwhm)
        if self.show_lines:
            draw_lines(image, placements)
        return image


# ---------------------------------------------------------------------------
# Spectrum-y lookup (for overlap scoring)
# ---------------------------------------------------------------------------

def _spectrum_screen_y(
    data: SpectrumData, viewport: Viewport, width: int, height: int
) -> np.ndarray:
    """Per-column spectrum screen-y array; -1 where outside viewport."""
    out = np.full(width, -1.0, dtype=np.float32)
    n = len(data.intensity)
    for x in range(width):
        idx = max(0, min(n - 1, int(round(viewport.screen_x_to_data(x, width)))))
        fi = float(idx)
        if fi < viewport.x_start or fi > viewport.x_end:
            continue
        val = data.intensity[idx]
        if viewport.y_min <= val <= viewport.y_max:
            out[x] = float(viewport.intensity_to_screen_y(val, height))
    return out


# ---------------------------------------------------------------------------
# Placement types and pipeline
# ---------------------------------------------------------------------------

# (screen_x, label_x1, label_x2, box_y1, box_y2, peak, left_aligned)
_Placement = tuple[int, int, int, int, int, Peak, bool]


def _place_all(
    peaks: list[Peak],
    viewport: Viewport,
    width: int,
    height: int,
    spectrum_y: np.ndarray,
    font: int,
    font_scale: float,
    show_width: bool = False,
) -> list[_Placement]:
    visible = _visible_peaks(peaks, viewport, width, height)
    other_sx = [sx for sx, _, _, _ in visible]
    placements: list[_Placement] = []
    delta_scale = font_scale * _DELTA_SCALE if show_width else font_scale

    for sx, peak_y, _, peak in visible:
        (tw, th), _ = cv2.getTextSize(f"{peak.wavelength:.0f}", font, font_scale, 1)
        box_w = tw + _PAD * 2
        if show_width:
            (_, th2), _ = cv2.getTextSize("d99.9", font, delta_scale, 1)
            box_h = th + th2 + 4 + _PAD * 2
        else:
            box_h = th + _PAD * 2
        result = _best_position(sx, peak_y, box_w, box_h, width, height,
                                visible, other_sx, placements, spectrum_y)
        if result is not None:
            x1, x2, y1, y2, left = result
            placements.append((sx, x1, x2, y1, y2, peak, left))

    return placements


def _visible_peaks(
    peaks: list[Peak], viewport: Viewport, width: int, height: int
) -> list[tuple[int, int, float, Peak]]:
    out = []
    for p in peaks:
        fi = float(p.index)
        if fi < viewport.x_start or fi > viewport.x_end:
            continue
        if p.intensity < viewport.y_min or p.intensity > viewport.y_max:
            continue
        sx = viewport.data_x_to_screen(fi, width)
        sy = viewport.intensity_to_screen_y(p.intensity, height)
        out.append((sx, sy, p.wavelength, p))
    out.sort(key=lambda t: t[0])
    return out


def _best_position(
    sx: int,
    peak_y: int,
    box_w: int,
    box_h: int,
    width: int,
    height: int,
    visible: list,
    other_sx: list[int],
    placements: list[_Placement],
    spectrum_y: np.ndarray,
) -> tuple[int, int, int, int, bool] | None:
    candidates = []

    for attempt in range(100):
        for above in (True, False):
            y1, y2 = _vert(peak_y, box_h, attempt, above)
            if y1 < 0 or y2 > height:
                continue
            for left in (True, False):
                x1, x2 = (sx - box_w, sx) if left else (sx, sx + box_w)
                if x1 < 0 or x2 > width:
                    continue
                if _hits_peak(x1, x2, y1, y2, visible):
                    continue
                if _hits_label(x1, x2, y1, y2, placements):
                    continue
                crossed = sum(1 for ox in other_sx if ox != sx and x1 <= ox <= x2)
                overlap = _curve_overlap(x1, x2, y1, y2, spectrum_y)
                candidates.append((x1, x2, y1, y2, crossed, overlap, left))

    if not candidates:
        y1 = max(0, min(peak_y - _GAP - box_h, height - box_h))
        x1 = max(0, min(sx, width - box_w))
        x2, y2 = x1 + box_w, y1 + box_h
        return (x1, x2, y1, y2, True) if x2 <= width and y2 <= height else None

    x1, x2, y1, y2, _, _, left = min(candidates, key=lambda c: (c[4], c[5]))
    return x1, x2, y1, y2, left


def _vert(peak_y: int, box_h: int, attempt: int, above: bool) -> tuple[int, int]:
    offset = attempt * (box_h + _GAP) if attempt > 0 else 0
    if above:
        y2 = peak_y - _GAP - offset
        return y2 - box_h, y2
    y1 = peak_y + _GAP + offset
    return y1, y1 + box_h


def _hits_peak(x1: int, x2: int, y1: int, y2: int, visible: list) -> bool:
    return any(x1 <= osx <= x2 and y1 <= opy <= y2 for osx, opy, _, _ in visible)


def _hits_label(x1: int, x2: int, y1: int, y2: int, placed: list[_Placement]) -> bool:
    for _, px1, px2, py1, py2, _, _ in placed:
        if x2 <= px1 or x1 >= px2:
            continue
        if y2 + _GAP <= py1 or y1 >= py2 + _GAP:
            continue
        return True
    return False


def _curve_overlap(x1: int, x2: int, y1: int, y2: int, spectrum_y: np.ndarray) -> float:
    return sum(
        1.0
        for sx in range(max(0, x1), min(len(spectrum_y), x2 + 1))
        if 0 <= spectrum_y[sx] and y1 <= spectrum_y[sx] <= y2
    )


# ---------------------------------------------------------------------------
# Drawing (shared by peaks and markers: yellow box labels, thick vertical lines)
# ---------------------------------------------------------------------------

def draw_labels(
    image: np.ndarray,
    placements: list[_Placement],
    font: int,
    font_scale: float,
    get_fwhm: Callable[[Peak], float | None],
) -> None:
    """Draw wavelength labels (yellow box + black text) and optional FWHM line.

    get_fwhm(peak) returns FWHM in nm or None to skip delta line.
    """
    width = image.shape[1]
    delta_scale = font_scale * _DELTA_SCALE
    for sx, x1, x2, y1, y2, peak, left in placements:
        label = f"{peak.wavelength:.0f}"
        rx1, rx2 = max(0, x1), min(x2, width)
        if rx1 < rx2 and y1 < y2:
            roi = image[y1:y2, rx1:rx2]
            bg = roi.copy()
            bg[:] = (0, 255, 255)
            cv2.addWeighted(bg, _ALPHA, roi, 1 - _ALPHA, 0, roi)
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
        tx = x1 + _PAD if left else x2 - _PAD - tw
        tx = max(0, min(tx, width - tw))
        wl_baseline = y1 + _PAD + th
        cv2.putText(image, label, (tx, wl_baseline), font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
        fwhm = get_fwhm(peak)
        if fwhm is not None:
            delta_label = f"d{fwhm:.1f}"
            (dw, dh), _ = cv2.getTextSize(delta_label, font, delta_scale, 1)
            dx = x1 + _PAD if left else x2 - _PAD - dw
            dx = max(0, min(dx, width - dw))
            cv2.putText(
                image, delta_label, (dx, wl_baseline + 2 + dh),
                font, delta_scale, (0, 0, 0), 1, cv2.LINE_AA,
            )


def draw_lines(
    image: np.ndarray,
    placements: list[_Placement],
    y_start: int = 0,
    y_end: int | None = None,
) -> None:
    """Draw thick black vertical lines at each placement screen-x.

    y_start/y_end: vertical range (default 0 to image height). Used by markers on waterfall.
    """
    height = image.shape[0]
    end = height if y_end is None else y_end
    for sx, *_ in placements:
        cv2.line(image, (sx, y_start), (sx, end), (0, 0, 0), 2)
