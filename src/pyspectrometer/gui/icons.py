"""OpenCV primitive icon drawing for spectroscopy UI buttons.

All public functions have the signature:
    draw(image, x1, y1, w, h, color) -> None

Icons are drawn centered within the (x1, y1, w, h) bounds using only
cv2 primitives — no external fonts or assets required.
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _inner(x1: int, y1: int, w: int, h: int, frac: float = 0.22) -> tuple[int, int, int, int]:
    """Return (bx1, by1, bx2, by2) inset by frac of the smaller dimension."""
    m = max(2, int(min(w, h) * frac))
    return x1 + m, y1 + m, x1 + w - m - 1, y1 + h - m - 1


def _center(x1: int, y1: int, w: int, h: int) -> tuple[int, int]:
    return x1 + w // 2, y1 + h // 2


def _polyline(img: np.ndarray, pts: list[tuple[int, int]], c: tuple, closed: bool = False) -> None:
    arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [arr], closed, c, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Icon drawing functions
# ---------------------------------------------------------------------------

def _save(img: np.ndarray, x1: int, y1: int, w: int, h: int, c: tuple) -> None:
    """Floppy disk: outer body, shutter slot (top-right), label area."""
    bx1, by1, bx2, by2 = _inner(x1, y1, w, h)
    iw, ih = bx2 - bx1, by2 - by1
    cv2.rectangle(img, (bx1, by1), (bx2, by2), c, 1)
    # Shutter notch (solid block top-right)
    slot_w = max(2, iw * 2 // 5)
    slot_h = max(2, ih // 3)
    cv2.rectangle(img, (bx2 - slot_w, by1), (bx2, by1 + slot_h), c, -1)
    # Label area (outline, lower half)
    cv2.rectangle(img, (bx1 + 2, by1 + ih // 2), (bx2 - 2, by2 - 2), c, 1)


def _load(img: np.ndarray, x1: int, y1: int, w: int, h: int, c: tuple) -> None:
    """Folder icon with up-arrow inside."""
    bx1, by1, bx2, by2 = _inner(x1, y1, w, h, frac=0.18)
    iw, ih = bx2 - bx1, by2 - by1
    # Folder tab
    tab_w = max(3, iw // 3)
    tab_y = by1 - max(2, ih // 6)
    _polyline(img, [(bx1, by1), (bx1 + tab_w, by1), (bx1 + tab_w, tab_y),
                    (bx1 + tab_w * 2, tab_y), (bx1 + tab_w * 2, by1)], c)
    # Folder body
    cv2.rectangle(img, (bx1, by1), (bx2, by2), c, 1)
    # Up arrow
    cx = (bx1 + bx2) // 2
    cv2.arrowedLine(img, (cx, by2 - 2), (cx, by1 + 2), c, 1, tipLength=0.45, line_type=cv2.LINE_AA)


def _load_plus(img: np.ndarray, x1: int, y1: int, w: int, h: int, c: tuple) -> None:
    """Same as load, plus a + badge (add overlay / stack trace)."""
    _load(img, x1, y1, w, h, c)
    bx1, by1, bx2, by2 = _inner(x1, y1, w, h, frac=0.18)
    r = max(2, min(bx2 - bx1, by2 - by1) // 6)
    cx = bx2 - r - 1
    cy = by2 - r - 1
    cv2.circle(img, (cx, cy), r, c, 1, cv2.LINE_AA)
    hl = max(1, r * 2 // 3)
    cv2.line(img, (cx - hl // 2, cy), (cx + hl // 2, cy), c, 1)
    cv2.line(img, (cx, cy - hl // 2), (cx, cy + hl // 2), c, 1)


def _quit(img: np.ndarray, x1: int, y1: int, w: int, h: int, c: tuple) -> None:
    """× (close/quit)."""
    bx1, by1, bx2, by2 = _inner(x1, y1, w, h)
    cv2.line(img, (bx1, by1), (bx2, by2), c, 1, cv2.LINE_AA)
    cv2.line(img, (bx2, by1), (bx1, by2), c, 1, cv2.LINE_AA)


def _sensitivity(img: np.ndarray, x1: int, y1: int, w: int, h: int, c: tuple) -> None:
    """S-shaped spectral sensitivity curve."""
    bx1, by1, bx2, by2 = _inner(x1, y1, w, h)
    iw, ih = bx2 - bx1, by2 - by1
    n = max(12, iw)
    pts = []
    for i in range(n + 1):
        t = (i / n - 0.5) * 4.0
        s = 1.0 / (1.0 + np.exp(-t))          # sigmoid
        px = bx1 + int(i * iw / n)
        py = by2 - int(s * ih)
        pts.append((px, py))
    _polyline(img, pts, c)


def _avg(img: np.ndarray, x1: int, y1: int, w: int, h: int, c: tuple) -> None:
    """Three horizontal lines (averaging/integration)."""
    bx1, by1, bx2, by2 = _inner(x1, y1, w, h)
    ih = by2 - by1
    for i in range(3):
        y = by1 + i * ih // 2
        cv2.line(img, (bx1, y), (bx2, y), c, 1)


def _peak_hold(img: np.ndarray, x1: int, y1: int, w: int, h: int, c: tuple) -> None:
    """Peak shape with dashed hold line at base."""
    bx1, by1, bx2, by2 = _inner(x1, y1, w, h)
    iw, _ = bx2 - bx1, by2 - by1
    cx = (bx1 + bx2) // 2
    # Peak chevron
    cv2.line(img, (bx1, by2 - 2), (cx, by1 + 1), c, 1, cv2.LINE_AA)
    cv2.line(img, (cx, by1 + 1), (bx2, by2 - 2), c, 1, cv2.LINE_AA)
    # Dashed baseline
    for dx in range(0, iw, 4):
        if (dx // 4) % 2 == 0:
            x_a = min(bx1 + dx, bx2)
            x_b = min(bx1 + dx + 2, bx2)
            cv2.line(img, (x_a, by2 - 1), (x_b, by2 - 1), c, 1)


def _acc(img: np.ndarray, x1: int, y1: int, w: int, h: int, c: tuple) -> None:
    """Three stacked offset squares (accumulation)."""
    bx1, by1, bx2, by2 = _inner(x1, y1, w, h)
    iw, ih = bx2 - bx1, by2 - by1
    s = max(3, min(iw, ih) * 2 // 3)
    off = max(2, (min(iw, ih) - s) // 2)
    for i in range(3):
        ox = i * off
        oy = (2 - i) * off
        cv2.rectangle(img, (bx1 + ox, by1 + oy), (bx1 + ox + s, by1 + oy + s), c, 1)


def _dark(img: np.ndarray, x1: int, y1: int, w: int, h: int, c: tuple) -> None:
    """Dark reference: box with diagonal hatching."""
    bx1, by1, bx2, by2 = _inner(x1, y1, w, h)
    iw, ih = bx2 - bx1, by2 - by1
    cv2.rectangle(img, (bx1, by1), (bx2, by2), c, 1)
    # Hatch lines
    step = max(3, min(iw, ih) // 3)
    for k in range(-ih, iw, step):
        x_a = max(bx1, bx1 + k)
        y_a = by1 if k >= 0 else by1 - k
        x_b = min(bx2, bx1 + k + ih)
        y_b = by2 if (k + ih) <= iw else by1 + (iw - k)
        if x_a < x_b:
            dim = (c[0] // 3, c[1] // 3, c[2] // 3)
            cv2.line(img, (x_a, y_a), (x_b, y_b), dim, 1)


def _white(img: np.ndarray, x1: int, y1: int, w: int, h: int, c: tuple) -> None:
    """White reference: filled bright square with thin inner border."""
    bx1, by1, bx2, by2 = _inner(x1, y1, w, h)
    cv2.rectangle(img, (bx1, by1), (bx2, by2), c, -1)
    cv2.rectangle(img, (bx1 + 2, by1 + 2), (bx2 - 2, by2 - 2), (0, 0, 0), 1)


def _absorption(img: np.ndarray, x1: int, y1: int, w: int, h: int, c: tuple) -> None:
    """Absorption dip: flat line with V-dip in center."""
    bx1, by1, bx2, by2 = _inner(x1, y1, w, h)
    iw, ih = bx2 - bx1, by2 - by1
    cy = by1 + ih // 3
    cx = (bx1 + bx2) // 2
    dw = iw // 4
    cv2.line(img, (bx1, cy), (cx - dw, cy), c, 1, cv2.LINE_AA)
    cv2.line(img, (cx - dw, cy), (cx, by2 - 1), c, 1, cv2.LINE_AA)
    cv2.line(img, (cx, by2 - 1), (cx + dw, cy), c, 1, cv2.LINE_AA)
    cv2.line(img, (cx + dw, cy), (bx2, cy), c, 1, cv2.LINE_AA)


def _bars(img: np.ndarray, x1: int, y1: int, w: int, h: int, c: tuple) -> None:
    """Bar chart: 4 filled bars of varying heights."""
    bx1, by1, bx2, by2 = _inner(x1, y1, w, h)
    iw, ih = bx2 - bx1, by2 - by1
    n = 4
    heights = [0.45, 1.0, 0.65, 0.35]
    bw = max(1, (iw - n + 1) // n)
    for i in range(n):
        bx = bx1 + i * (bw + 1)
        bh = max(1, int(ih * heights[i]))
        cv2.rectangle(img, (bx, by2 - bh), (bx + bw - 1, by2), c, -1)


def _zoom_x(img: np.ndarray, x1: int, y1: int, w: int, h: int, c: tuple) -> None:
    """Horizontal double arrow ↔."""
    cx, cy = _center(x1, y1, w, h)
    r = min(w, h) // 2 - 2
    cv2.arrowedLine(img, (cx + 1, cy), (cx - r, cy), c, 1, tipLength=0.5, line_type=cv2.LINE_AA)
    cv2.arrowedLine(img, (cx - 1, cy), (cx + r, cy), c, 1, tipLength=0.5, line_type=cv2.LINE_AA)


def _zoom_y(img: np.ndarray, x1: int, y1: int, w: int, h: int, c: tuple) -> None:
    """Vertical double arrow ↕."""
    cx, cy = _center(x1, y1, w, h)
    r = min(w, h) // 2 - 2
    cv2.arrowedLine(img, (cx, cy + 1), (cx, cy - r), c, 1, tipLength=0.5, line_type=cv2.LINE_AA)
    cv2.arrowedLine(img, (cx, cy - 1), (cx, cy + r), c, 1, tipLength=0.5, line_type=cv2.LINE_AA)


def _lamp(img: np.ndarray, x1: int, y1: int, w: int, h: int, c: tuple) -> None:
    """Light bulb: circle + base cap."""
    cx, cy = _center(x1, y1, w, h)
    r = max(3, min(w, h) // 2 - 3)
    # Bulb
    cv2.circle(img, (cx, cy - 1), r - 1, c, 1, cv2.LINE_AA)
    # Base
    br = max(2, r // 2)
    cv2.line(img, (cx - br, cy + r), (cx + br, cy + r), c, 1)
    cv2.line(img, (cx - br, cy + r + 2), (cx + br, cy + r + 2), c, 1)
    # Cap
    cv2.line(img, (cx - br + 1, cy + r + 2), (cx - br + 1, cy + r - 1), c, 1)
    cv2.line(img, (cx + br - 1, cy + r + 2), (cx + br - 1, cy + r - 1), c, 1)


def _peaks(img: np.ndarray, x1: int, y1: int, w: int, h: int, c: tuple) -> None:
    """Peak triangle (detect peaks)."""
    bx1, by1, bx2, by2 = _inner(x1, y1, w, h)
    cx = (bx1 + bx2) // 2
    _polyline(img, [(bx1, by2), (cx, by1), (bx2, by2)], c, closed=True)


def _snap(img: np.ndarray, x1: int, y1: int, w: int, h: int, c: tuple) -> None:
    """Snap-to-peak: chevron tip with vertical snap line and tick."""
    bx1, by1, bx2, by2 = _inner(x1, y1, w, h)
    _, ih = bx2 - bx1, by2 - by1
    cx = (bx1 + bx2) // 2
    mid_y = by1 + ih // 3
    # Small peak chevron
    cv2.line(img, (bx1, mid_y), (cx, by1), c, 1, cv2.LINE_AA)
    cv2.line(img, (cx, by1), (bx2, mid_y), c, 1, cv2.LINE_AA)
    # Snap tick at tip
    cv2.line(img, (cx - 3, by1), (cx + 3, by1), c, 1)
    # Downward snap arrow
    cv2.arrowedLine(img, (cx, mid_y), (cx, by2 - 1), c, 1, tipLength=0.3, line_type=cv2.LINE_AA)


def _delta(img: np.ndarray, x1: int, y1: int, w: int, h: int, c: tuple) -> None:
    """Two vertical markers with delta arrow (peak separation)."""
    bx1, by1, bx2, by2 = _inner(x1, y1, w, h)
    iw = bx2 - bx1
    x_a = bx1 + iw // 3
    x_b = bx1 + 2 * iw // 3
    # Vertical markers
    cv2.line(img, (x_a, by1), (x_a, by2), c, 1)
    cv2.line(img, (x_b, by1), (x_b, by2), c, 1)
    # Double-headed arrow between them
    cy = (by1 + by2) // 2
    if x_b - x_a > 4:
        cv2.arrowedLine(img, (x_a + 1, cy), (x_b - 1, cy), c, 1, tipLength=0.4, line_type=cv2.LINE_AA)


def _clear(img: np.ndarray, x1: int, y1: int, w: int, h: int, c: tuple) -> None:
    """Trash can: lid + body + stripes."""
    bx1, by1, bx2, by2 = _inner(x1, y1, w, h)
    iw, ih = bx2 - bx1, by2 - by1
    lid_y = by1 + ih // 4
    cx = (bx1 + bx2) // 2
    # Handle
    hw = max(2, iw // 4)
    cv2.line(img, (cx - hw, lid_y), (cx - hw, by1 + 1), c, 1)
    cv2.line(img, (cx - hw, by1 + 1), (cx + hw, by1 + 1), c, 1)
    cv2.line(img, (cx + hw, by1 + 1), (cx + hw, lid_y), c, 1)
    # Lid line
    cv2.line(img, (bx1, lid_y), (bx2, lid_y), c, 1)
    # Body
    cv2.rectangle(img, (bx1 + 1, lid_y + 1), (bx2 - 1, by2), c, 1)
    # Stripes
    stripe_w = max(3, iw // 3)
    for i in range(1, 3):
        sx = bx1 + 1 + i * stripe_w
        if sx < bx2 - 1:
            cv2.line(img, (sx, lid_y + 2), (sx, by2 - 2), c, 1)


def _reference(img: np.ndarray, x1: int, y1: int, w: int, h: int, c: tuple) -> None:
    """Two offset Gaussian curves (measured vs reference)."""
    bx1, by1, bx2, by2 = _inner(x1, y1, w, h)
    iw, ih = bx2 - bx1, by2 - by1
    n = max(8, iw)
    for offset, peak_frac in [(0.3, 0.85), (0.6, 0.65)]:
        pts = []
        for i in range(n + 1):
            t = i / n
            v = np.exp(-((t - offset) ** 2) / 0.02) * peak_frac
            pts.append((bx1 + int(t * iw), by2 - int(v * ih)))
        _polyline(img, pts, c)


def _overlay(img: np.ndarray, x1: int, y1: int, w: int, h: int, c: tuple) -> None:
    """Two offset rectangles (raw overlay)."""
    bx1, by1, bx2, by2 = _inner(x1, y1, w, h)
    off = max(2, min(bx2 - bx1, by2 - by1) // 4)
    cv2.rectangle(img, (bx1 + off, by1 + off), (bx2, by2), c, 1)
    cv2.rectangle(img, (bx1, by1), (bx2 - off, by2 - off), c, 1)


def _gain(img: np.ndarray, x1: int, y1: int, w: int, h: int, c: tuple) -> None:
    """Horizontal slider with knob (gain control)."""
    bx1, by1, bx2, by2 = _inner(x1, y1, w, h)
    iw, ih = bx2 - bx1, by2 - by1
    cy = (by1 + by2) // 2
    # Track
    cv2.line(img, (bx1, cy), (bx2, cy), c, 1)
    # Knob (right of center)
    kx = bx1 + iw * 2 // 3
    kr = max(2, ih // 3)
    cv2.circle(img, (kx, cy), kr, c, 1)
    cv2.line(img, (kx, by1), (kx, by2), c, 1)


def _exposure(img: np.ndarray, x1: int, y1: int, w: int, h: int, c: tuple) -> None:
    """Shutter/aperture ring with 4 blades."""
    cx, cy = _center(x1, y1, w, h)
    r = max(3, min(w, h) // 2 - 2)
    cv2.circle(img, (cx, cy), r - 1, c, 1, cv2.LINE_AA)
    for deg in (0, 45, 90, 135):
        rad = np.radians(deg)
        dx, dy = int((r - 2) * np.cos(rad)), int((r - 2) * np.sin(rad))
        cv2.line(img, (cx - dx, cy - dy), (cx + dx, cy + dy), c, 1)


def _auto_gain(img: np.ndarray, x1: int, y1: int, w: int, h: int, c: tuple) -> None:
    """Gain slider + small dot marking auto mode."""
    _gain(img, x1, y1, w, h, c)
    bx1, by1 = _inner(x1, y1, w, h)[:2]
    cv2.circle(img, (bx1 + 2, by1 + 2), 2, c, -1)


def _auto_exposure(img: np.ndarray, x1: int, y1: int, w: int, h: int, c: tuple) -> None:
    """Aperture + small dot marking auto mode."""
    _exposure(img, x1, y1, w, h, c)
    bx1, by1 = _inner(x1, y1, w, h)[:2]
    cv2.circle(img, (bx1 + 2, by1 + 2), 2, c, -1)


def _eye(img: np.ndarray, x1: int, y1: int, w: int, h: int, c: tuple) -> None:
    """Eye (preview toggle): ellipse + pupil."""
    cx, cy = _center(x1, y1, w, h)
    rx = max(3, min(w, h) // 2 - 1)
    ry = max(2, rx // 2)
    cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, c, 1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), max(1, ry // 2), c, -1)


def _calibrate(img: np.ndarray, x1: int, y1: int, w: int, h: int, c: tuple) -> None:
    """Ruler with tick marks (wavelength calibration)."""
    bx1, by1, bx2, by2 = _inner(x1, y1, w, h)
    iw, ih = bx2 - bx1, by2 - by1
    cy = (by1 + by2) // 2
    cv2.line(img, (bx1, cy), (bx2, cy), c, 1)
    for i in range(6):
        tx = bx1 + i * iw // 5
        tk = ih // 3 if i % 2 == 0 else ih // 5
        cv2.line(img, (tx, cy - tk), (tx, cy + tk // 2), c, 1)


def _level(img: np.ndarray, x1: int, y1: int, w: int, h: int, c: tuple) -> None:
    """Level indicator: horizontal bar with centered bubble circle."""
    bx1, by1, bx2, by2 = _inner(x1, y1, w, h)
    cy = (by1 + by2) // 2
    cv2.line(img, (bx1, cy), (bx2, cy), c, 2)
    cx = (bx1 + bx2) // 2
    r = max(2, (by2 - by1) // 3)
    cv2.circle(img, (cx, cy), r, c, 1, cv2.LINE_AA)


def _reset(img: np.ndarray, x1: int, y1: int, w: int, h: int, c: tuple) -> None:
    """Circular arrow (reset/undo)."""
    cx, cy = _center(x1, y1, w, h)
    r = max(3, min(w, h) // 2 - 2)
    cv2.ellipse(img, (cx, cy), (r - 1, r - 1), -90, 30, 300, c, 1, cv2.LINE_AA)
    # Arrowhead at start of arc
    ax = cx + int((r - 1) * np.cos(np.radians(-90 + 30)))
    ay = cy + int((r - 1) * np.sin(np.radians(-90 + 30)))
    cv2.circle(img, (ax, ay), 2, c, -1)


def _pdf(img: np.ndarray, x1: int, y1: int, w: int, h: int, c: tuple) -> None:
    """Document sheet with folded corner (PDF report)."""
    bx1, by1, bx2, by2 = _inner(x1, y1, w, h, frac=0.18)
    iw, ih = bx2 - bx1, by2 - by1
    fold = max(2, min(iw, ih) // 4)
    # Body (L-shape: missing top-right where fold sits)
    cv2.line(img, (bx1, by1), (bx2 - fold, by1), c, 1)
    cv2.line(img, (bx1, by1), (bx1, by2), c, 1)
    cv2.line(img, (bx1, by2), (bx2, by2), c, 1)
    cv2.line(img, (bx2, by2), (bx2, by1 + fold), c, 1)
    # Fold diagonal
    cv2.line(img, (bx2 - fold, by1), (bx2, by1 + fold), c, 1)
    # Text lines
    ly = by1 + ih // 3
    gap = max(2, ih // 5)
    for i in range(3):
        y = ly + i * gap
        if y < by2 - 2:
            cv2.line(img, (bx1 + 2, y), (bx2 - fold - 2, y), c, 1)


# ---------------------------------------------------------------------------
# Registry and public API
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Callable] = {
    "save": _save,
    "load": _load,
    "load_plus": _load_plus,
    "quit": _quit,
    "sensitivity": _sensitivity,
    "avg": _avg,
    "peak_hold": _peak_hold,
    "acc": _acc,
    "dark": _dark,
    "white": _white,
    "absorption": _absorption,
    "bars": _bars,
    "zoom_x": _zoom_x,
    "zoom_y": _zoom_y,
    "lamp": _lamp,
    "peaks": _peaks,
    "snap": _snap,
    "delta": _delta,
    "clear": _clear,
    "reference": _reference,
    "overlay": _overlay,
    "gain": _gain,
    "exposure": _exposure,
    "auto_gain": _auto_gain,
    "auto_exposure": _auto_exposure,
    "eye": _eye,
    "calibrate": _calibrate,
    "level": _level,
    "reset": _reset,
    "pdf": _pdf,
}


def draw(
    image: np.ndarray,
    x1: int,
    y1: int,
    w: int,
    h: int,
    icon: str,
    color: tuple[int, int, int],
) -> bool:
    """Draw a named icon centered in (x1, y1, w, h).

    Returns True if the icon was recognized and drawn, False if unknown
    (caller can fall back to text rendering).
    """
    fn = _REGISTRY.get(icon)
    if fn is None:
        return False
    fn(image, x1, y1, w, h, color)
    return True


def known(icon: str) -> bool:
    """Return True if the icon name is registered."""
    return icon in _REGISTRY
