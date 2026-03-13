"""Color-swatch model, rendering, and delta-E for Color Science mode.

A swatch stores one color measurement (XYZ, L*a*b*, mode, spectrum) and can
be compared against another using CIE 1976 ΔE*.  Up to GRID_COLS×GRID_ROWS
swatches are held in a grid rendered into a graph-area image.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

try:
    import colour
    _COLOUR = True
except ImportError:
    _COLOUR = False

GRID_COLS = 5
GRID_ROWS = 3

_MODE_LABELS = {"illumination": "I", "reflectance": "R", "transmittance": "T"}


@dataclass
class ColorSwatch:
    """One stored color measurement."""

    X: float
    Y: float
    Z: float
    L: float
    a: float
    b: float
    mode: str          # "I", "T", or "R"
    wavelengths: np.ndarray
    spectrum: np.ndarray
    label: str = ""
    selected: bool = False

    @property
    def lab(self) -> tuple[float, float, float]:
        return (self.L, self.a, self.b)

    @property
    def bgr(self) -> tuple[int, int, int]:
        return xyz_to_display_bgr(self.X, self.Y, self.Z)


# ---------------------------------------------------------------------------
# Color conversion
# ---------------------------------------------------------------------------

def xyz_to_display_bgr(X: float, Y: float, Z: float) -> tuple[int, int, int]:
    """XYZ → gamut-clipped sRGB for display, returned as BGR."""
    try:
        if _COLOUR:
            with colour.domain_range_scale("1"):
                rgb = np.asarray(colour.XYZ_to_sRGB(np.array([X, Y, Z]) / 100.0),
                                 dtype=float)
        else:
            xyz = np.array([X, Y, Z]) / 100.0
            M = np.array([[ 3.2406, -1.5372, -0.4986],
                          [-0.9689,  1.8758,  0.0415],
                          [ 0.0557, -0.2040,  1.0570]])
            lin = xyz @ M.T
            rgb = np.power(np.clip(lin, 0.0, 1.0), 1.0 / 2.2)
        rgb = np.clip(rgb, 0.0, 1.0)
        return (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
    except Exception:
        return (50, 50, 50)


# ---------------------------------------------------------------------------
# Delta-E
# ---------------------------------------------------------------------------

def delta_e_cie76(
    lab1: tuple[float, float, float],
    lab2: tuple[float, float, float],
) -> float:
    """CIE 1976 ΔE* — Euclidean distance in CIELAB."""
    return float(sum((a - b) ** 2 for a, b in zip(lab1, lab2)) ** 0.5)


# ---------------------------------------------------------------------------
# Preview-strip renderers
# ---------------------------------------------------------------------------

def render_color_preview(
    width: int,
    height: int,
    xyz: tuple[float, float, float] | None,
    info_lines: list[str] | None = None,
) -> np.ndarray:
    """Solid color fill from XYZ for the preview strip, with optional text."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = xyz_to_display_bgr(*xyz) if xyz is not None else (20, 20, 20)

    for i, line in enumerate(info_lines or []):
        y = 15 + i * 18
        cv2.putText(img, line, (9, y + 1), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def render_spectrum_strip(
    width: int,
    height: int,
    wavelengths: np.ndarray,
    spectrum: np.ndarray,
    wl_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Wavelength-colored spectrum bar + white line for the preview strip.

    wl_range: optional (wl_min, wl_max) to show only that wavelength window
    (e.g. (400, 750) for the visible range).  Defaults to the full spectrum.
    """
    from ..utils.color import rgb_to_bgr, wavelength_to_rgb

    bar = np.zeros((height, width, 3), dtype=np.uint8)
    n = len(spectrum)
    if n < 2:
        return bar

    # Crop to wavelength range when requested
    if wl_range is not None:
        wl_min, wl_max = wl_range
        mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
        if np.any(mask):
            wavelengths = wavelengths[mask]
            spectrum    = spectrum[mask]
            n = len(spectrum)
            if n < 2:
                return bar

    for x in range(width):
        idx = int(x * (n - 1) / max(width - 1, 1))
        idx = max(0, min(n - 1, idx))
        intensity = float(spectrum[idx])
        wl = round(float(wavelengths[idx]))
        bgr = rgb_to_bgr(wavelength_to_rgb(wl))
        bar[:, x] = tuple(int(c * intensity) for c in bgr)

    # Spectrum line on top
    pts = []
    for x in range(width):
        idx = int(x * (n - 1) / max(width - 1, 1))
        idx = max(0, min(n - 1, idx))
        val = float(spectrum[idx])
        sy = height - 1 - int(val * (height - 1))
        pts.append((x, sy))
    if len(pts) >= 2:
        cv2.polylines(bar, [np.array(pts, np.int32)], False,
                      (255, 255, 255), 1, cv2.LINE_AA)
    return bar


# ---------------------------------------------------------------------------
# Swatch grid renderer
# ---------------------------------------------------------------------------

def render_swatch_grid(
    width: int,
    height: int,
    swatches: list[ColorSwatch],
    current_xyz_lab: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
    current_wavelengths: np.ndarray | None = None,
    current_spectrum: np.ndarray | None = None,
    cols: int = GRID_COLS,
    rows: int = GRID_ROWS,
) -> np.ndarray:
    """Render the swatch grid with stored swatches + one pending current slot."""
    img = np.full((height, width, 3), 25, dtype=np.uint8)

    sw = width // cols
    sh = height // rows

    for i, swatch in enumerate(swatches[: cols * rows]):
        _draw_cell(img, (i % cols) * sw, (i // cols) * sh, sw, sh, swatch)

    # Pending slot for the current measurement
    n = len(swatches)
    if n < cols * rows and current_xyz_lab is not None:
        (X, Y, Z), (L, a, b) = current_xyz_lab
        pending = ColorSwatch(
            X=X, Y=Y, Z=Z, L=L, a=a, b=b, mode="+",
            wavelengths=current_wavelengths if current_wavelengths is not None else np.array([]),
            spectrum=current_spectrum if current_spectrum is not None else np.array([]),
        )
        _draw_cell(img, (n % cols) * sw, (n // cols) * sh, sw, sh,
                   pending, is_pending=True)

    # ΔE overlay when exactly 2 swatches are selected
    sel = [s for s in swatches if s.selected]
    if len(sel) == 2:
        de = delta_e_cie76(sel[0].lab, sel[1].lab)
        text = f"dE76 = {de:.2f}"
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cx = (width - tw) // 2
        cv2.putText(img, text, (cx + 1, height - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, (cx, height - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    return img


_FONT  = cv2.FONT_HERSHEY_SIMPLEX
_PLAIN = cv2.FONT_HERSHEY_PLAIN


def _put_shadowed(
    img: np.ndarray,
    text: str,
    x: int, y: int,
    scale: float,
    color: tuple[int, int, int],
    font: int = _PLAIN,
) -> None:
    """Draw text with a dark shadow for legibility on any background."""
    cv2.putText(img, text, (x + 1, y + 1), font, scale, (0, 0, 0),   2, cv2.LINE_AA)
    cv2.putText(img, text, (x,     y),     font, scale, color,        1, cv2.LINE_AA)


def _draw_cell(
    img: np.ndarray,
    x0: int, y0: int,
    sw: int, sh: int,
    swatch: ColorSwatch,
    is_pending: bool = False,
) -> None:
    """Draw one cell of the swatch grid with XYZ, L*a*b*, mode badge, mini-spectrum."""
    color_h = max(4, int(sh * 0.52))
    spec_h  = sh - color_h

    # ── Solid color fill (top) ────────────────────────────────────────────────
    cv2.rectangle(img, (x0, y0), (x0 + sw - 1, y0 + color_h - 1), swatch.bgr, -1)

    # ── Mini spectrum (bottom) ────────────────────────────────────────────────
    if len(swatch.spectrum) > 1 and spec_h > 3 and sw > 4:
        _mini_spectrum(img, x0, y0 + color_h, sw - 1, spec_h - 1,
                       swatch.wavelengths, swatch.spectrum)
    else:
        cv2.rectangle(img, (x0, y0 + color_h), (x0 + sw - 1, y0 + sh - 1),
                      (10, 10, 10), -1)

    # ── Cell border ───────────────────────────────────────────────────────────
    border = (0, 255, 255) if swatch.selected else (0, 200, 0) if is_pending else (70, 70, 70)
    thick  = 2 if (swatch.selected or is_pending) else 1
    cv2.rectangle(img, (x0, y0), (x0 + sw - 1, y0 + sh - 1), border, thick)

    # ── Semi-transparent dark band for text readability ───────────────────────
    band_h = min(44, color_h - 2)
    if band_h > 0:
        roi = img[y0 + color_h - band_h : y0 + color_h, x0 : x0 + sw - 1]
        dark = roi.copy(); dark[:] = (0, 0, 0)
        cv2.addWeighted(dark, 0.55, roi, 0.45, 0, roi)

    # ── Mode badge + label (top of cell) ─────────────────────────────────────
    badge_char  = ">" if is_pending else swatch.mode
    badge_color = (200, 220, 200) if is_pending else (220, 220, 220)
    _put_shadowed(img, badge_char, x0 + 3, y0 + 13, 0.42, badge_color)
    if swatch.label:
        _put_shadowed(img, swatch.label, x0 + 20, y0 + 13, 0.8, (180, 180, 180), _PLAIN)

    # ── XYZ row ───────────────────────────────────────────────────────────────
    xyz_txt = f"XYZ:{swatch.X:.0f},{swatch.Y:.0f},{swatch.Z:.0f}"
    _put_shadowed(img, xyz_txt, x0 + 2, y0 + color_h - 20, 0.82,
                  (180, 255, 180) if is_pending else (200, 230, 200), _PLAIN)

    # ── L*a*b* row ────────────────────────────────────────────────────────────
    lab_txt = f"LAB:{swatch.L:.0f},{swatch.a:+.0f},{swatch.b:+.0f}"
    _put_shadowed(img, lab_txt, x0 + 2, y0 + color_h - 4, 0.82,
                  (180, 255, 255) if is_pending else (255, 255, 200), _PLAIN)


def _mini_spectrum(
    img: np.ndarray,
    x0: int, y0: int,
    w: int, h: int,
    wavelengths: np.ndarray,
    spectrum: np.ndarray,
) -> None:
    """Tiny wavelength-colored filled spectrum + white line."""
    from ..utils.color import rgb_to_bgr, wavelength_to_rgb

    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (5, 5, 5), -1)
    n = len(spectrum)
    if n < 2 or w < 2 or h < 2:
        return

    mx = float(np.max(spectrum))
    if mx < 1e-10:
        return

    pts = []
    for px in range(w):
        idx = int(px * (n - 1) / max(w - 1, 1))
        idx = max(0, min(n - 1, idx))
        val = float(spectrum[idx]) / mx
        sy  = y0 + h - 1 - int(val * (h - 2))
        pts.append((x0 + px, sy))
        if len(wavelengths) > idx:
            bgr = rgb_to_bgr(wavelength_to_rgb(round(float(wavelengths[idx]))))
            cv2.line(img, (x0 + px, y0 + h - 1), (x0 + px, sy),
                     tuple(int(c * val * 0.8) for c in bgr), 1)

    if len(pts) >= 2:
        cv2.polylines(img, [np.array(pts, np.int32)], False,
                      (180, 180, 180), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Hit-testing
# ---------------------------------------------------------------------------

def swatch_index_at(
    x: int,
    rel_y: int,
    width: int,
    height: int,
    count: int,
    cols: int = GRID_COLS,
    rows: int = GRID_ROWS,
) -> int | None:
    """Return swatch index for a click at (x, rel_y) in the grid, or None."""
    sw = max(1, width // cols)
    sh = max(1, height // rows)
    col, row = x // sw, rel_y // sh
    if col >= cols or row >= rows:
        return None
    idx = row * cols + col
    return idx if idx < count else None
