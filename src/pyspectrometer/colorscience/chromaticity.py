"""CIE 1931 xy chromaticity diagram renderer.

Renders the spectral locus horseshoe with color-accurate fill, Planckian locus,
configurable illuminant reference points, and an optional current measured xy.

Architecture
------------
The expensive base diagram (gamut fill, spectral locus, Planckian locus, axes)
is built once per (width, height) and cached at module level. Reference points
and the live xy marker are drawn on a per-frame copy, keeping interactive
rendering fast.

Usage
-----
    from pyspectrometer.colorscience.chromaticity import render_chromaticity, RefPoint, STANDARD_REFS

    img = render_chromaticity(800, 400, xy_current=(0.31, 0.33), refs=STANDARD_REFS)
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:
    import colour

    _COLOUR_AVAILABLE = True
except ImportError:
    _COLOUR_AVAILABLE = False

try:
    import cv2

    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


# ---------------------------------------------------------------------------
# Reference point type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RefPoint:
    """A named chromaticity reference point for diagram annotation."""
    xy: tuple[float, float]
    label: str
    color: tuple[int, int, int]   # BGR


def illuminant_xy(name: str) -> tuple[float, float]:
    """Compute CIE 1931 xy chromaticity of a named colour-science illuminant."""
    sd = colour.SDS_ILLUMINANTS[name]
    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    wl = cmfs.wavelengths
    cmf_xyz = cmfs.values
    E = np.interp(wl, sd.wavelengths.astype(float), sd.values.astype(float),
                  left=0.0, right=0.0)
    X = float(np.trapezoid(E * cmf_xyz[:, 0], wl))
    Y = float(np.trapezoid(E * cmf_xyz[:, 1], wl))
    Z = float(np.trapezoid(E * cmf_xyz[:, 2], wl))
    s = X + Y + Z
    return (X / s, Y / s) if s > 1e-12 else (0.333, 0.333)


# Pre-built standard illuminant references (lazy-initialized on first import)
def _std(name: str, label: str, color: tuple) -> RefPoint:
    return RefPoint(xy=illuminant_xy(name), label=label, color=color)


# Standard references drawn by default on every diagram
STANDARD_REFS: tuple[RefPoint, ...] = ()  # populated below on first use
_STANDARD_REFS_READY = False


def _ensure_standard_refs() -> tuple[RefPoint, ...]:
    global STANDARD_REFS, _STANDARD_REFS_READY
    if not _STANDARD_REFS_READY and _COLOUR_AVAILABLE:
        STANDARD_REFS = (
            _std("D65", "D65", (0, 230, 230)),
            _std("D50", "D50", (100, 200, 255)),
        )
        _STANDARD_REFS_READY = True
    return STANDARD_REFS


# ---------------------------------------------------------------------------
# Viewport / coordinate mapping
# ---------------------------------------------------------------------------

# Chromaticity viewport bounds
_X_MIN, _X_MAX = 0.0, 0.82
_Y_MIN, _Y_MAX = 0.0, 0.92

# Module-level cache: (width, height) → base diagram (fill + locus + axes)
_BASE_CACHE: dict[tuple[int, int], np.ndarray] = {}

# Planckian locus color temperature samples (K)
_PLANCK_TEMPS = [
    1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500,
    5000, 5500, 6000, 6500, 7000, 8000, 10000, 15000, 25000,
]


def _to_pixel(x: float, y: float, margin: int, draw_w: int, draw_h: int) -> tuple[int, int]:
    px = margin + int((x - _X_MIN) / (_X_MAX - _X_MIN) * draw_w)
    py = margin + int((1.0 - (y - _Y_MIN) / (_Y_MAX - _Y_MIN)) * draw_h)
    return (px, py)


# ---------------------------------------------------------------------------
# Spectral data
# ---------------------------------------------------------------------------

def _spectral_locus_xy() -> tuple[np.ndarray, np.ndarray]:
    """CIE 1931 2° spectral locus: returns (wavelengths, xy array (N,2))."""
    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    wl, xyz = cmfs.wavelengths, cmfs.values
    total = xyz.sum(axis=1, keepdims=True)
    total = np.where(total < 1e-12, 1.0, total)
    return wl, xyz[:, :2] / total


def _planckian_locus_xy() -> np.ndarray:
    """Planckian locus xy via direct Planck function integration (CIE 1931 2°)."""
    h, c, k = 6.62607015e-34, 2.99792458e8, 1.380649e-23
    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    wl, xyz = cmfs.wavelengths, cmfs.values
    wl_m = wl * 1e-9

    points = []
    for T in _PLANCK_TEMPS:
        exp_arg = np.minimum(h * c / (wl_m * k * T), 700.0)
        B = wl_m ** (-5) / (np.exp(exp_arg) - 1.0 + 1e-300)
        X = float(np.trapezoid(B * xyz[:, 0], wl_m))
        Y = float(np.trapezoid(B * xyz[:, 1], wl_m))
        Z = float(np.trapezoid(B * xyz[:, 2], wl_m))
        s = X + Y + Z
        if s > 1e-30:
            points.append((X / s, Y / s))

    return np.array(points) if points else np.empty((0, 2))


# ---------------------------------------------------------------------------
# Color fill
# ---------------------------------------------------------------------------

def _wavelength_bgr(wl: float) -> tuple[int, int, int]:
    """Approximate BGR for a monochromatic wavelength (locus outline only)."""
    if wl < 380 or wl > 780:
        return (80, 80, 80)
    if wl < 450:
        t = (wl - 380) / 70.0
        return (255, int(20 * t), int(100 * (1 - t) + 180 * t))
    if wl < 490:
        t = (wl - 450) / 40.0
        return (255, int(220 * t), 0)
    if wl < 510:
        t = (wl - 490) / 20.0
        return (int(255 * (1 - t)), 255, 0)
    if wl < 560:
        t = (wl - 510) / 50.0
        return (0, 255, int(180 * t))
    if wl < 590:
        t = (wl - 560) / 30.0
        return (0, int(255 * (1 - t * 0.5)), 255)
    if wl < 640:
        t = (wl - 590) / 50.0
        return (0, int(128 * (1 - t)), 255)
    t = min(1.0, (wl - 640) / 80.0)
    return (0, 0, int(255 * (1.0 - 0.4 * t)))


def _xy_to_bgr(xc: np.ndarray, yc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """xy (Y=1) → sRGB BGR using the exact colour-science diagram fill algorithm.

    Replicates plot_chromaticity_diagram_colours "RGB" mode:
      xy → XYZ(Y=1) → colour.XYZ_to_sRGB → normalise_maximum(axis=-1) → clip[0,1]

    colour.XYZ_to_sRGB and colour.utilities.normalise_maximum are in colour core —
    no matplotlib required.  Falls back to a plain D65-matrix path if colour is absent.
    """
    y = np.maximum(yc, 1e-10)
    XYZ = np.stack([xc / y, np.ones_like(xc), np.maximum((1.0 - xc - yc) / y, 0.0)], axis=-1)

    if _COLOUR_AVAILABLE:
        with colour.domain_range_scale("1"):
            RGB = colour.XYZ_to_sRGB(XYZ)
    else:
        # Plain D65 sRGB matrix (fallback, no chromatic adaptation)
        M = np.array([[ 3.2406, -1.5372, -0.4986],
                      [-0.9689,  1.8758,  0.0415],
                      [ 0.0557, -0.2040,  1.0570]])
        lin = XYZ @ M.T
        RGB = np.power(np.clip(lin, 0.0, 1.0), 1.0 / 2.2)

    # normalise_maximum: divide each pixel by its max channel (makes max=1 per pixel)
    mx = np.maximum(RGB.max(axis=-1, keepdims=True), 1e-10)
    RGB = np.clip(RGB / mx, 0.0, 1.0)

    return (
        (RGB[:, 2] * 255).astype(np.uint8),
        (RGB[:, 1] * 255).astype(np.uint8),
        (RGB[:, 0] * 255).astype(np.uint8),
    )


def _draw_gamut_fill(img: np.ndarray, poly_px: np.ndarray, margin: int, draw_w: int, draw_h: int) -> None:
    """Fill spectral locus interior with color-accurate sRGB fill (vectorized)."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly_px], 255)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return

    xc = _X_MIN + (xs.astype(np.float32) - margin) / draw_w * (_X_MAX - _X_MIN)
    yc = _Y_MAX - (ys.astype(np.float32) - margin) / draw_h * (_Y_MAX - _Y_MIN)

    valid = yc > 1e-5
    xs, ys = xs[valid], ys[valid]
    xc, yc = xc[valid], yc[valid]
    if len(xs) == 0:
        return

    b_ch, g_ch, r_ch = _xy_to_bgr(xc, yc)
    img[ys, xs, 0] = b_ch
    img[ys, xs, 1] = g_ch
    img[ys, xs, 2] = r_ch


# ---------------------------------------------------------------------------
# Axes, locus, planckian drawing
# ---------------------------------------------------------------------------

def _draw_spectral_locus(img: np.ndarray, locus_px: list, wavelengths: np.ndarray) -> None:
    for i in range(len(locus_px) - 1):
        color = _wavelength_bgr(float(wavelengths[i]))
        cv2.line(img, locus_px[i], locus_px[i + 1], color, 2, cv2.LINE_AA)
    if len(locus_px) >= 2:
        cv2.line(img, locus_px[-1], locus_px[0], (160, 80, 180), 1, cv2.LINE_AA)


def _draw_planckian(img: np.ndarray, planck: np.ndarray, margin: int, draw_w: int, draw_h: int) -> None:
    if len(planck) < 2:
        return
    pts = [_to_pixel(xy[0], xy[1], margin, draw_w, draw_h) for xy in planck]
    for i in range(len(pts) - 1):
        cv2.line(img, pts[i], pts[i + 1], (200, 200, 160), 1, cv2.LINE_AA)


def _draw_axes(img: np.ndarray, margin: int, draw_w: int, draw_h: int) -> None:
    h, w = img.shape[:2]
    color = (100, 100, 100)
    font, fs = cv2.FONT_HERSHEY_SIMPLEX, 0.30

    for xv in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        px = margin + int((xv - _X_MIN) / (_X_MAX - _X_MIN) * draw_w)
        py_base = margin + draw_h
        cv2.line(img, (px, py_base), (px, py_base + 4), color, 1)
        cv2.putText(img, f"{xv:.1f}", (px - 7, py_base + 14), font, fs, color, 1)

    for yv in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        py = margin + int((1.0 - (yv - _Y_MIN) / (_Y_MAX - _Y_MIN)) * draw_h)
        cv2.line(img, (margin - 4, py), (margin, py), color, 1)
        cv2.putText(img, f"{yv:.1f}", (2, py + 4), font, fs, color, 1)

    cv2.putText(img, "x", (w // 2, h - 3), font, 0.38, color, 1)
    cv2.putText(img, "y", (3, h // 2), font, 0.38, color, 1)


def _draw_ref_point(
    img: np.ndarray,
    ref: RefPoint,
    margin: int,
    draw_w: int,
    draw_h: int,
    radius: int = 5,
) -> None:
    px, py = _to_pixel(ref.xy[0], ref.xy[1], margin, draw_w, draw_h)
    cv2.circle(img, (px, py), radius, ref.color, -1)
    cv2.circle(img, (px, py), radius, (255, 255, 255), 1)
    cv2.putText(
        img, ref.label, (px + radius + 2, py + 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, ref.color, 1, cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# Base diagram builder (cached per size)
# ---------------------------------------------------------------------------

def _build_base(width: int, height: int) -> np.ndarray:
    """Build the base diagram: dark background, gamut fill, locus, axes.

    Reference points and current xy are NOT in the base so they can change
    without invalidating the cache.
    """
    img = np.full((height, width, 3), 22, dtype=np.uint8)

    if not (_COLOUR_AVAILABLE and _CV2_AVAILABLE):
        if _CV2_AVAILABLE:
            cv2.putText(img, "colour-science not available",
                        (20, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        return img

    margin = 32
    draw_w, draw_h = width - 2 * margin, height - 2 * margin

    try:
        wavelengths, locus_xy = _spectral_locus_xy()
        locus_px = [_to_pixel(xy[0], xy[1], margin, draw_w, draw_h) for xy in locus_xy]
        poly_px = np.array(locus_px, dtype=np.int32)

        _draw_gamut_fill(img, poly_px, margin, draw_w, draw_h)

        planck = _planckian_locus_xy()
        _draw_planckian(img, planck, margin, draw_w, draw_h)

        _draw_spectral_locus(img, locus_px, wavelengths)
    except Exception as e:
        print(f"[CHROMA] Base diagram error: {e}")

    _draw_axes(img, margin, draw_w, draw_h)
    return img


def _get_base(width: int, height: int) -> np.ndarray:
    key = (width, height)
    if key not in _BASE_CACHE:
        print(f"[CHROMA] Building {width}×{height} base diagram…")
        _BASE_CACHE[key] = _build_base(width, height)
    return _BASE_CACHE[key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_chromaticity(
    width: int,
    height: int,
    xy_current: tuple[float, float] | None = None,
    refs: Sequence[RefPoint] | None = None,
) -> np.ndarray:
    """Render the CIE 1931 xy chromaticity diagram.

    The expensive base (gamut fill + locus + axes) is cached per size.
    Reference points and the live xy marker are composited on each call.

    Args:
        width, height: Output image dimensions in pixels.
        xy_current: Current measured xy, shown as a bright green dot.
        refs: Reference illuminant points to annotate. Defaults to D65 + D50.

    Returns:
        BGR image of shape (height, width, 3).
    """
    img = _get_base(width, height).copy()

    if not _CV2_AVAILABLE:
        return img

    margin = 32
    draw_w, draw_h = width - 2 * margin, height - 2 * margin

    # Draw reference points
    points = refs if refs is not None else _ensure_standard_refs()
    for ref in points:
        _draw_ref_point(img, ref, margin, draw_w, draw_h)

    # Current measurement (larger, bright)
    if xy_current is not None:
        px, py = _to_pixel(xy_current[0], xy_current[1], margin, draw_w, draw_h)
        cv2.circle(img, (px, py), 9, (0, 255, 80), -1)
        cv2.circle(img, (px, py), 9, (255, 255, 255), 2)
        label = f"x={xy_current[0]:.4f}  y={xy_current[1]:.4f}"
        cv2.putText(img, label, (px + 12, py + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 255, 180), 1, cv2.LINE_AA)

    return img
