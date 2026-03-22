"""Spectrum curve and wavelength-colored fill renderer.

Delegates the actual polyline drawing to the existing ``render_polyline_overlay``
in ``overlay_utils`` — no duplication.  This module owns only what is unique to
the main spectrum view: smoothing, wavelength-colored fill, selection highlight,
and the intensity grid.

Composable: renders onto any BGR image, so the same renderer works for the main
graph, a waterfall strip, or a colour-swatch thumbnail.

Typical usage
-------------
Main display (full-resolution, colored fill)::

    renderer = SpectrogramRenderer(filled=True, show_grid=True)
    renderer.render(graph, data, viewport)

Colour-swatch thumbnail (smoothed, curve only)::

    renderer = SpectrogramRenderer(smoothing=3)
    mini = np.zeros((32, 80, 3), dtype=np.uint8)
    vp = Viewport(x_end=float(len(data.intensity) - 1))
    renderer.render(mini, data, vp)
"""

from dataclasses import dataclass

import cv2
import numpy as np

from ..core.spectrum import SpectrumData
from ..utils.color import rgb_to_bgr, wavelength_to_rgb
from .overlay_utils import render_polyline_overlay
from .viewport import Viewport


@dataclass
class SpectrogramRenderer:
    """Renders a spectrum curve and optional wavelength-colored fill.

    Attributes
    ----------
    filled:
        Draw wavelength-colored vertical strips from baseline to curve.
        The curve is redrawn on top so it stays visible over the fill.
    show_curve:
        Draw the spectrum polyline (delegates to render_polyline_overlay).
    show_grid:
        Overlay subtle horizontal lines at 25 / 50 / 75 % intensity.
    smoothing:
        Half-window of a box filter applied to intensity before rendering
        (0 = off).  Reduces jaggedness for small viewports like swatch thumbnails.
    curve_color:
        BGR colour of the spectrum polyline.
    curve_thickness:
        Polyline thickness in pixels.
    selected_index:
        Data index to highlight with a white vertical bar (used in bar mode).
    """

    filled: bool = False
    show_curve: bool = True
    show_grid: bool = False
    smoothing: int = 0
    curve_color: tuple[int, int, int] = (0, 0, 0)
    curve_thickness: int = 1
    selected_index: int | None = None

    def render(self, image: np.ndarray, data: SpectrumData, viewport: Viewport) -> np.ndarray:
        """Render spectrum onto *image* in-place.  Returns image for chaining."""
        intensity = _smooth(data.intensity, self.smoothing)
        if len(intensity) == 0:
            return image

        if self.show_grid:
            _draw_intensity_grid(image)

        if self.filled:
            _draw_fill(image, intensity, data.wavelengths, viewport)
            _draw_selected(image, intensity, self.selected_index, viewport)

        if self.show_curve:
            render_polyline_overlay(
                image, intensity, self.curve_color,
                thickness=self.curve_thickness, viewport=viewport,
            )

        return image

    def render_bar(self, image: np.ndarray, data: SpectrumData, viewport: Viewport) -> np.ndarray:
        """Render a wavelength-colored intensity preview bar into *image*.

        Each column is colored by its wavelength and brightness-scaled by
        intensity — a horizontal strip showing the spectrum as a glowing bar.
        Viewport-aware: respects the current horizontal zoom.
        """
        height, width = image.shape[:2]
        intensity = _smooth(data.intensity, self.smoothing)
        n = len(intensity)
        n_wl = len(data.wavelengths)
        if n == 0:
            return image

        image[:] = 0
        for x in range(width):
            data_x = viewport.screen_x_to_data(x, width)
            idx = max(0, min(n - 1, int(round(data_x))))
            wl_idx = min(idx, n_wl - 1)
            bgr = rgb_to_bgr(wavelength_to_rgb(round(float(data.wavelengths[wl_idx]))))
            val = max(0.0, min(1.0, float(intensity[idx])))
            image[:, x] = tuple(int(c * val) for c in bgr)
        return image


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _smooth(raw: np.ndarray, half_window: int) -> np.ndarray:
    """Box-filter smoothing; returns float32 (no-op when half_window=0)."""
    intensity = np.asarray(raw, dtype=np.float32)
    if half_window <= 0:
        return intensity
    size = 2 * half_window + 1
    kernel = np.ones(size, dtype=np.float32) / size
    return np.convolve(intensity, kernel, mode="same")


def _draw_fill(
    image: np.ndarray,
    intensity: np.ndarray,
    wavelengths: np.ndarray,
    viewport: Viewport,
) -> None:
    """One wavelength-colored vertical strip per screen column."""
    height, width = image.shape[:2]
    n = len(intensity)
    n_wl = len(wavelengths)
    for x in range(width):
        data_x = viewport.screen_x_to_data(x, width)
        idx = max(0, min(n - 1, int(round(data_x))))
        if float(idx) < viewport.x_start or float(idx) > viewport.x_end:
            continue
        wl_idx = min(idx, n_wl - 1)
        bgr = rgb_to_bgr(wavelength_to_rgb(round(float(wavelengths[wl_idx]))))
        sy = viewport.intensity_to_screen_y(float(intensity[idx]), height)
        cv2.line(image, (x, height - 1), (x, sy), bgr, 1)


def _draw_selected(
    image: np.ndarray,
    intensity: np.ndarray,
    sel_idx: int | None,
    viewport: Viewport,
) -> None:
    """Highlight selected wavelength column with a white vertical bar."""
    if sel_idx is None or not (0 <= sel_idx < len(intensity)):
        return
    height, width = image.shape[:2]
    sx = viewport.data_x_to_screen(float(sel_idx), width)
    if 0 <= sx < width:
        sy = viewport.intensity_to_screen_y(float(intensity[sel_idx]), height)
        cv2.line(image, (sx, height - 1), (sx, sy), (255, 255, 255), 2)


def _draw_intensity_grid(image: np.ndarray) -> None:
    """Subtle horizontal lines at 25 / 50 / 75 % intensity."""
    height, width = image.shape[:2]
    for pct in (0.25, 0.50, 0.75):
        y = int(height * (1.0 - pct))
        cv2.line(image, (0, y), (width - 1, y), (200, 200, 200), 1)
