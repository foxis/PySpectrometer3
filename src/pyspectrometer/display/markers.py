"""Vertical marker-line renderer for waterfall and spectrum graphs.

Markers are user-placed vertical lines identified by a data-array index.
Each line is annotated with its wavelength value; when show_width is True and
the marker is on a peak (local maximum), FWHM is shown below.

Composable: renders onto any BGR image that shares the same Viewport.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .peak_width import fwhm_nm, is_local_max
from .viewport import Viewport

_DELTA_SCALE = 0.75


@dataclass
class MarkersRenderer:
    """Renders vertical marker lines with wavelength labels.

    Attributes
    ----------
    color:
        BGR colour for idle marker lines and labels.
    drag_color:
        BGR colour used for the line currently being dragged.
    show_width:
        When True, show FWHM below the wavelength if the marker is on a peak.
    font:
        OpenCV font face.
    font_scale:
        Font scale factor.
    """

    color: tuple[int, int, int] = (0, 255, 255)
    drag_color: tuple[int, int, int] = (0, 200, 255)
    show_width: bool = False
    font: int = cv2.FONT_HERSHEY_SIMPLEX
    font_scale: float = 0.45

    def render(
        self,
        image: np.ndarray,
        lines: list[int],
        wavelengths: np.ndarray,
        viewport: Viewport,
        y_offset: int = 0,
        dragging_idx: int | None = None,
        intensity: np.ndarray | None = None,
    ) -> np.ndarray:
        """Render marker lines onto *image* in-place.  Returns image for chaining.

        Parameters
        ----------
        image:
            Target BGR image (the full composed frame).
        lines:
            Data indices of active marker lines.
        wavelengths:
            Wavelength array matching the spectrum data (nm or cm⁻¹).
        viewport:
            Current zoom/pan state used to map data indices to screen x.
        y_offset:
            Top y-coordinate where lines begin (e.g. top of the waterfall strip
            within a larger composed frame).
        dragging_idx:
            Index into *lines* of the line currently being dragged, or None.
        intensity:
            Spectrum intensity (0–1). Required for show_width; used to detect
            peak and compute FWHM. Delta is only drawn when marker is on a peak.
        """
        width = image.shape[1]
        height = image.shape[0]
        n_wl = len(wavelengths)

        for i, data_idx in enumerate(lines):
            if not (0 <= data_idx < n_wl):
                continue
            sx = viewport.data_x_to_screen(float(data_idx), width)
            if not (0 <= sx < width):
                continue
            color = self.drag_color if i == dragging_idx else self.color
            cv2.line(image, (sx, y_offset), (sx, height - 1), color, 1, cv2.LINE_AA)
            fwhm: float | None = None
            if self.show_width and intensity is not None and is_local_max(data_idx, intensity):
                fwhm = fwhm_nm(data_idx, intensity, wavelengths)
            _draw_label(
                image, sx, y_offset, wavelengths[data_idx], color, width,
                self.font, self.font_scale, fwhm,
            )

        return image


def _draw_label(
    image: np.ndarray,
    sx: int,
    y_offset: int,
    wavelength: float,
    color: tuple[int, int, int],
    width: int,
    font: int,
    font_scale: float,
    fwhm: float | None = None,
) -> None:
    label = f"{wavelength:.0f}"
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
    lx = max(0, min(sx + 2, width - tw - 2))
    ly = y_offset + th + 4
    cv2.putText(image, label, (lx + 1, ly + 1), font, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, label, (lx, ly), font, font_scale, color, 1, cv2.LINE_AA)
    if fwhm is not None:
        delta_scale = font_scale * _DELTA_SCALE
        delta_label = f"Δ{fwhm:.1f}nm"
        (dw, dh), _ = cv2.getTextSize(delta_label, font, delta_scale, 1)
        dx = max(0, min(lx, width - dw - 2))
        dy = ly + dh + 2
        cv2.putText(image, delta_label, (dx + 1, dy + 1), font, delta_scale, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, delta_label, (dx, dy), font, delta_scale, color, 1, cv2.LINE_AA)
