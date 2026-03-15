"""Vertical marker-line renderer for waterfall and spectrum graphs.

Markers are user-placed vertical lines identified by a data-array index.
Each line is annotated with its wavelength value.  The renderer is stateless —
interaction state (which index is being dragged) is passed at render time so
callers retain full control over the marker list.

Composable: renders onto any BGR image that shares the same Viewport.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .viewport import Viewport


@dataclass
class MarkersRenderer:
    """Renders vertical marker lines with wavelength labels.

    Attributes
    ----------
    color:
        BGR colour for idle marker lines and labels.
    drag_color:
        BGR colour used for the line currently being dragged.
    font:
        OpenCV font face.
    font_scale:
        Font scale factor.
    """

    color: tuple[int, int, int] = (0, 255, 255)
    drag_color: tuple[int, int, int] = (0, 200, 255)
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
            _draw_label(image, sx, y_offset, wavelengths[data_idx], color, width,
                        self.font, self.font_scale)

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
) -> None:
    label = f"{wavelength:.0f}"
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
    lx = max(0, min(sx + 2, width - tw - 2))
    ly = y_offset + th + 4
    cv2.putText(image, label, (lx + 1, ly + 1), font, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, label, (lx, ly), font, font_scale, color, 1, cv2.LINE_AA)
