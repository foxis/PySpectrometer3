"""Vertical marker-line renderer for waterfall and spectrum graphs.

Markers are user-placed vertical lines. They use the same rendering as peaks:
intelligent label placement, yellow box labels, thick black vertical lines.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from ..core.spectrum import Peak
from .peak_width import fwhm_nm, is_local_max
from .peaks import compute_placements, draw_labels, draw_lines
from .viewport import Viewport


def _marker_peaks(
    lines: list[int],
    wavelengths: np.ndarray,
    intensity: np.ndarray,
    n_wl: int,
) -> list[Peak]:
    """Build Peak-like objects for marker indices (for placement and drawing)."""
    out = []
    for data_idx in lines:
        if not (0 <= data_idx < n_wl):
            continue
        out.append(
            Peak(
                index=data_idx,
                wavelength=float(wavelengths[data_idx]),
                intensity=float(intensity[data_idx]),
            )
        )
    return out


@dataclass
class MarkersRenderer:
    """Renders marker lines using the same placement and drawing as peaks."""

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
        spectrum_screen_y: np.ndarray | None = None,
        graph_height: int | None = None,
    ) -> np.ndarray:
        """Render marker lines in-place. Same style as peaks: yellow box, thick black lines."""
        width = image.shape[1]
        height = image.shape[0]
        n_wl = len(wavelengths)
        use_placement = (
            spectrum_screen_y is not None
            and graph_height is not None
            and intensity is not None
            and len(spectrum_screen_y) == width
        )

        marker_peaks = _marker_peaks(lines, wavelengths, intensity, n_wl)
        if not marker_peaks:
            return image

        if use_placement:
            placements = compute_placements(
                marker_peaks,
                viewport,
                width,
                graph_height,
                spectrum_screen_y,
                self.font,
                self.font_scale,
                self.show_width,
            )
            # Placements are in graph coordinates; offset for composed frame (e.g. waterfall)
            adjusted = [
                (sx, x1, x2, y1 + y_offset, y2 + y_offset, peak, left)
                for sx, x1, x2, y1, y2, peak, left in placements
            ]
            get_fwhm = self._fwhm_getter(intensity, wavelengths)
            draw_labels(image, adjusted, self.font, self.font_scale, get_fwhm)
            draw_lines(image, adjusted, y_start=y_offset, y_end=height - 1)
        else:
            # Lines only (no placement data): same thick black line style
            minimal = [
                (viewport.data_x_to_screen(float(p.index), width), 0, 0, 0, 0, p, False)
                for p in marker_peaks
                if 0 <= viewport.data_x_to_screen(float(p.index), width) < width
            ]
            draw_lines(image, minimal, y_start=y_offset, y_end=height - 1)

        return image

    def _fwhm_getter(self, intensity: np.ndarray, wavelengths: np.ndarray):
        """Return get_fwhm(peak) for draw_labels: FWHM when show_width and on local max."""
        show = self.show_width

        def get_fwhm(p: Peak) -> float | None:
            if not show or not is_local_max(p.index, intensity):
                return None
            return fwhm_nm(p.index, intensity, wavelengths)

        return get_fwhm
