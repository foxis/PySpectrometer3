"""Graticule rendering for spectrum display."""

import cv2
import numpy as np

from ..core.calibration import GraticuleData


class GraticuleRenderer:
    """Renders calibrated graticule lines on spectrum graphs."""

    def __init__(
        self,
        font: int = cv2.FONT_HERSHEY_SIMPLEX,
        font_scale: float = 0.4,
        text_offset: int = 12,
    ):
        """Initialize graticule renderer.

        Args:
            font: OpenCV font type
            font_scale: Font scale factor
            text_offset: Horizontal text offset for labels
        """
        self.font = font
        self.font_scale = font_scale
        self.text_offset = text_offset

    def render_on_graph(
        self,
        graph: np.ndarray,
        graticule: GraticuleData,
    ) -> None:
        """Render graticule lines on a spectrum graph.

        Args:
            graph: Graph image to draw on (modified in place)
            graticule: Graticule data with line positions
        """
        height = graph.shape[0]

        for position in graticule.tens:
            cv2.line(
                graph,
                (position, 15),
                (position, height),
                (200, 200, 200),
                1,
            )

        for pos, label in graticule.fifties:
            cv2.line(
                graph,
                (pos, 15),
                (pos, height),
                (0, 0, 0),
                1,
            )
            unit = getattr(graticule, "unit", "nm")
            cv2.putText(
                graph,
                f"{label}{unit}",
                (pos - self.text_offset, 12),
                self.font,
                self.font_scale,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    def render_horizontal_lines(
        self,
        graph: np.ndarray,
        spacing: int = 64,
    ) -> None:
        """Render horizontal grid lines on a graph.

        Args:
            graph: Graph image to draw on (modified in place)
            spacing: Spacing between horizontal lines
        """
        height = graph.shape[0]
        width = graph.shape[1]

        for i in range(spacing, height, spacing):
            cv2.line(
                graph,
                (0, i),
                (width, i),
                (100, 100, 100),
                1,
            )

    def render_on_waterfall(
        self,
        waterfall: np.ndarray,
        graticule: GraticuleData,
        y_offset: int = 162,
    ) -> None:
        """Render graticule markers on waterfall display.

        Args:
            waterfall: Waterfall image to draw on (modified in place)
            graticule: Graticule data with line positions
            y_offset: Y offset for the waterfall region
        """
        height = waterfall.shape[0]

        for pos, label in graticule.fifties:
            for i in range(y_offset, height - 5, 20):
                cv2.line(
                    waterfall,
                    (pos, i),
                    (pos, i + 1),
                    (0, 0, 0),
                    2,
                )
                cv2.line(
                    waterfall,
                    (pos, i),
                    (pos, i + 1),
                    (255, 255, 255),
                    1,
                )

            unit = getattr(graticule, "unit", "nm")
            cv2.putText(
                waterfall,
                f"{label}{unit}",
                (pos - self.text_offset, height - 5),
                self.font,
                self.font_scale,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                waterfall,
                f"{label}{unit}",
                (pos - self.text_offset, height - 5),
                self.font,
                self.font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
