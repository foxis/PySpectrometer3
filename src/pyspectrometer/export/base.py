"""Base interface for data exporters."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..core.spectrum import SpectrumData


class ExporterInterface(ABC):
    """Abstract base class for spectrum data exporters.

    Exporters handle saving spectrum data in various formats (CSV, JSON,
    database, etc.). Each exporter implements a specific output format.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this exporter."""
        ...

    @property
    @abstractmethod
    def extension(self) -> str:
        """File extension for this export format (e.g., '.csv')."""
        ...

    @abstractmethod
    def export(
        self,
        data: "SpectrumData",
        path: Path,
    ) -> Path:
        """Export spectrum data to a file.

        Args:
            data: Spectrum data to export
            path: Output file path

        Returns:
            Path to the created file
        """
        ...

    def export_with_images(
        self,
        data: "SpectrumData",
        path: Path,
        spectrum_image: np.ndarray | None = None,
        waterfall_image: np.ndarray | None = None,
    ) -> list[Path]:
        """Export spectrum data along with associated images.

        Args:
            data: Spectrum data to export
            path: Base output file path
            spectrum_image: Optional spectrum display image to save
            waterfall_image: Optional waterfall display image to save

        Returns:
            List of paths to created files
        """
        created_files = [self.export(data, path)]

        if spectrum_image is not None:
            import cv2

            img_path = path.with_suffix(".png")
            cv2.imwrite(str(img_path), spectrum_image)
            created_files.append(img_path)

        if waterfall_image is not None:
            import cv2

            waterfall_path = path.with_name(f"{path.stem}_waterfall.png")
            cv2.imwrite(str(waterfall_path), waterfall_image)
            created_files.append(waterfall_path)

        return created_files
