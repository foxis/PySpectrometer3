"""CSV export for spectrum data."""

from pathlib import Path
import time

from ..core.spectrum import SpectrumData
from .base import ExporterInterface


class CSVExporter(ExporterInterface):
    """Exports spectrum data to CSV format.

    The CSV file contains: Pixel (index), Wavelength (nm), Intensity (float32 0-1).
    """
    
    def __init__(
        self,
        output_dir: Path = None,
        timestamp_format: str = "%Y%m%d--%H%M%S",
    ):
        """Initialize CSV exporter.
        
        Args:
            output_dir: Directory for output files (defaults to current directory)
            timestamp_format: Format string for timestamps in filenames
        """
        self._output_dir = output_dir or Path(".")
        self._timestamp_format = timestamp_format
    
    @property
    def name(self) -> str:
        return "CSV Exporter"
    
    @property
    def extension(self) -> str:
        return ".csv"
    
    @property
    def output_dir(self) -> Path:
        return self._output_dir
    
    @output_dir.setter
    def output_dir(self, value: Path) -> None:
        self._output_dir = value
    
    def export(
        self,
        data: SpectrumData,
        path: Path = None,
    ) -> Path:
        """Export spectrum data to CSV file.
        
        Args:
            data: Spectrum data to export
            path: Output file path (auto-generated if None)
            
        Returns:
            Path to the created CSV file
        """
        if path is None:
            timestamp = time.strftime(self._timestamp_format)
            path = self._output_dir / f"Spectrum-{timestamp}.csv"
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            f.write("Pixel,Wavelength,Intensity\r\n")
            for pixel_idx, (wavelength, intensity) in enumerate(data.to_csv_rows()):
                f.write(f"{pixel_idx},{wavelength},{intensity}\r\n")
        
        return path
    
    def generate_filename(self, prefix: str = "Spectrum") -> Path:
        """Generate a timestamped filename.
        
        Args:
            prefix: Filename prefix
            
        Returns:
            Path with timestamped filename
        """
        timestamp = time.strftime(self._timestamp_format)
        return self._output_dir / f"{prefix}-{timestamp}{self.extension}"
