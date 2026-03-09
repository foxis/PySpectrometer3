"""Signal processing components for PySpectrometer3."""

from .base import ProcessorInterface
from .filters import SavitzkyGolayFilter
from .peak_detection import PeakDetector
from .pipeline import ProcessingPipeline

__all__ = [
    "ProcessorInterface",
    "ProcessingPipeline",
    "SavitzkyGolayFilter",
    "PeakDetector",
]
