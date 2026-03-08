"""Signal processing components for PySpectrometer3."""

from .base import ProcessorInterface
from .pipeline import ProcessingPipeline
from .filters import SavitzkyGolayFilter
from .peak_detection import PeakDetector

__all__ = [
    "ProcessorInterface",
    "ProcessingPipeline",
    "SavitzkyGolayFilter",
    "PeakDetector",
]
