"""
PySpectrometer3 - A modular spectrometer application.

This package provides a flexible, extensible spectrometer system with:
- Pluggable camera backends
- Configurable processing pipelines
- Multiple display and export options
"""

__version__ = "3.0.0"

from .config import Config
from .spectrometer import Spectrometer

__all__ = ["Config", "Spectrometer", "__version__"]
