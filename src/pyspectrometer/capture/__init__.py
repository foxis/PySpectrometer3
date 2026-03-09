"""Camera capture backends for PySpectrometer3."""

from .base import CameraInterface
from .picamera import Capture

__all__ = ["CameraInterface", "Capture"]
