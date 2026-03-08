"""Camera capture backends for PySpectrometer3."""

from .base import CameraInterface
from .picamera import PicameraCapture

__all__ = ["CameraInterface", "PicameraCapture"]
