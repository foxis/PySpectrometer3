"""Base interface for camera capture backends."""

from abc import ABC, abstractmethod

import numpy as np

# All camera backends return uint16 grayscale scaled linearly to 0..CAPTURE_UINT16_MAX.
CAPTURE_UINT16_MAX = 65535


def scale_to_uint16_full_scale(arr: np.ndarray, in_max: int) -> np.ndarray:
    """Map native integer samples [0..in_max] linearly to uint16 [0..CAPTURE_UINT16_MAX].

    Preserves linearity so downstream can normalize with max_val=CAPTURE_UINT16_MAX.
    """
    if in_max <= 0:
        return np.zeros(arr.shape, dtype=np.uint16)
    x = arr.astype(np.float32) * (float(CAPTURE_UINT16_MAX) / float(in_max))
    return np.clip(np.rint(x), 0, CAPTURE_UINT16_MAX).astype(np.uint16)


def mirror_horizontal(frame: np.ndarray) -> np.ndarray:
    """Left-right mirror of a 2D image or HWC array (axis 1 reversed)."""
    return np.fliplr(frame)


class CameraInterface(ABC):
    """Abstract base class for camera capture backends.

    This interface defines the contract that all camera implementations
    must follow, enabling support for different camera types (Picamera2,
    USB webcams, etc.) through a unified API.
    """

    @property
    @abstractmethod
    def width(self) -> int:
        """Get frame width in pixels."""
        ...

    @property
    @abstractmethod
    def height(self) -> int:
        """Get frame height in pixels."""
        ...

    @property
    @abstractmethod
    def gain(self) -> float:
        """Get current camera gain."""
        ...

    @gain.setter
    @abstractmethod
    def gain(self, value: float) -> None:
        """Set camera gain."""
        ...

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """Check if camera is currently capturing."""
        ...

    @abstractmethod
    def start(self) -> None:
        """Start camera capture."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop camera capture."""
        ...

    @abstractmethod
    def capture(self) -> np.ndarray:
        """Capture a single frame.

        Returns:
            2D uint16 grayscale (height, width), values 0..CAPTURE_UINT16_MAX (65535),
            linearly scaled from the sensor or source native range.
        """
        ...

    def __enter__(self) -> "CameraInterface":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()
