"""Base interface for camera capture backends."""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


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
            2D uint16 (monochrome) or 3D uint8 (RGB) array.
        """
        ...

    def __enter__(self) -> "CameraInterface":
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()
