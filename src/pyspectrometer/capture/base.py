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
            RGB image as numpy array with shape (height, width, 3)
        """
        ...
    
    def extract_spectrum_region(
        self,
        frame: np.ndarray,
        rows_to_average: int = 3,
        crop_height: int = 80,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract the spectrum region from a captured frame.
        
        This method extracts a horizontal strip from the center of the frame
        and computes the intensity values by averaging multiple rows.
        
        Args:
            frame: Captured frame as numpy array
            rows_to_average: Number of pixel rows to average for intensity
            crop_height: Height of the cropped preview region
            
        Returns:
            Tuple of (cropped_frame, intensity_array)
        """
        height = frame.shape[0]
        width = frame.shape[1]
        
        y_origin = (height // 2) - (crop_height // 2)
        
        cropped = frame[y_origin:y_origin + crop_height, 0:width]
        
        import cv2
        grayscale = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        rows, cols = grayscale.shape
        halfway = rows // 2
        
        intensity = np.zeros(cols, dtype=np.uint8)
        half_avg = rows_to_average // 2
        
        for i in range(cols):
            total = 0
            for offset in range(-half_avg, half_avg + 1):
                row_idx = halfway + offset
                if 0 <= row_idx < rows:
                    total += int(grayscale[row_idx, i])
            intensity[i] = total // rows_to_average
        
        return cropped, intensity
    
    def __enter__(self) -> "CameraInterface":
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()
