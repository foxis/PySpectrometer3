"""Picamera2 capture backend for Raspberry Pi cameras."""

from typing import Optional
import numpy as np

from .base import CameraInterface


class PicameraCapture(CameraInterface):
    """Camera capture implementation using Picamera2.
    
    This backend is designed for Raspberry Pi camera modules and uses
    the Picamera2 library for capture.
    """
    
    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        gain: float = 10.0,
        fps: int = 30,
    ):
        """Initialize Picamera2 capture.
        
        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            gain: Initial camera gain
            fps: Target frames per second
        """
        self._width = width
        self._height = height
        self._gain = gain
        self._fps = fps
        self._running = False
        self._camera: Optional["Picamera2"] = None
    
    @property
    def width(self) -> int:
        return self._width
    
    @property
    def height(self) -> int:
        return self._height
    
    @property
    def gain(self) -> float:
        return self._gain
    
    @gain.setter
    def gain(self, value: float) -> None:
        self._gain = max(0.0, min(50.0, value))
        if self._camera is not None and self._running:
            self._camera.set_controls({"AnalogueGain": self._gain})
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    def start(self) -> None:
        """Start Picamera2 capture."""
        if self._running:
            return
        
        try:
            from picamera2 import Picamera2
        except ImportError as e:
            raise ImportError(
                "Picamera2 is required for this capture backend. "
                "Install it with: pip install picamera2"
            ) from e
        
        self._camera = Picamera2()
        
        frame_duration = 1_000_000 // self._fps
        
        video_config = self._camera.create_video_configuration(
            main={
                "format": "RGB888",
                "size": (self._width, self._height),
            },
            controls={
                "FrameDurationLimits": (frame_duration, frame_duration),
            },
        )
        
        self._camera.configure(video_config)
        self._camera.start()
        self._camera.set_controls({"AnalogueGain": self._gain})
        
        self._running = True
    
    def stop(self) -> None:
        """Stop Picamera2 capture."""
        if not self._running:
            return
        
        if self._camera is not None:
            self._camera.stop()
            self._camera.close()
            self._camera = None
        
        self._running = False
    
    def capture(self) -> np.ndarray:
        """Capture a single frame from Picamera2.
        
        Returns:
            RGB image as numpy array
            
        Raises:
            RuntimeError: If camera is not running
        """
        if not self._running or self._camera is None:
            raise RuntimeError("Camera is not running. Call start() first.")
        
        return self._camera.capture_array()
