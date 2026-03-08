"""Picamera2 capture backend for Raspberry Pi cameras."""

from typing import Optional
import numpy as np

from .base import CameraInterface


class PicameraCapture(CameraInterface):
    """Camera capture implementation using Picamera2.
    
    This backend is designed for Raspberry Pi camera modules and uses
    the Picamera2 library for capture.
    
    Supports both color (RGB888) and monochrome (Y10/Y16) modes.
    Monochrome mode provides higher bit depth (10-bit or 16-bit)
    for better dynamic range.
    """
    
    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        gain: float = 10.0,
        fps: int = 30,
        monochrome: bool = False,
        bit_depth: int = 10,
    ):
        """Initialize Picamera2 capture.
        
        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            gain: Initial camera gain
            fps: Target frames per second
            monochrome: If True, use monochrome format with higher bit depth
            bit_depth: Bit depth for monochrome mode (10 or 16)
        """
        self._width = width
        self._height = height
        self._gain = gain
        self._fps = fps
        self._monochrome = monochrome
        self._bit_depth = bit_depth if bit_depth in (10, 16) else 10
        self._running = False
        self._camera: Optional["Picamera2"] = None
        self._actual_bit_depth: int = 8
    
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
    
    @property
    def bit_depth(self) -> int:
        """Get actual bit depth being used."""
        return self._actual_bit_depth
    
    @property
    def is_monochrome(self) -> bool:
        """Check if camera is in monochrome mode."""
        return self._monochrome
    
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
        
        if self._monochrome:
            # Monochrome mode: use Y10 or Y16 format for higher bit depth
            # Y10 = 10-bit monochrome, Y16 = 16-bit monochrome
            mono_format = f"Y{self._bit_depth}"
            
            # Try to configure monochrome format
            try:
                video_config = self._camera.create_video_configuration(
                    main={
                        "format": mono_format,
                        "size": (self._width, self._height),
                    },
                    controls={
                        "FrameDurationLimits": (frame_duration, frame_duration),
                    },
                )
                self._actual_bit_depth = self._bit_depth
                print(f"Camera configured for {mono_format} ({self._bit_depth}-bit monochrome)")
            except Exception as e:
                # Fallback: try raw format
                print(f"Y{self._bit_depth} not available, trying raw format: {e}")
                try:
                    video_config = self._camera.create_video_configuration(
                        main={
                            "format": "YUV420",
                            "size": (self._width, self._height),
                        },
                        raw={
                            "size": (self._width, self._height),
                        },
                        controls={
                            "FrameDurationLimits": (frame_duration, frame_duration),
                        },
                    )
                    self._actual_bit_depth = 8
                    print("Fallback to YUV420 (8-bit)")
                except Exception:
                    # Final fallback to RGB
                    video_config = self._camera.create_video_configuration(
                        main={
                            "format": "RGB888",
                            "size": (self._width, self._height),
                        },
                        controls={
                            "FrameDurationLimits": (frame_duration, frame_duration),
                        },
                    )
                    self._actual_bit_depth = 8
                    self._monochrome = False
                    print("Fallback to RGB888 (8-bit color)")
        else:
            # Color mode: use RGB888 (8-bit per channel)
            video_config = self._camera.create_video_configuration(
                main={
                    "format": "RGB888",
                    "size": (self._width, self._height),
                },
                controls={
                    "FrameDurationLimits": (frame_duration, frame_duration),
                },
            )
            self._actual_bit_depth = 8
        
        self._camera.configure(video_config)
        
        # Log actual sensor configuration
        sensor_config = self._camera.camera_configuration().get("sensor", {})
        sensor_bit_depth = sensor_config.get("bit_depth", "unknown")
        print(f"Sensor bit depth: {sensor_bit_depth}")
        
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
            Image as numpy array:
            - Monochrome: 2D array (H, W) with dtype uint16 for 10/16-bit
            - Color: 3D array (H, W, 3) with dtype uint8 (RGB)
            
        Raises:
            RuntimeError: If camera is not running
        """
        if not self._running or self._camera is None:
            raise RuntimeError("Camera is not running. Call start() first.")
        
        frame = self._camera.capture_array()
        
        # Handle monochrome high bit-depth formats
        if self._monochrome and self._actual_bit_depth > 8:
            # Y10/Y16 data comes as bytes, view as uint16
            if frame.dtype == np.uint8 and frame.ndim == 2:
                # Packed format: reshape and view as uint16
                frame = frame.view(np.uint16)
            elif frame.dtype == np.uint8 and frame.ndim == 3:
                # If still 3D, take first channel (Y from YUV)
                frame = frame[:, :, 0].astype(np.uint16)
        elif self._monochrome and frame.ndim == 3:
            # Monochrome but got color frame, convert to grayscale
            # Use luminance formula: Y = 0.299*R + 0.587*G + 0.114*B
            frame = (0.299 * frame[:, :, 0] + 
                     0.587 * frame[:, :, 1] + 
                     0.114 * frame[:, :, 2]).astype(np.uint8)
        
        return frame
    
    def capture_normalized(self) -> np.ndarray:
        """Capture a frame normalized to 0-255 range.
        
        Useful for display when using high bit-depth capture.
        
        Returns:
            Image normalized to uint8 (0-255)
        """
        frame = self.capture()
        
        if self._actual_bit_depth > 8:
            # Scale down from 10/16-bit to 8-bit
            max_val = (1 << self._actual_bit_depth) - 1
            frame = (frame.astype(np.float32) * 255 / max_val).astype(np.uint8)
        
        return frame
