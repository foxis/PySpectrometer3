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
        self._use_raw = False  # Track if we're using raw capture
        
        # Log camera info and sensor modes
        sensor_modes = self._camera.sensor_modes
        print(f"Camera: {self._camera.camera_properties.get('Model', 'Unknown')}")
        print(f"Available sensor modes ({len(sensor_modes)}):")
        for i, mode in enumerate(sensor_modes):
            fmt = mode.get("format", "?")
            size = mode.get("size", (0, 0))
            bit_depth = mode.get("bit_depth", "?")
            print(f"  [{i}] {fmt} {size[0]}x{size[1]} {bit_depth}-bit")
        
        frame_duration = 1_000_000 // self._fps
        video_config = self._configure_camera(frame_duration)
        
        self._camera.configure(video_config)
        
        # Log actual configuration
        config = self._camera.camera_configuration()
        main_config = config.get("main", {})
        sensor_config = config.get("sensor", {})
        raw_config = config.get("raw", {})
        
        print(f"Main stream: {main_config.get('format', '?')} {main_config.get('size', '?')}")
        if raw_config:
            print(f"Raw stream: {raw_config.get('format', '?')} {raw_config.get('size', '?')}")
        print(f"Sensor bit depth: {sensor_config.get('bit_depth', '?')}")
        print(f"Actual bit depth used: {self._actual_bit_depth}, Use raw: {self._use_raw}")
        
        self._camera.start()
        self._camera.set_controls({"AnalogueGain": self._gain})
        
        self._running = True
    
    def _configure_camera(self, frame_duration: int) -> dict:
        """Configure camera based on requested mode and available formats."""
        # Try formats in order of preference for monochrome high bit-depth
        if self._monochrome:
            # OV9281 and similar monochrome sensors often support SRGGB10 or similar raw formats
            # We can use raw capture to get higher bit depth
            
            # First, check what formats are available
            sensor_modes = self._camera.sensor_modes
            
            # Find best monochrome/raw mode based on requested bit depth
            best_mode = None
            for mode in sensor_modes:
                mode_bit_depth = mode.get("bit_depth", 8)
                if mode_bit_depth >= self._bit_depth:
                    if best_mode is None or mode_bit_depth < best_mode.get("bit_depth", 99):
                        best_mode = mode
            
            # If we found a high bit-depth mode, use raw capture
            if best_mode and best_mode.get("bit_depth", 8) >= 10:
                raw_format = best_mode.get("format")
                raw_size = best_mode.get("size", (self._width, self._height))
                
                print(f"Using raw capture: {raw_format} {raw_size} {best_mode.get('bit_depth')}-bit")
                
                # Create config with raw stream for high bit-depth
                video_config = self._camera.create_video_configuration(
                    main={
                        "format": "YUV420",  # Main stream still needs standard format
                        "size": (self._width, self._height),
                    },
                    raw={
                        "format": raw_format,
                        "size": raw_size,
                    },
                    controls={
                        "FrameDurationLimits": (frame_duration, frame_duration),
                    },
                )
                self._actual_bit_depth = best_mode.get("bit_depth", 10)
                self._use_raw = True
                return video_config
            
            # Fallback: Try standard monochrome formats (Y10, Y16, etc.)
            for fmt in ["Y16", "Y10", "Y8", "YUV420"]:
                try:
                    video_config = self._camera.create_video_configuration(
                        main={
                            "format": fmt,
                            "size": (self._width, self._height),
                        },
                        controls={
                            "FrameDurationLimits": (frame_duration, frame_duration),
                        },
                    )
                    match fmt:
                        case "Y16":
                            self._actual_bit_depth = 16
                        case "Y10":
                            self._actual_bit_depth = 10
                        case _:
                            self._actual_bit_depth = 8
                    print(f"Using monochrome format: {fmt}")
                    return video_config
                except Exception:
                    continue
        
        # Color mode or final fallback: use RGB888 (8-bit per channel)
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
        print("Using RGB888 (8-bit color)")
        return video_config
    
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
        
        # If using raw capture, get raw frame for higher bit depth
        if self._use_raw:
            arrays = self._camera.capture_arrays(["main", "raw"])
            raw_frame = arrays["raw"]
            return self._process_raw_frame(raw_frame)
        
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
    
    def _process_raw_frame(self, raw_frame: np.ndarray) -> np.ndarray:
        """Process raw frame from sensor to extract monochrome data.
        
        Raw frames from sensors like OV9281 come in Bayer format (e.g., SRGGB10)
        but for monochrome sensors, all pixels are the same so we can use directly.
        
        Args:
            raw_frame: Raw sensor data
            
        Returns:
            Processed monochrome frame as uint16
        """
        # Raw frame is typically 16-bit packed or already uint16
        if raw_frame.dtype == np.uint16:
            return raw_frame
        
        # If 8-bit packed data for 10-bit, unpack
        if raw_frame.dtype == np.uint8:
            # For 10-bit packed in 16-bit: every 2 bytes = 1 pixel
            if len(raw_frame.shape) == 2:
                # Already 2D monochrome
                return raw_frame.astype(np.uint16)
            elif len(raw_frame.shape) == 3:
                # Bayer pattern - for monochrome sensor, just take any channel
                # Or average for better result
                return raw_frame[:, :, 0].astype(np.uint16)
        
        return raw_frame.astype(np.uint16)
    
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
