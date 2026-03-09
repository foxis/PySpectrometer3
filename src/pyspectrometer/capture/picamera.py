"""Picamera2 capture backend for Raspberry Pi cameras."""

from typing import Optional
import numpy as np

from .base import CameraInterface


class Capture(CameraInterface):
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
        self._exposure = 10000  # Default exposure in microseconds
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
    def exposure(self) -> int:
        """Get exposure time in microseconds."""
        return self._exposure
    
    @exposure.setter
    def exposure(self, value: int) -> None:
        """Set exposure time in microseconds."""
        self._exposure = max(100, min(100000, value))
        if self._camera is not None and self._running:
            self._camera.set_controls({"ExposureTime": self._exposure})
    
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
        video_config = self._configure(frame_duration)
        
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
    
    def _configure(self, frame_duration: int) -> dict:
        """Configure based on requested mode and available formats."""
        if not self._monochrome:
            return self._config_color(frame_duration)
        sensor_modes = self._camera.sensor_modes
        best_mode = self._find_best_monochrome_mode(sensor_modes)
        if best_mode and best_mode.get("bit_depth", 8) >= 10:
            return self._config_raw(best_mode, frame_duration)
        result = self._try_monochrome_formats(frame_duration)
        if result is not None:
            return result
        return self._config_color(frame_duration)

    def _find_best_monochrome_mode(self, sensor_modes: list) -> Optional[dict]:
        """Find best monochrome mode meeting bit depth, preferring minimal depth."""
        best = None
        for mode in sensor_modes:
            mode_bit_depth = mode.get("bit_depth", 8)
            if mode_bit_depth >= self._bit_depth:
                if best is None or mode_bit_depth < best.get("bit_depth", 99):
                    best = mode
        return best

    def _config_raw(self, best_mode: dict, frame_duration: int) -> dict:
        """Create config for raw capture (10+ bit)."""
        raw_format = best_mode.get("format")
        raw_size = best_mode.get("size", (self._width, self._height))
        print(f"Using raw capture: {raw_format} {raw_size} {best_mode.get('bit_depth')}-bit")
        self._actual_bit_depth = best_mode.get("bit_depth", 10)
        self._use_raw = True
        return self._camera.create_video_configuration(
            main={"format": "YUV420", "size": (self._width, self._height)},
            raw={"format": raw_format, "size": raw_size},
            controls={"FrameDurationLimits": (frame_duration, frame_duration)},
        )

    def _try_monochrome_formats(self, frame_duration: int) -> Optional[dict]:
        """Try Y16, Y10, Y8, YUV420; return config or None."""
        for fmt in ["Y16", "Y10", "Y8", "YUV420"]:
            try:
                config = self._camera.create_video_configuration(
                    main={"format": fmt, "size": (self._width, self._height)},
                    controls={"FrameDurationLimits": (frame_duration, frame_duration)},
                )
                match fmt:
                    case "Y16":
                        self._actual_bit_depth = 16
                    case "Y10":
                        self._actual_bit_depth = 10
                    case _:
                        self._actual_bit_depth = 8
                print(f"Using monochrome format: {fmt}")
                return config
            except Exception:
                continue
        return None

    def _config_color(self, frame_duration: int) -> dict:
        """Create config for RGB888 color (8-bit)."""
        self._actual_bit_depth = 8
        self._monochrome = False
        print("Using RGB888 (8-bit color)")
        return self._camera.create_video_configuration(
            main={"format": "RGB888", "size": (self._width, self._height)},
            controls={"FrameDurationLimits": (frame_duration, frame_duration)},
        )
    
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
            # capture_array("raw") returns the raw stream directly
            raw_frame = self._camera.capture_array("raw")
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
        
        Raw frames from OV9281 with R10_CSI2P format come as packed 10-bit data.
        The raw frame is typically uint8 with packed pixels that need unpacking,
        or uint16 if the driver already unpacked them.
        
        For monochrome sensors like OV9281, all pixels are grayscale (no Bayer).
        
        Args:
            raw_frame: Raw sensor data (uint8 packed or uint16)
            
        Returns:
            Processed monochrome frame as uint16
        """
        # Log raw frame info on first call for debugging
        if not hasattr(self, '_raw_logged'):
            self._raw_logged = True
            print(f"[RAW] Frame shape: {raw_frame.shape}, dtype: {raw_frame.dtype}")
            print(f"[RAW] Expected: {self._height}x{self._width}, {self._actual_bit_depth}-bit")
        
        # Already uint16 - use directly (driver unpacked for us)
        if raw_frame.dtype == np.uint16:
            # Crop to expected dimensions (remove stride padding)
            h, w = raw_frame.shape[:2]
            if w > self._width or h > self._height:
                raw_frame = raw_frame[:self._height, :self._width]
            return raw_frame
        
        # uint8 data needs processing
        if raw_frame.dtype == np.uint8:
            if raw_frame.ndim == 2:
                height, byte_width = raw_frame.shape
                
                # Calculate expected packed width for 10-bit MIPI CSI-2
                # 4 pixels = 5 bytes, so width/4*5, rounded up
                groups = (self._width + 3) // 4
                min_packed_width = groups * 5
                
                # Log packing detection once
                if not hasattr(self, '_pack_logged'):
                    self._pack_logged = True
                    print(f"[RAW] byte_width={byte_width}, min_packed={min_packed_width}")
                
                # Detect packed format: width is significantly larger than pixel width
                # but not exactly 2x (which would indicate simple uint16)
                if byte_width >= min_packed_width and byte_width < self._width * 2:
                    # This is packed 10-bit data - unpack it
                    return self._unpack_raw10(raw_frame, self._width, self._height)
                elif byte_width >= self._width * 2:
                    # Might be uint16 stored as uint8 pairs
                    # View as uint16
                    reshaped = raw_frame.view(np.uint16)
                    if reshaped.shape[1] > self._width:
                        reshaped = reshaped[:self._height, :self._width]
                    return reshaped
                else:
                    # Simple 8-bit grayscale - scale up
                    return raw_frame[:self._height, :self._width].astype(np.uint16) * 4
            
            if raw_frame.ndim == 3:
                # 3D array - take first channel
                frame = raw_frame[:, :, 0]
                if frame.shape[1] > self._width:
                    frame = frame[:self._height, :self._width]
                return frame.astype(np.uint16) * 4
        
        # Unknown format - convert and hope for the best
        return np.asarray(raw_frame, dtype=np.uint16)
    
    def _unpack_raw10(
        self,
        packed: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Unpack 10-bit packed raw data (MIPI CSI-2 format) using vectorized NumPy.
        
        In packed format, 4 pixels occupy 5 bytes:
        - Byte 0: P0[9:2] (high 8 bits of pixel 0)
        - Byte 1: P1[9:2] (high 8 bits of pixel 1)
        - Byte 2: P2[9:2] (high 8 bits of pixel 2)
        - Byte 3: P3[9:2] (high 8 bits of pixel 3)
        - Byte 4: P3[1:0] | P2[1:0] | P1[1:0] | P0[1:0] (low 2 bits of each)
        
        Args:
            packed: Packed uint8 array from sensor
            width: Expected output width in pixels
            height: Expected output height in pixels
            
        Returns:
            Unpacked uint16 array with 10-bit values
        """
        # Number of 4-pixel groups
        groups = width // 4
        bytes_per_row = groups * 5
        
        # Trim packed array to needed data (remove stride padding)
        packed_height = min(height, packed.shape[0])
        packed = packed[:packed_height, :bytes_per_row]
        
        # Reshape to groups of 5 bytes: (height, groups, 5)
        packed = packed.reshape(packed_height, groups, 5)
        
        # Extract high 8 bits of each pixel (first 4 bytes)
        high_bits = packed[:, :, :4].astype(np.uint16)  # (H, G, 4)
        
        # Extract low 2 bits from byte 4
        low_byte = packed[:, :, 4]  # (H, G)
        
        # Create low bits array (H, G, 4) - extract 2 bits for each pixel
        low_bits = np.zeros((packed_height, groups, 4), dtype=np.uint16)
        low_bits[:, :, 0] = (low_byte >> 0) & 0x03
        low_bits[:, :, 1] = (low_byte >> 2) & 0x03
        low_bits[:, :, 2] = (low_byte >> 4) & 0x03
        low_bits[:, :, 3] = (low_byte >> 6) & 0x03
        
        # Combine: pixel = (high << 2) | low
        pixels = (high_bits << 2) | low_bits  # (H, G, 4)
        
        # Reshape to final image (H, W) - flatten groups
        output = pixels.reshape(packed_height, groups * 4)
        
        # Pad width if needed (when width % 4 != 0)
        if output.shape[1] < width:
            pad_width = width - output.shape[1]
            output = np.pad(output, ((0, 0), (0, pad_width)), mode='constant')
        
        # Pad height if needed
        if output.shape[0] < height:
            pad_height = height - output.shape[0]
            output = np.pad(output, ((0, pad_height), (0, 0)), mode='constant')
        
        return output
    
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
