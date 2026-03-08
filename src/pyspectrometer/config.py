"""Configuration management for PySpectrometer3."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CameraConfig:
    """Camera-related configuration."""
    
    frame_width: int = 800
    frame_height: int = 600
    gain: float = 10.0
    gain_min: float = 0.0
    gain_max: float = 50.0
    fps: int = 30
    
    # Monochrome camera support (default: True for spectrometer use)
    monochrome: bool = True
    bit_depth: int = 10  # 8, 10, or 16 (for monochrome cameras)


@dataclass
class DisplayConfig:
    """Display-related configuration."""
    
    fullscreen: bool = False
    waterfall_enabled: bool = False
    graph_height: int = 320
    preview_height: int = 80
    message_height: int = 80
    
    # Window size (canvas will be scaled to fit)
    window_width: int = 800
    window_height: int = 480
    
    # UI scaling for small displays
    font_scale: float = 0.4
    text_thickness: int = 1
    status_col1_x: int = 490
    status_col2_x: int = 640
    
    @property
    def stack_height(self) -> int:
        """Total height of the stacked display."""
        return self.graph_height + self.preview_height + self.message_height


@dataclass
class ProcessingConfig:
    """Signal processing configuration."""
    
    savgol_window: int = 17
    savgol_poly: int = 7
    savgol_poly_min: int = 0
    savgol_poly_max: int = 15
    
    peak_min_distance: int = 15  # Lower for sharp Hg lines (577/579 ~7 px apart)
    peak_min_distance_min: int = 1
    peak_min_distance_max: int = 100
    
    peak_threshold: int = 20
    peak_threshold_min: int = 0
    peak_threshold_max: int = 100
    
    pixel_rows_to_average: int = 3


@dataclass
class CalibrationConfig:
    """Calibration-related configuration."""
    
    data_file: Path = field(default_factory=lambda: Path("caldata.txt"))
    default_pixels: tuple[int, ...] = (0, 400, 800)
    default_wavelengths: tuple[float, ...] = (380.0, 560.0, 750.0)


@dataclass
class ExportConfig:
    """Export-related configuration."""
    
    output_dir: Path = field(default_factory=lambda: Path("."))
    timestamp_format: str = "%Y%m%d--%H%M%S"
    time_format: str = "%H:%M:%S"


@dataclass
class WaterfallConfig:
    """Waterfall display configuration."""
    
    contrast: float = 2.5
    brightness: int = 10


@dataclass
class ExtractionConfig:
    """Spectrum extraction configuration."""
    
    method: str = "weighted_sum"  # "median", "weighted_sum", "gaussian"
    rotation_angle: float = 0.0  # degrees, loaded from calibration
    perpendicular_width: int = 20  # pixels to sample perpendicular to axis
    perpendicular_width_min: int = 5
    perpendicular_width_max: int = 100
    background_percentile: float = 10.0  # for background subtraction
    spectrum_y_center: int = 0  # 0 = auto (frame center)


@dataclass
class Config:
    """Main configuration container for PySpectrometer3.
    
    This dataclass aggregates all configuration options for the spectrometer
    application, providing sensible defaults that match the original behavior.
    """
    
    camera: CameraConfig = field(default_factory=CameraConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    waterfall: WaterfallConfig = field(default_factory=WaterfallConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    
    # Window titles
    spectrograph_title: str = "PySpectrometer 3 - Spectrograph"
    waterfall_title: str = "PySpectrometer 3 - Waterfall"
    
    @classmethod
    def from_args(
        cls,
        fullscreen: bool = False,
        waterfall: bool = False,
        waveshare: bool = False,
        gain: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        monochrome: bool = False,
        bit_depth: int = 10,
    ) -> "Config":
        """Create configuration from command-line arguments."""
        config = cls()
        
        config.display.fullscreen = fullscreen
        config.display.waterfall_enabled = waterfall
        
        if gain is not None:
            config.camera.gain = gain
        if width is not None:
            config.camera.frame_width = width
        if height is not None:
            config.camera.frame_height = height
        
        config.camera.monochrome = monochrome
        config.camera.bit_depth = bit_depth
        
        if waveshare:
            config.apply_waveshare_preset()
            
        return config
    
    def apply_waveshare_preset(self) -> None:
        """Apply settings optimized for Waveshare display (640x400).
        
        Layout breakdown for 400px height:
        - message_height: 60 (control bar with 2 rows of buttons)
        - preview_height: 60 (camera preview window)
        - graph_height: 280 (spectrum graph)
        Total: 400px
        """
        self.camera.frame_width = 640
        self.camera.frame_height = 400
        
        self.display.window_width = 640
        self.display.window_height = 400
        self.display.graph_height = 280
        self.display.preview_height = 60
        self.display.message_height = 60
        
        self.display.font_scale = 0.35
        self.display.text_thickness = 1
        self.display.status_col1_x = 340
        self.display.status_col2_x = 500
        
        self.calibration.default_pixels = (0, 320, 640)

        self.processing.peak_min_distance = 15  # Sharp Hg lines
    
    @classmethod
    def waveshare_35(cls) -> "Config":
        """Create configuration for Waveshare 3.5\" display (640x480)."""
        config = cls()
        config.apply_waveshare_preset()
        return config
    
    @classmethod
    def standard_800(cls) -> "Config":
        """Create standard 800x480 configuration."""
        return cls()
