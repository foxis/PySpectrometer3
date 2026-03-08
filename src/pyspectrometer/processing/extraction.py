"""Spectrum extraction with rotation correction and multiple methods."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional
import numpy as np
import cv2


class ExtractionMethod(Enum):
    """Available spectrum extraction methods."""
    MEDIAN = auto()
    WEIGHTED_SUM = auto()
    GAUSSIAN = auto()
    
    def __str__(self) -> str:
        return self.name.replace("_", " ").title()
    
    def next(self) -> "ExtractionMethod":
        """Cycle to next method."""
        members = list(ExtractionMethod)
        idx = members.index(self)
        return members[(idx + 1) % len(members)]


@dataclass
class ExtractionResult:
    """Result of spectrum extraction."""
    intensity: np.ndarray
    cropped_frame: np.ndarray
    method_used: ExtractionMethod
    rotation_angle: float
    perpendicular_width: int


class SpectrumExtractor:
    """Extracts spectrum intensity from camera frames.
    
    Handles rotated spectrum lines and vertical structure by sampling
    perpendicular to the spectrum axis and applying one of three methods:
    - Median: robust to outliers
    - Weighted Sum: best S/N ratio (default)
    - Gaussian: most accurate, fits profile
    """
    
    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        method: ExtractionMethod = ExtractionMethod.WEIGHTED_SUM,
        rotation_angle: float = 0.0,
        perpendicular_width: int = 20,
        spectrum_y_center: Optional[int] = None,
        crop_height: int = 80,
        background_percentile: float = 10.0,
    ):
        """Initialize spectrum extractor.
        
        Args:
            frame_width: Width of input frames in pixels
            frame_height: Height of input frames in pixels
            method: Extraction method to use
            rotation_angle: Rotation of spectrum line in degrees (+ = clockwise)
            perpendicular_width: Number of pixels to sample perpendicular to axis
            spectrum_y_center: Y coordinate of spectrum center (None = frame center)
            crop_height: Height of cropped preview region
            background_percentile: Percentile for background estimation
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.method = method
        self.rotation_angle = rotation_angle
        self.perpendicular_width = perpendicular_width
        self.spectrum_y_center = spectrum_y_center or (frame_height // 2)
        self.crop_height = crop_height
        self.background_percentile = background_percentile
        
        self._perpendicular_width_min = 5
        self._perpendicular_width_max = 100
        
        self._last_gaussian_params: Optional[np.ndarray] = None
        
        self._precompute_sampling_coords()
    
    def _precompute_sampling_coords(self) -> None:
        """Precompute sampling coordinates for current rotation angle."""
        angle_rad = np.radians(self.rotation_angle)
        
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        perp_cos = -sin_a
        perp_sin = cos_a
        
        half_perp = self.perpendicular_width // 2
        perp_offsets = np.arange(-half_perp, half_perp + 1)
        
        self._perp_dy = perp_offsets * perp_cos
        self._perp_dx = perp_offsets * perp_sin
        
        x_positions = np.arange(self.frame_width)
        
        x_offset_from_center = x_positions - (self.frame_width // 2)
        self._base_y = self.spectrum_y_center + x_offset_from_center * np.tan(angle_rad)
    
    def extract(self, frame: np.ndarray) -> ExtractionResult:
        """Extract spectrum intensity from frame.
        
        Args:
            frame: Input frame as BGR numpy array (height, width, 3)
            
        Returns:
            ExtractionResult with intensity array and cropped frame
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        intensity = self._extract_with_method(gray)
        
        cropped = self._create_cropped_preview(frame)
        
        return ExtractionResult(
            intensity=intensity,
            cropped_frame=cropped,
            method_used=self.method,
            rotation_angle=self.rotation_angle,
            perpendicular_width=self.perpendicular_width,
        )
    
    def _extract_with_method(self, gray: np.ndarray) -> np.ndarray:
        """Apply selected extraction method."""
        match self.method:
            case ExtractionMethod.MEDIAN:
                return self._extract_median(gray)
            case ExtractionMethod.WEIGHTED_SUM:
                return self._extract_weighted_sum(gray)
            case ExtractionMethod.GAUSSIAN:
                return self._extract_gaussian(gray)
    
    def _sample_perpendicular(self, gray: np.ndarray, x: int) -> np.ndarray:
        """Sample pixels perpendicular to spectrum axis at given x position.
        
        Args:
            gray: Grayscale image as float32
            x: X position to sample at
            
        Returns:
            Array of sampled intensity values
        """
        base_y = self._base_y[x]
        
        y_coords = base_y + self._perp_dy
        x_coords = x + self._perp_dx
        
        y_coords = np.clip(y_coords, 0, gray.shape[0] - 1).astype(np.int32)
        x_coords = np.clip(x_coords, 0, gray.shape[1] - 1).astype(np.int32)
        
        return gray[y_coords, x_coords]
    
    def _extract_median(self, gray: np.ndarray) -> np.ndarray:
        """Extract using median along perpendicular slices."""
        intensity = np.zeros(self.frame_width, dtype=np.float32)
        
        for x in range(self.frame_width):
            samples = self._sample_perpendicular(gray, x)
            intensity[x] = np.median(samples)
        
        return np.clip(intensity, 0, 255).astype(np.uint8)
    
    def _extract_weighted_sum(self, gray: np.ndarray) -> np.ndarray:
        """Extract using intensity-weighted sum along perpendicular slices."""
        intensity = np.zeros(self.frame_width, dtype=np.float32)
        
        for x in range(self.frame_width):
            samples = self._sample_perpendicular(gray, x)
            
            bg = np.percentile(samples, self.background_percentile)
            samples_bg = np.maximum(samples - bg, 0)
            
            total = np.sum(samples_bg)
            if total > 0:
                weights = samples_bg / total
                intensity[x] = np.sum(samples * weights)
            else:
                intensity[x] = np.mean(samples)
        
        return np.clip(intensity, 0, 255).astype(np.uint8)
    
    def _extract_gaussian(self, gray: np.ndarray) -> np.ndarray:
        """Extract by fitting Gaussian to each perpendicular slice."""
        from scipy.optimize import curve_fit
        
        intensity = np.zeros(self.frame_width, dtype=np.float32)
        
        half_perp = self.perpendicular_width // 2
        x_fit = np.arange(-half_perp, half_perp + 1, dtype=np.float32)
        
        for x in range(self.frame_width):
            samples = self._sample_perpendicular(gray, x)
            
            try:
                bg = np.percentile(samples, self.background_percentile)
                amplitude = np.max(samples) - bg
                center = x_fit[np.argmax(samples)]
                sigma = 3.0
                
                if self._last_gaussian_params is not None:
                    _, prev_center, prev_sigma, _ = self._last_gaussian_params
                    center = prev_center
                    sigma = prev_sigma
                
                p0 = [amplitude, center, sigma, bg]
                
                bounds = (
                    [0, -half_perp, 0.5, 0],
                    [300, half_perp, half_perp, 255]
                )
                
                popt, _ = curve_fit(
                    self._gaussian,
                    x_fit,
                    samples,
                    p0=p0,
                    bounds=bounds,
                    maxfev=50,
                )
                
                self._last_gaussian_params = popt
                
                amplitude, _, sigma, _ = popt
                intensity[x] = amplitude
                
            except (RuntimeError, ValueError):
                intensity[x] = np.max(samples) - np.percentile(samples, self.background_percentile)
        
        max_val = np.max(intensity)
        if max_val > 0:
            intensity = (intensity / max_val) * 255
        
        return np.clip(intensity, 0, 255).astype(np.uint8)
    
    @staticmethod
    def _gaussian(x: np.ndarray, amplitude: float, center: float, sigma: float, background: float) -> np.ndarray:
        """1D Gaussian function for curve fitting."""
        return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2) + background
    
    def _create_cropped_preview(self, frame: np.ndarray) -> np.ndarray:
        """Create cropped preview region centered on spectrum."""
        half_crop = self.crop_height // 2
        y_start = max(0, self.spectrum_y_center - half_crop)
        y_end = min(frame.shape[0], self.spectrum_y_center + half_crop)
        
        return frame[y_start:y_end, :].copy()
    
    def detect_angle(self, frame: np.ndarray, visualize: bool = False) -> tuple[float, Optional[np.ndarray]]:
        """Auto-detect spectrum rotation angle from vertical spectral lines.
        
        Detects near-vertical stripes (spectral lines) and measures their
        deviation from true vertical. Returns the angle needed to rotate
        the spectrum so lines become horizontal for proper extraction.
        
        Args:
            frame: Input frame as BGR numpy array
            visualize: If True, return visualization image
            
        Returns:
            Tuple of (detected_angle_degrees, visualization_image_or_None)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Adaptive blur based on image size
        blur_size = max(3, (min(width, height) // 100) | 1)
        blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
        
        # Use adaptive thresholding for better edge detection on varied lighting
        block_size = max(11, (min(width, height) // 20) | 1)
        edges = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, 2
        )
        edges = cv2.Canny(edges, 30, 100)
        
        # Morphological operations to connect broken edges
        kernel_height = max(3, height // 30)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_height))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Minimum line length based on image height (stripes are tall)
        min_line_length = height // 4
        max_line_gap = height // 15
        
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=max(30, height // 10),
            minLineLength=min_line_length,
            maxLineGap=max_line_gap,
        )
        
        if lines is None or len(lines) == 0:
            return 0.0, None
        
        angles = []
        lengths = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx * dx + dy * dy)
            
            # Calculate angle from vertical (90 degrees)
            # atan2 gives angle from horizontal, so vertical is ±90°
            angle_from_horizontal = np.degrees(np.arctan2(dy, dx))
            
            # Convert to deviation from vertical
            # Vertical lines have angle_from_horizontal near ±90°
            if angle_from_horizontal > 0:
                angle_from_vertical = angle_from_horizontal - 90
            else:
                angle_from_vertical = angle_from_horizontal + 90
            
            # Only consider near-vertical lines (within 45° of vertical)
            if abs(angle_from_vertical) < 45:
                angles.append(angle_from_vertical)
                lengths.append(length)
        
        if not angles:
            return 0.0, None
        
        lengths = np.array(lengths)
        angles = np.array(angles)
        
        # Weight by line length
        weights = lengths / np.sum(lengths)
        detected_angle = float(np.sum(angles * weights))
        
        vis_image = None
        if visualize:
            vis_image = frame.copy()
            
            # Draw detected lines
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw the detected angle indicator (horizontal line showing correction)
            center_x = width // 2
            center_y = height // 2
            line_len = width // 3
            angle_rad = np.radians(detected_angle)
            
            x1 = int(center_x - line_len * np.cos(angle_rad))
            y1 = int(center_y - line_len * np.sin(angle_rad))
            x2 = int(center_x + line_len * np.cos(angle_rad))
            y2 = int(center_y + line_len * np.sin(angle_rad))
            
            cv2.line(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            cv2.putText(
                vis_image,
                f"Angle: {detected_angle:.2f} deg",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                vis_image,
                f"Lines: {len(angles)}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
        
        return detected_angle, vis_image
    
    def set_method(self, method: ExtractionMethod) -> None:
        """Set the extraction method."""
        self.method = method
        self._last_gaussian_params = None
    
    def cycle_method(self) -> ExtractionMethod:
        """Cycle to next extraction method."""
        self.method = self.method.next()
        self._last_gaussian_params = None
        return self.method
    
    def set_rotation_angle(self, angle: float) -> None:
        """Set rotation angle and recompute sampling coordinates."""
        self.rotation_angle = angle
        self._precompute_sampling_coords()
    
    def set_spectrum_y_center(self, y: int) -> None:
        """Set spectrum Y center and recompute sampling coordinates."""
        self.spectrum_y_center = y
        self._precompute_sampling_coords()
    
    def increase_perpendicular_width(self, step: int = 2) -> int:
        """Increase perpendicular sampling width."""
        self.perpendicular_width = min(
            self.perpendicular_width + step,
            self._perpendicular_width_max
        )
        self._precompute_sampling_coords()
        return self.perpendicular_width
    
    def decrease_perpendicular_width(self, step: int = 2) -> int:
        """Decrease perpendicular sampling width."""
        self.perpendicular_width = max(
            self.perpendicular_width - step,
            self._perpendicular_width_min
        )
        self._precompute_sampling_coords()
        return self.perpendicular_width
