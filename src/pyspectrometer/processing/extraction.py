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
        """Precompute rotation matrix for current rotation angle.
        
        We ROTATE THE IMAGE (not the crop box) to straighten the spectrum.
        getRotationMatrix2D(center, -angle, 1): OpenCV positive = CCW.
        So -angle rotates image CCW by angle, straightening CW-tilted stripes.
        The crop is a horizontal band in the ROTATED (straightened) image.
        """
        self._rotation_center = (self.frame_width // 2, self.frame_height // 2)
        self._rotation_matrix = cv2.getRotationMatrix2D(
            self._rotation_center, -self.rotation_angle, 1.0
        )
    
    def _rotate_frame(self, frame: np.ndarray) -> np.ndarray:
        """Rotate frame to straighten spectrum lines.
        
        Args:
            frame: Input frame (BGR or grayscale)
            
        Returns:
            Rotated frame with straightened spectrum
        """
        if abs(self.rotation_angle) < 0.01:
            return frame
        
        return cv2.warpAffine(
            frame, self._rotation_matrix,
            (self.frame_width, self.frame_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
    
    def extract(self, frame: np.ndarray) -> ExtractionResult:
        """Extract spectrum intensity from frame.
        
        Args:
            frame: Input frame as numpy array:
                   - BGR color: (height, width, 3) uint8
                   - Monochrome 8-bit: (height, width) uint8
                   - Monochrome 10/16-bit: (height, width) uint16
            
        Returns:
            ExtractionResult with intensity array and cropped frame
        """
        # Rotate frame to straighten spectrum lines
        rotated_frame = self._rotate_frame(frame)
        
        # Convert to grayscale float for processing
        gray = self._frame_to_gray(rotated_frame)
        
        intensity = self._extract_with_method(gray)
        
        # Use rotated frame for preview so lines appear straight
        cropped = self._create_cropped_preview(rotated_frame)
        
        return ExtractionResult(
            intensity=intensity,
            cropped_frame=cropped,
            method_used=self.method,
            rotation_angle=self.rotation_angle,
            perpendicular_width=self.perpendicular_width,
        )
    
    def _frame_to_gray(self, frame: np.ndarray) -> np.ndarray:
        """Convert frame to grayscale float32.
        
        Handles:
        - BGR color (3D uint8) -> grayscale (0-255)
        - Monochrome (2D uint8 or uint16) -> float32, PRESERVING original range
        
        Note: For 10-bit data, values are 0-1023. We do NOT scale to 0-255
        to preserve full dynamic range. Display code handles scaling separately.
        
        Args:
            frame: Input frame (2D or 3D)
            
        Returns:
            Grayscale float32 array (0-255 for 8-bit, 0-1023 for 10-bit, etc.)
        """
        if frame.ndim == 3:
            # Color frame: convert BGR to grayscale
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # Monochrome frame - preserve original bit depth
        return frame.astype(np.float32)
    
    def _extract_with_method(self, gray: np.ndarray) -> np.ndarray:
        """Apply selected extraction method."""
        match self.method:
            case ExtractionMethod.MEDIAN:
                return self._extract_median(gray)
            case ExtractionMethod.WEIGHTED_SUM:
                return self._extract_weighted_sum(gray)
            case ExtractionMethod.GAUSSIAN:
                return self._extract_gaussian(gray)
    
    def _extract_median(self, gray: np.ndarray) -> np.ndarray:
        """Extract using median along vertical slices.
        
        Preserves original bit depth (0-1023 for 10-bit, 0-255 for 8-bit).
        """
        half_perp = self.perpendicular_width // 2
        y_start = max(0, self.spectrum_y_center - half_perp)
        y_end = min(gray.shape[0], self.spectrum_y_center + half_perp + 1)
        
        # Extract the ROI and compute median along vertical axis
        roi = gray[y_start:y_end, :]
        intensity = np.median(roi, axis=0)
        
        # Return as float32, preserving original range
        return intensity.astype(np.float32)
    
    def _extract_weighted_sum(self, gray: np.ndarray) -> np.ndarray:
        """Extract using intensity-weighted sum along vertical slices.
        
        Preserves original bit depth (0-1023 for 10-bit, 0-255 for 8-bit).
        """
        half_perp = self.perpendicular_width // 2
        y_start = max(0, self.spectrum_y_center - half_perp)
        y_end = min(gray.shape[0], self.spectrum_y_center + half_perp + 1)
        
        # Extract the ROI
        roi = gray[y_start:y_end, :]
        
        # Compute background per column
        bg = np.percentile(roi, self.background_percentile, axis=0)
        roi_bg = np.maximum(roi - bg, 0)
        
        # Weighted sum per column
        total = np.sum(roi_bg, axis=0)
        total = np.maximum(total, 1e-6)  # Avoid division by zero
        weights = roi_bg / total
        intensity = np.sum(roi * weights, axis=0)
        
        # Return as float32, preserving original range
        return intensity.astype(np.float32)
    
    def _extract_gaussian(self, gray: np.ndarray) -> np.ndarray:
        """Extract by fitting Gaussian to each vertical slice.
        
        Preserves original bit depth (0-1023 for 10-bit, 0-255 for 8-bit).
        Returns the fitted amplitude (not normalized).
        """
        from scipy.optimize import curve_fit
        
        intensity = np.zeros(self.frame_width, dtype=np.float32)
        
        half_perp = self.perpendicular_width // 2
        y_start = max(0, self.spectrum_y_center - half_perp)
        y_end = min(gray.shape[0], self.spectrum_y_center + half_perp + 1)
        
        roi = gray[y_start:y_end, :]
        n_samples = roi.shape[0]
        y_fit = np.arange(n_samples, dtype=np.float32) - n_samples // 2
        
        # Detect intensity range for proper bounds
        max_possible = float(np.max(gray)) if gray.size > 0 else 255.0
        max_possible = max(max_possible, 255.0)  # At least 8-bit
        
        for x in range(self.frame_width):
            samples = roi[:, x]
            
            try:
                bg = np.percentile(samples, self.background_percentile)
                amplitude = np.max(samples) - bg
                center = y_fit[np.argmax(samples)]
                sigma = 3.0
                
                if self._last_gaussian_params is not None:
                    _, prev_center, prev_sigma, _ = self._last_gaussian_params
                    center = prev_center
                    sigma = prev_sigma
                
                p0 = [amplitude, center, sigma, bg]
                
                # Use detected max for bounds
                bounds = (
                    [0, -n_samples // 2, 0.5, 0],
                    [max_possible, n_samples // 2, n_samples // 2, max_possible]
                )
                
                popt, _ = curve_fit(
                    self._gaussian,
                    y_fit,
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
        
        # Return as float32, preserving original range (no normalization)
        return intensity.astype(np.float32)
    
    @staticmethod
    def _gaussian(x: np.ndarray, amplitude: float, center: float, sigma: float, background: float) -> np.ndarray:
        """1D Gaussian function for curve fitting."""
        return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2) + background
    
    def _create_cropped_preview(self, frame: np.ndarray) -> np.ndarray:
        """Create cropped preview region centered on spectrum.
        
        Handles both color (3D) and monochrome (2D) frames.
        Monochrome frames are converted to 3-channel grayscale for display.
        
        Uses self.spectrum_y_center to determine crop position.
        """
        half_crop = self.crop_height // 2
        y_start = max(0, self.spectrum_y_center - half_crop)
        y_end = min(frame.shape[0], self.spectrum_y_center + half_crop)
        
        cropped = frame[y_start:y_end, :].copy()
        
        # Convert monochrome to 3-channel for display
        if cropped.ndim == 2:
            # Scale to 8-bit if high bit-depth
            if cropped.dtype == np.uint16:
                max_val = cropped.max()
                if max_val > 255:
                    if max_val > 4095:
                        scale = 255.0 / 65535.0
                    elif max_val > 1023:
                        scale = 255.0 / 4095.0
                    else:
                        scale = 255.0 / 1023.0
                    cropped = (cropped.astype(np.float32) * scale).astype(np.uint8)
                else:
                    cropped = cropped.astype(np.uint8)
            # Convert grayscale to BGR for display
            cropped = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)
        
        return cropped
    
    def detect_angle(self, frame: np.ndarray, visualize: bool = False) -> tuple[float, int, Optional[np.ndarray]]:
        """Auto-detect spectrum rotation angle and Y center using gradient analysis.
        
        Analyzes horizontal gradients (edges of vertical stripes) and uses
        PCA to find the dominant orientation. Computes the optimal Y center
        on the ROTATED frame to ensure proper centering after correction.
        
        Args:
            frame: Input frame (BGR color or monochrome)
            visualize: If True, return visualization image
            
        Returns:
            Tuple of (detected_angle_degrees, optimal_y_center, visualization_image_or_None)
        """
        # Convert to grayscale, preserving bit depth for thresholding precision
        # 10-bit/16-bit: keep full range as float32 (no 8-bit scaling before Sobel)
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        elif frame.dtype == np.uint16:
            # Preserve 10/16-bit range - don't scale to 8-bit before thresholding
            gray = frame.astype(np.float32)
            print(f"[AutoLevel] uint16 frame: max_val={gray.max():.0f} (preserving bit depth)")
        elif frame.dtype == np.uint8:
            gray = frame.astype(np.float32)
        else:
            gray = np.asarray(frame, dtype=np.float32)
        height, width = gray.shape
        print(f"[AutoLevel] gray shape={gray.shape}, dtype={gray.dtype}, range=[{gray.min()},{gray.max()}]")
        
        # Compute horizontal gradient (detects vertical edges/stripes)
        # Use Sobel with larger kernel for noise robustness
        ksize = max(3, (min(width, height) // 80) | 1)
        if ksize % 2 == 0:
            ksize += 1
        ksize = min(ksize, 31)
        
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        
        # Compute gradient magnitude
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Threshold: sample first 20 rows for noise, use 3x max as threshold
        n_rows = min(20, height)
        noise_region = magnitude[:n_rows, :]
        noise_max = float(noise_region.max()) if noise_region.size > 0 else 0.0
        mag_thresh = max(noise_max * 3.0, magnitude.min() * 1.1)
        strong_mask = magnitude > mag_thresh
        
        # Get coordinates of strong gradient points
        y_coords, x_coords = np.where(strong_mask)
        
        if len(x_coords) < 10:
            return 0.0, height // 2, None
        
        # Weight points by gradient magnitude
        weights = magnitude[strong_mask]
        
        # Compute weighted centroid
        total_weight = np.sum(weights)
        cx = np.sum(x_coords * weights) / total_weight
        cy = np.sum(y_coords * weights) / total_weight
        
        # Center the coordinates
        x_centered = x_coords - cx
        y_centered = y_coords - cy
        
        # Compute weighted covariance matrix for PCA
        cov_xx = np.sum(weights * x_centered * x_centered) / total_weight
        cov_yy = np.sum(weights * y_centered * y_centered) / total_weight
        cov_xy = np.sum(weights * x_centered * y_centered) / total_weight
        
        cov_matrix = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]])
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # The eigenvector with largest eigenvalue is the principal direction
        # For vertical stripes, this should be near-vertical
        principal_idx = np.argmax(eigenvalues)
        principal_vec = eigenvectors[:, principal_idx]
        
        # Calculate angle from vertical (y-axis)
        # principal_vec is (vx, vy), vertical is (0, 1)
        # Angle from vertical = atan2(vx, vy)
        angle_from_vertical = np.degrees(np.arctan2(principal_vec[0], principal_vec[1]))
        
        # Normalize to [-45, 45] range (stripes should be near-vertical)
        if angle_from_vertical > 45:
            angle_from_vertical -= 90
        elif angle_from_vertical < -45:
            angle_from_vertical += 90
        
        detected_angle = float(angle_from_vertical)
        
        # Now rotate the frame and find Y center on the rotated image
        rotation_center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(rotation_center, -detected_angle, 1.0)
        rotated_gray = cv2.warpAffine(
            gray, rotation_matrix, (width, height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        
        # Find Y center using multiple methods and pick the best one:
        # 1. Weighted centroid of strong gradient points (best for spectrum stripes)
        # 2. Row with maximum intensity variation (horizontal gradients = vertical stripes)
        # 3. Row with maximum intensity sum (fallback)
        
        # Method 1: Use weighted centroid Y from gradient analysis (already computed)
        # This is the Y position where most vertical stripes are
        gradient_y_center = int(cy)
        
        # Method 2: Find row with maximum horizontal gradient sum (vertical edges)
        rotated_grad_x = cv2.Sobel(rotated_gray, cv2.CV_64F, 1, 0, ksize=ksize)
        row_gradient_sums = np.sum(np.abs(rotated_grad_x), axis=1)
        gradient_y_center_rotated = int(np.argmax(row_gradient_sums))
        
        # Method 3: Row with maximum intensity (original method, less reliable)
        row_intensity_sums = np.sum(rotated_gray, axis=1)
        intensity_y_center = int(np.argmax(row_intensity_sums))
        
        # Pick the gradient-based Y center (more reliable for spectrum stripes)
        # Use the rotated gradient analysis since that's where we'll extract from
        optimal_y_center = gradient_y_center_rotated
        
        print(f"[AutoLevel] Y candidates: gradient={gradient_y_center}, "
              f"rotated_gradient={gradient_y_center_rotated}, intensity={intensity_y_center}")
        print(f"[AutoLevel] Selected Y center: {optimal_y_center}")
        
        vis_image = None
        if visualize:
            print("[AutoLevel] Building visualization...")
            # Left panel: thresholded gradient (strong_mask as grayscale BGR)
            thresh_display = (strong_mask.astype(np.uint8) * 255)
            thresh_bgr = cv2.cvtColor(thresh_display, cv2.COLOR_GRAY2BGR)
            cv2.putText(thresh_bgr, "Thresholded", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Find contours and fit ellipses (on original frame coords, before rotation)
            thresh_u8 = strong_mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(thresh_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            ellipses: list[tuple] = []
            min_pts = 5
            min_axis = 5
            for cnt in contours:
                if len(cnt) >= min_pts:
                    try:
                        el = cv2.fitEllipse(cnt)
                        if el[1][0] >= min_axis and el[1][1] >= min_axis:
                            ellipses.append(el)
                    except cv2.error:
                        pass

            # Right panel: ORIGINAL frame with ellipses (same coords - no transform)
            # Ellipses are fitted on original; crop box is on ROTATED frame, so draw it on rotated
            vis_orig = frame.copy()
            if vis_orig.dtype == np.uint16:
                max_val = max(vis_orig.max(), 1)
                vis_orig = (vis_orig.astype(np.float32) * 255 / max_val).astype(np.uint8)
            if vis_orig.ndim == 2:
                vis_orig = cv2.cvtColor(vis_orig, cv2.COLOR_GRAY2BGR)

            # Draw ellipses in original coords (no transform - image and ellipses match)
            for el in ellipses:
                try:
                    cv2.ellipse(vis_orig, el, (0, 255, 0), 1)
                except cv2.error:
                    pass

            # Draw ROTATED frame with crop box (crop box uses optimal_y_center from rotated)
            rotated_frame = cv2.warpAffine(
                frame, rotation_matrix, (width, height),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )
            vis_rot = rotated_frame.copy()
            if vis_rot.dtype == np.uint16:
                max_val = max(vis_rot.max(), 1)
                vis_rot = (vis_rot.astype(np.float32) * 255 / max_val).astype(np.uint8)
            if vis_rot.ndim == 2:
                vis_rot = cv2.cvtColor(vis_rot, cv2.COLOR_GRAY2BGR)

            half_perp = self.perpendicular_width // 2
            half_crop = self.crop_height // 2
            y_top = max(0, optimal_y_center - half_crop)
            y_bottom = min(height - 1, optimal_y_center + half_crop)
            cv2.rectangle(vis_rot, (0, y_top), (width, y_bottom), (0, 255, 255), 2)
            cv2.line(vis_rot, (0, optimal_y_center), (width, optimal_y_center), (255, 255, 0), 2)

            cv2.putText(vis_rot, "Rotated + crop", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.putText(
                vis_orig,
                "Original + ellipses",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                vis_rot,
                f"Angle: {detected_angle:.2f} deg  Y: {optimal_y_center}",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                vis_rot,
                "Click or press key to close",
                (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                2,
            )

            vis_image = np.hstack([thresh_bgr, vis_orig, vis_rot])
            print("[AutoLevel] Visualization complete, returning")
        
        return detected_angle, optimal_y_center, vis_image
    
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
