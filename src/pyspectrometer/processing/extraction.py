"""Spectrum extraction with rotation correction and multiple methods."""

from dataclasses import dataclass
from enum import Enum, auto

import cv2
import numpy as np

from ..utils.display import scale_to_uint8
from .spectrum_transform import (
    SpectrumTransformParams,
    apply_forward_transform,
    crop_region,
    detect_orientation_and_offset,
    inverse_params,
    params_from_saved,
)


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
    """Result of spectrum extraction.

    intensity: float32 array in 0-1 range (normalized by sensor full scale).
    """

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
        spectrum_y_center: int | None = None,
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

        self._last_gaussian_params: np.ndarray | None = None

    def set_dimensions(self, frame_width: int, frame_height: int) -> None:
        """Update frame dimensions (e.g. after camera reports actual size)."""
        self.frame_width = frame_width
        self.frame_height = frame_height

    def _transform_params(self) -> SpectrumTransformParams:
        """Build transform params from saved rotation_angle and spectrum_y_center."""
        return params_from_saved(
            self.rotation_angle,
            self.spectrum_y_center,
            self.frame_height,
        )

    def _rotate_frame(self, frame: np.ndarray) -> np.ndarray:
        """Rotate frame to straighten spectrum lines. Uses spectrum_transform."""
        params = self._transform_params()
        return apply_forward_transform(frame, params)

    def extract(
        self,
        frame: np.ndarray,
        max_val: float | None = None,
    ) -> ExtractionResult:
        """Extract spectrum intensity from frame.

        Args:
            frame: Input frame as numpy array:
                   - BGR color: (height, width, 3) uint8
                   - Monochrome 8-bit: (height, width) uint8
                   - Monochrome 10/16-bit: (height, width) uint16
            max_val: Max sensor value for 0-1 normalization (e.g. 1023 for 10-bit).
                     If None, defaults to 1023 (10-bit sensor).

        Returns:
            ExtractionResult with intensity in 0-1 range and cropped frame
        """
        # Rotate frame to straighten spectrum lines
        rotated_frame = self._rotate_frame(frame)

        # Convert to grayscale float for processing
        gray = self._frame_to_gray(rotated_frame)

        intensity = self._extract_with_method(gray)

        # Normalize to 0-1 using sensor full scale
        scale = max_val if max_val is not None and max_val > 0 else 1023.0
        intensity = (intensity.astype(np.float32) / scale).astype(np.float32)

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
        Uses spectrum_y_center (absolute Y in rotated frame) - same offset as crop.
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
        total = np.sum(roi_bg, axis=0)

        # Saturated flat-top: when all pixels ≈ max, roi_bg ≈ 0, total ≈ 0.
        # weights would be 0 → intensity = 0. Fallback to max(roi) for those columns.
        total_safe = np.maximum(total, 1e-6)
        weights = roi_bg / total_safe
        intensity_weighted = np.sum(roi * weights, axis=0)
        intensity_fallback = np.max(roi, axis=0).astype(np.float64)
        intensity = np.where(total > 1e-6, intensity_weighted, intensity_fallback)

        return intensity.astype(np.float32)

    def _extract_gaussian(self, gray: np.ndarray) -> np.ndarray:
        """Extract by fitting Gaussian to each vertical slice.

        Preserves original bit depth (0-1023 for 10-bit, 0-255 for 8-bit).
        Returns the fitted amplitude (not normalized).
        Uses spectrum_y_center (absolute Y in rotated frame) - same offset as crop.
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
                    [max_possible, n_samples // 2, n_samples // 2, max_possible],
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
    def _gaussian(
        x: np.ndarray, amplitude: float, center: float, sigma: float, background: float
    ) -> np.ndarray:
        """1D Gaussian function for curve fitting."""
        return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2) + background

    def _create_cropped_preview(self, frame: np.ndarray) -> np.ndarray:
        """Create cropped preview region centered on spectrum. Uses spectrum_transform."""
        cropped = crop_region(frame, self.spectrum_y_center, self.crop_height)

        # Convert monochrome to 3-channel for display
        if cropped.ndim == 2:
            # Scale to 8-bit if high bit-depth (use bit-depth-based max for consistency)
            if cropped.dtype == np.uint16:
                m = cropped.max()
                max_val = 65535.0 if m > 4095 else (4095.0 if m > 1023 else 1023.0)
                cropped = scale_to_uint8(cropped, max_val)
            # Convert grayscale to BGR for display
            cropped = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)

        return cropped

    def detect_angle(
        self, frame: np.ndarray, visualize: bool = False
    ) -> tuple[float, int, np.ndarray | None]:
        """Auto-detect spectrum rotation angle and Y center using gradient analysis.

        Analyzes horizontal gradients (edges of vertical stripes) and uses
        PCA to find the dominant orientation. Computes the optimal Y center
        on the ROTATED frame to ensure proper centering after correction.

        Args:
            frame: Input frame (BGR color or monochrome)
            visualize: If True, return visualization image

        Returns:
            Tuple of (correction_angle, spectrum_y_center, visualization_or_None)
            correction_angle is the inverse of orientation (what we save).
        """
        result = detect_orientation_and_offset(frame, return_debug=visualize)
        if visualize and len(result) == 6:
            orientation, y_offset, spectrum_y, strong_mask, gray, rotated_gray = result
        else:
            orientation, y_offset, spectrum_y = result[:3]
            strong_mask = gray = rotated_gray = None
        params = inverse_params(orientation, y_offset)
        correction_angle = params.rotation_angle
        optimal_y_center = spectrum_y
        height, width = self.frame_height, self.frame_width

        print(f"[AutoLevel] Orientation: {orientation:.2f}°  Offset: {y_offset:.1f}px")
        print(f"[AutoLevel] Correction: angle={correction_angle:.2f}°  Y center={optimal_y_center}")

        vis_image = None
        if visualize and strong_mask is not None and gray is not None and rotated_gray is not None:
            print("[AutoLevel] Building visualization...")
            # Left panel: thresholded gradient (strong_mask as grayscale BGR)
            thresh_display = strong_mask.astype(np.uint8) * 255
            thresh_bgr = cv2.cvtColor(thresh_display, cv2.COLOR_GRAY2BGR)
            cv2.putText(
                thresh_bgr, "Thresholded", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )

            # Find contours and fit ellipses; keep only narrow vertical stripes
            thresh_u8 = strong_mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(thresh_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            ellipses: list[tuple] = []
            min_pts = 5
            min_axis = 5
            # Slits are narrow vertical stripes: minor/major axis ratio must be small
            max_axis_ratio = 0.4  # Keep only if min/max <= 0.4 (narrow vertical)
            for cnt in contours:
                if len(cnt) >= min_pts:
                    try:
                        el = cv2.fitEllipse(cnt)
                        w, h = el[1][0], el[1][1]
                        if w < min_axis or h < min_axis:
                            continue
                        minor, major = min(w, h), max(w, h)
                        if minor / major <= max_axis_ratio:
                            ellipses.append(el)
                    except cv2.error:
                        pass

            # Right panel: ORIGINAL frame with ellipses (same coords - no transform)
            vis_orig = frame.copy()
            if vis_orig.dtype == np.uint16:
                vis_orig = scale_to_uint8(vis_orig)
            if vis_orig.ndim == 2:
                vis_orig = cv2.cvtColor(vis_orig, cv2.COLOR_GRAY2BGR)

            # Draw ellipses in original coords (no transform - image and ellipses match)
            for el in ellipses:
                try:
                    cv2.ellipse(vis_orig, el, (0, 255, 0), 1)
                except cv2.error:
                    pass

            # Draw ROTATED frame with crop box (uses spectrum_transform forward)
            rotated_frame = apply_forward_transform(frame, params)
            vis_rot = rotated_frame.copy()
            if vis_rot.dtype == np.uint16:
                vis_rot = scale_to_uint8(vis_rot)
            if vis_rot.ndim == 2:
                vis_rot = cv2.cvtColor(vis_rot, cv2.COLOR_GRAY2BGR)

            half_crop = self.crop_height // 2
            y_top = max(0, optimal_y_center - half_crop)
            y_bottom = min(height - 1, optimal_y_center + half_crop)
            cv2.rectangle(vis_rot, (0, y_top), (width, y_bottom), (0, 255, 255), 2)
            cv2.line(vis_rot, (0, optimal_y_center), (width, optimal_y_center), (255, 255, 0), 2)

            cv2.putText(
                vis_rot, "Rotated + crop", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )

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
                f"Angle: {correction_angle:.2f} deg  Y: {optimal_y_center}",
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

        return correction_angle, optimal_y_center, vis_image

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
        """Set rotation angle (correction = inverse of orientation)."""
        self.rotation_angle = angle

    def set_spectrum_y_center(self, y: int) -> None:
        """Set spectrum Y center in corrected frame."""
        self.spectrum_y_center = y

    def set_perpendicular_width(self, width: int) -> None:
        """Set perpendicular sampling width (clamped to valid range)."""
        self.perpendicular_width = max(
            self._perpendicular_width_min,
            min(width, self._perpendicular_width_max),
        )

    def increase_perpendicular_width(self, step: int = 2) -> int:
        """Increase perpendicular sampling width."""
        self.perpendicular_width = min(
            self.perpendicular_width + step, self._perpendicular_width_max
        )
        return self.perpendicular_width

    def decrease_perpendicular_width(self, step: int = 2) -> int:
        """Decrease perpendicular sampling width."""
        self.perpendicular_width = max(
            self.perpendicular_width - step, self._perpendicular_width_min
        )
        return self.perpendicular_width
