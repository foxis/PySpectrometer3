"""Spectrum transform: orientation detection, correction, crop, and display bbox.

Workflow:
1. Detect ORIENTATION of spectrum stripes (angle from vertical)
2. Detect OFFSET of spectrum center from frame center (Y pixels)
3. INVERSE both: save correction_angle = -orientation, correction_offset = -offset
4. Apply forward transform: rotate by correction_angle, then crop at center + correction_offset
5. For display: transform crop bbox from corrected coords to original coords

All transforms use cv2.getRotationMatrix2D and cv2.warpAffine (no manual math).
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SpectrumTransformParams:
    """Saved correction parameters (inverse of detected orientation/offset).

    Apply these to restore stripe verticalness:
    - rotation_angle: degrees, correction rotation (inverse of stripe tilt)
    - y_offset: pixels, offset from frame center for crop (inverse of spectrum offset)
    - spectrum_y_center: absolute Y in corrected frame = height/2 + y_offset
    """

    rotation_angle: float  # degrees
    y_offset: float  # pixels from frame center (can be negative)

    def spectrum_y_center(self, height: int) -> int:
        """Crop center Y in corrected frame."""
        return int(height / 2 + self.y_offset)


def detect_orientation_and_offset(
    frame: np.ndarray,
    ksize: int | None = None,
    return_debug: bool = False,
) -> tuple[float, float, int] | tuple[float, float, int, np.ndarray, np.ndarray, np.ndarray]:
    """Detect stripe orientation and spectrum center offset from frame center.

    Uses gradient analysis + PCA on spectrum stripe edges.
    Returns values in ORIGINAL frame: orientation angle, offset, spectrum Y in rotated.

    Args:
        frame: Input frame (BGR or monochrome)
        ksize: Sobel kernel size (odd, default auto)

    Returns:
        If return_debug False: (orientation, y_offset, spectrum_y_in_rotated)
        If return_debug True: same plus (strong_mask, gray, rotated_gray)
        - orientation: stripe tilt from vertical (positive = CW)
        - y_offset: spectrum_center_y - height/2 in original
        - spectrum_y_in_rotated: Y row in corrected frame for crop center
    """
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray = frame.astype(np.float32)
    height, width = gray.shape

    ksize = ksize or max(3, (min(width, height) // 80) | 1)
    if ksize % 2 == 0:
        ksize += 1
    ksize = min(ksize, 31)

    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # 50% threshold: keep only top half of gradient magnitudes for narrower peaks
    # and more precise angle extraction (less liberal than noise-based threshold)
    mag_thresh = float(np.percentile(magnitude, 50))
    strong_mask = magnitude > mag_thresh

    y_coords, x_coords = np.where(strong_mask)
    if len(x_coords) < 10:
        if return_debug:
            return 0.0, 0.0, height // 2, strong_mask, gray, gray.copy()
        return 0.0, 0.0, height // 2

    weights = magnitude[strong_mask]
    total_weight = float(np.sum(weights))
    cx = np.sum(x_coords * weights) / total_weight
    cy = np.sum(y_coords * weights) / total_weight

    x_centered = x_coords - cx
    y_centered = y_coords - cy
    cov_xx = np.sum(weights * x_centered * x_centered) / total_weight
    cov_yy = np.sum(weights * y_centered * y_centered) / total_weight
    cov_xy = np.sum(weights * x_centered * y_centered) / total_weight
    cov_matrix = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]])
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    principal_idx = np.argmax(eigenvalues)
    principal_vec = eigenvectors[:, principal_idx]

    angle_from_vertical = np.degrees(np.arctan2(principal_vec[0], principal_vec[1]))
    if angle_from_vertical > 45:
        angle_from_vertical -= 90
    elif angle_from_vertical < -45:
        angle_from_vertical += 90
    orientation = float(angle_from_vertical)

    center = (width / 2, height / 2)
    rot_matrix = cv2.getRotationMatrix2D(center, -orientation, 1.0)
    rotated_gray = cv2.warpAffine(
        gray, rot_matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    rotated_grad_x = cv2.Sobel(rotated_gray, cv2.CV_64F, 1, 0, ksize=ksize)
    row_sums = np.sum(np.abs(rotated_grad_x), axis=1)
    spectrum_y_rotated = int(np.argmax(row_sums))

    frame_center_y = height / 2
    y_offset_from_center = float(spectrum_y_rotated - frame_center_y)

    if return_debug:
        return (
            orientation,
            y_offset_from_center,
            spectrum_y_rotated,
            strong_mask,
            gray,
            rotated_gray,
        )
    return orientation, y_offset_from_center, spectrum_y_rotated


def inverse_params(
    orientation_angle: float,
    y_offset_from_center: float,
) -> SpectrumTransformParams:
    """Compute saved params: inverse of rotation and inverse of offset."""
    return SpectrumTransformParams(
        rotation_angle=-orientation_angle,
        y_offset=-y_offset_from_center,
    )


def build_forward_matrix(
    width: int,
    height: int,
    params: SpectrumTransformParams,
) -> np.ndarray:
    """Build 2x3 affine matrix for forward transform (restore stripe verticalness).

    Uses cv2.getRotationMatrix2D. Applies rotation only (offset handled by crop center).
    """
    center = (width / 2, height / 2)
    return cv2.getRotationMatrix2D(center, params.rotation_angle, 1.0)


def apply_forward_transform(
    frame: np.ndarray,
    params: SpectrumTransformParams,
) -> np.ndarray:
    """Apply forward transform: rotate to straighten spectrum. Uses cv2.warpAffine."""
    height, width = frame.shape[:2]
    if abs(params.rotation_angle) < 0.01:
        return frame
    M = build_forward_matrix(width, height, params)
    return cv2.warpAffine(
        frame, M, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def crop_region(
    frame: np.ndarray,
    center_y: int,
    crop_height: int,
) -> np.ndarray:
    """Crop horizontal band centered at center_y. Returns frame[y1:y2, :]."""
    half = crop_height // 2
    y1 = max(0, center_y - half)
    y2 = min(frame.shape[0], center_y + half)
    return frame[y1:y2, :].copy()


def build_inverse_matrix(
    width: int,
    height: int,
    params: SpectrumTransformParams,
) -> np.ndarray:
    """Build 2x3 matrix for rotated -> original (for display bbox). Uses getRotationMatrix2D.

    Forward uses R(params.rotation_angle). Inverse is R(-params.rotation_angle).
    """
    center = (width / 2, height / 2)
    return cv2.getRotationMatrix2D(center, -params.rotation_angle, 1.0)


def transform_bbox_rotated_to_original(
    corners_rotated: np.ndarray,
    params: SpectrumTransformParams,
    width: int,
    height: int,
) -> np.ndarray:
    """Transform crop bbox corners from corrected (rotated) coords to original image coords.

    Forward: orig -> rotated via R(params.rotation_angle).
    Inverse: rotated -> orig via R(-params.rotation_angle).
    Uses cv2.getRotationMatrix2D and cv2.transform (no manual cos/sin).

    Args:
        corners_rotated: Nx2 array of (x, y) in corrected frame
        params: Saved correction params
        width: Original frame width
        height: Original frame height

    Returns:
        Nx2 array of (x, y) in original frame
    """
    center = (width / 2, height / 2)
    M = cv2.getRotationMatrix2D(center, -params.rotation_angle, 1.0)
    pts = corners_rotated.astype(np.float32).reshape(-1, 1, 2)
    transformed = cv2.transform(pts, M)
    return transformed.reshape(-1, 2)


def get_crop_corners_rotated(
    width: int,
    center_y: int,
    crop_height: int,
) -> np.ndarray:
    """Get crop box corners in rotated frame: (x1,y1), (x2,y2), (x3,y3), (x4,y4)."""
    half = crop_height // 2
    y_top = center_y - half
    y_bot = center_y + half
    return np.array(
        [
            [0, y_top],
            [width, y_top],
            [width, y_bot],
            [0, y_bot],
        ],
        dtype=np.float32,
    )


def params_from_saved(
    rotation_angle: float,
    spectrum_y_center: int,
    height: int,
) -> SpectrumTransformParams:
    """Build params from saved calibration values (rotation_angle, spectrum_y_center)."""
    y_offset = spectrum_y_center - (height / 2)
    return SpectrumTransformParams(rotation_angle=rotation_angle, y_offset=y_offset)


def detect_and_compute_params(
    frame: np.ndarray,
) -> tuple[SpectrumTransformParams, int]:
    """Detect orientation/offset, inverse them, return params and spectrum_y_center."""
    orientation, y_offset, spectrum_y = detect_orientation_and_offset(frame)
    params = inverse_params(orientation, y_offset)
    return params, spectrum_y
