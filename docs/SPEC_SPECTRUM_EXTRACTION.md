# Spectrum Extraction Algorithm - Design Specification

**Status: IMPLEMENTED**

## Overview

Enhanced spectrum extraction that handles:
1. **Rotated spectrum lines** (5-15 degree rotation)
2. **Vertical structure due to coupling** (non-uniform intensity perpendicular to dispersion axis)
3. **Wide dynamic range** (very low to very high signals)

## Architecture

### New Module: `pyspectrometer/processing/extraction.py`

```
┌─────────────────────────────────────────────────────────────────┐
│                    SpectrumExtractor                            │
├─────────────────────────────────────────────────────────────────┤
│ - rotation_angle: float (degrees, from calibration)            │
│ - perpendicular_width: int (pixels to sample perpendicular)    │
│ - method: ExtractionMethod (MEDIAN, WEIGHTED_SUM, GAUSSIAN)    │
│ - background_threshold: float (for adaptive window)            │
├─────────────────────────────────────────────────────────────────┤
│ + extract(frame) -> (cropped, intensity)                       │
│ + detect_angle(frame) -> float (auto-detect via Hough)         │
│ + set_method(method: ExtractionMethod)                         │
└─────────────────────────────────────────────────────────────────┘
```

### Extraction Methods

#### 1. Weighted Sum (Default)
For each x-position along the rotated spectrum:
- Sample pixels perpendicular to spectrum axis
- Apply intensity-weighted sum: `sum(I * I) / sum(I)` or simple sum
- Normalizes to original intensity range

**Pros**: Best S/N ratio, captures all light
**Best for**: General use, low-to-medium signals

#### 2. Median
For each x-position along the rotated spectrum:
- Sample pixels perpendicular to spectrum axis
- Take median value

**Pros**: Robust to outliers, hot pixels, cosmic rays
**Best for**: Noisy conditions, high signal levels

#### 3. Gaussian Fitting
For each x-position along the rotated spectrum:
- Sample pixels perpendicular to spectrum axis
- Fit 1D Gaussian: `A * exp(-(x-μ)²/(2σ²)) + B`
- Use amplitude `A` as intensity (or integrated area `A * σ * sqrt(2π)`)

**Pros**: Most accurate, separates signal from background
**Best for**: Precision measurements, publication-quality data

### Angle Detection (Calibration)

Using Hough Line Transform:
1. Convert frame to grayscale
2. Apply edge detection (Canny)
3. Run probabilistic Hough transform
4. Filter lines by length and position (center region)
5. Calculate dominant angle from detected lines
6. Store angle in calibration data

### Data Flow

```
┌──────────┐    ┌───────────────────┐    ┌──────────────────┐
│  Frame   │───▶│ SpectrumExtractor │───▶│ (cropped, intens)│
└──────────┘    └───────────────────┘    └──────────────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │ 1. Apply rotation   │
              │ 2. Sample perp line │
              │ 3. Apply method     │
              │ 4. Return intensity │
              └─────────────────────┘
```

## Configuration Changes

### `ExtractionConfig` (new dataclass in config.py)

```python
@dataclass
class ExtractionConfig:
    method: str = "weighted_sum"  # "median", "weighted_sum", "gaussian"
    rotation_angle: float = 0.0   # degrees, loaded from calibration
    perpendicular_width: int = 20  # pixels to sample perpendicular to axis
    background_percentile: float = 10.0  # for background subtraction
    gaussian_sigma_init: float = 3.0  # initial sigma for Gaussian fit
```

### Calibration File Format (extended)

Current format:
```
pixels: 100,300,500,700
wavelengths: 400.0,500.0,600.0,700.0
```

Extended format:
```
pixels: 100,300,500,700
wavelengths: 400.0,500.0,600.0,700.0
rotation_angle: 7.5
spectrum_y_center: 240
perpendicular_width: 25
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| `e` | Cycle extraction method (median → weighted_sum → gaussian) |
| `E` (Shift+e) | Auto-detect rotation angle |
| `[` | Decrease perpendicular width |
| `]` | Increase perpendicular width |

## Implementation Steps

### Phase 1: Core Extraction Module
1. Create `ExtractionMethod` enum
2. Create `SpectrumExtractor` class with 3 methods
3. Implement rotation-corrected sampling
4. Unit tests with synthetic data

### Phase 2: Angle Detection
1. Implement Hough-based angle detection
2. Add angle to calibration data
3. Add keyboard shortcut for auto-detect
4. Visual feedback during detection

### Phase 3: Integration
1. Add `ExtractionConfig` to Config
2. Replace `CameraInterface.extract_spectrum_region` usage
3. Update DisplayManager to show current method
4. Add keyboard controls

### Phase 4: Calibration UI
1. Show detected angle during calibration
2. Allow manual angle adjustment
3. Save/load angle with calibration data

## Performance Considerations

- **Median**: O(n log n) per column, fast
- **Weighted Sum**: O(n) per column, fastest
- **Gaussian**: O(iterations * n) per column, slowest (use scipy.optimize.curve_fit)

For real-time (30 fps @ 800 pixels):
- Median/Weighted: < 1ms total
- Gaussian: ~10-50ms total (may need to run every N frames or in background)

### Optimization Strategy for Gaussian
1. Use previous frame's fit parameters as initial guess
2. Limit iterations (maxfev=100)
3. Pre-allocate arrays
4. Consider running on subset of columns and interpolating

## Testing

### Synthetic Test Cases
1. Horizontal spectrum (0° rotation) - baseline
2. Rotated spectrum (10° rotation)
3. Spectrum with hot pixels (median should handle)
4. Low signal spectrum (Gaussian should extract cleanly)
5. Saturated spectrum (all methods should handle)

### Validation
- Compare extracted intensity to known reference
- Measure S/N ratio improvement vs simple row average
- Profile performance of each method
