# PySpectrometer3 Operating Modes Specification

## Overview

PySpectrometer3 operates in four distinct modes, each with specialized functionality and GUI controls. All modes share common camera/acquisition controls but have mode-specific features.

## Mode Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PySpectrometer3                              │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │ Calibration │  │ Measurement │  │    Raman    │  │   Color     │ │
│  │    Mode     │  │    Mode     │  │    Mode     │  │  Science    │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘ │
│         │                │                │                │        │
│         └────────────────┴────────────────┴────────────────┘        │
│                                  │                                  │
│                    ┌─────────────┴─────────────┐                    │
│                    │     Common Controls       │                    │
│                    │  (Camera, Save/Load, etc) │                    │
│                    └───────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────────┘
```

## Common Controls (All Modes Except Calibration)

| Control | Description |
|---------|-------------|
| Save Spectrum | Save current spectrum to CSV |
| Load Spectrum | Load saved spectrum as reference |
| Average | Toggle averaging mode (accumulate frames) |
| Set Black | Capture dark/black reference |
| Set White | Capture white reference (100% baseline) |
| Gain +/- | Manual gain adjustment |
| Auto Gain | Automatic gain control |
| Exposure +/- | Manual exposure adjustment (if camera supports) |
| Auto Exposure | Automatic exposure control |
| Lamp On/Off | GPIO 22 light control |

---

## Mode 1: Calibration

### Purpose
Perform wavelength calibration using known spectral lines from reference light sources.

### Reference Sources
| Name | Description | Key Lines (nm) |
|------|-------------|----------------|
| FL | Compact Fluorescent Lamp | 405, 436, 546, 578, 611 (Hg + rare earth) |
| Hg | Mercury Low-Pressure Lamp | 404.7, 435.8, 546.1, 579.0 |
| Sun | Solar Spectrum | Fraunhofer lines: 486.1 (F), 589.3 (D), 656.3 (C) |

### Calibration Workflow
1. Select reference source (FL, Hg, Sun)
2. Auto-detect peaks in spectrum
3. Match detected peaks to known reference lines (best-fit algorithm)
4. User confirms/adjusts 4 calibration points (minimum)
5. Compute polynomial fit (3rd order for 4 points)
6. Save calibration

### GUI Controls

| Row | Controls |
|-----|----------|
| 1 | [FL] [Hg] [Sun] │ [Auto-Level] [Auto-Center] [Auto-Calibrate] │ Status |
| 2 | [Save Cal] [Clear Pts] │ Points: 0/4 │ Fit Error: -- |

### Algorithms

#### Auto-Level
Adjust gain to bring peak intensity to target range (200-240 out of 255).

#### Auto-Center Y
Find the vertical center of the spectrum band:
1. Sum intensity along X axis for each Y row
2. Find Y with maximum total intensity
3. Set spectrum_y_center to that value

#### Auto-Calibrate (Best-Fit Matching)

```python
def match_peaks_to_reference(
    detected_peaks: list[float],  # pixel positions
    reference_lines: list[float],  # known wavelengths (nm)
    initial_calibration: Calibration,
) -> list[tuple[float, float]]:  # (pixel, wavelength) pairs
    """
    Match detected peaks to reference spectral lines.
    
    Algorithm:
    1. Convert detected pixel positions to approximate wavelengths
       using initial (or default) calibration
    2. For each reference line, find nearest detected peak
    3. Score matches by proximity and consistency
    4. Return best N matches (N >= 4)
    """
```

This matching algorithm will be reused for:
- Material identification (match unknown to library)
- Reference spectrum alignment
- Raman peak identification

### Calibration Points
**CRITICAL: Use 4 points minimum (not 3)**
- 3 points → 2nd order polynomial (quadratic) - insufficient
- 4 points → 3rd order polynomial (cubic) - better accuracy
- 5+ points → 3rd order with least squares fit - best

---

## Mode 2: Measurement

### Purpose
General spectrum measurement with reference normalization.

### Features
- Capture and display live spectrum
- Normalize to reference spectrum
- Compare with saved spectra
- Identify materials (future: spectral matching)

### GUI Controls

| Row | Controls |
|-----|----------|
| 1 | [Save] [Load] [Set Ref] [Clear Ref] │ [Average] [Black] [White] │ Ref: None |
| 2 | [Gain+] [Gain-] [AutoG] │ [Lamp] │ Gain: 25 │ Avg: 1 |

### Processing Pipeline

```
Raw Frame → Dark Subtraction → White Normalization → Display
              ↓                     ↓
         (if black set)      (if white set)
         
Normalized = (Raw - Black) / (White - Black)
```

---

## Mode 3: Raman

### Purpose
Raman spectroscopy with automatic laser line detection and wavenumber conversion.

### Configuration
```ini
[raman]
laser_wavelength_nm = 785.0
laser_detection_range_nm = 5.0
wavenumber_range_min = 200
wavenumber_range_max = 3200
```

### Features
- Auto-detect laser line (785nm peak)
- Convert wavelength to Raman shift (wavenumber)
- Display in cm⁻¹ (Raman shift)
- Background fluorescence subtraction

### Wavenumber Calculation

```python
def wavelength_to_wavenumber(
    wavelength_nm: float,
    laser_nm: float = 785.0,
) -> float:
    """Convert wavelength to Raman shift in cm⁻¹."""
    return (1.0 / laser_nm - 1.0 / wavelength_nm) * 1e7
```

### GUI Controls

| Row | Controls |
|-----|----------|
| 1 | [Save] [Load] [Set Ref] │ [Average] [Black] │ Laser: 785nm |
| 2 | [Gain+] [Gain-] [AutoG] │ [Lamp] │ [Find Laser] │ Shift: 0-3200 cm⁻¹ |

---

## Mode 4: Color Science

### Purpose
Colorimetric analysis: transmittance, reflectance, illuminance, CRI calculation.

### Sub-Modes
| Sub-Mode | Description | Calculation |
|----------|-------------|-------------|
| Transmittance | Light passing through sample | T = I_sample / I_reference |
| Reflectance | Light reflected from sample | R = I_sample / I_white |
| Illuminance | Light source characterization | Direct spectrum display |

### Reference Data Required
- **D65 Daylight** - CIE Standard Illuminant D65
- **CIE 1931 2° Observer** - x̄, ȳ, z̄ color matching functions
- **CIE 1964 10° Observer** - x̄₁₀, ȳ₁₀, z̄₁₀ (preferred for large field)
- **Test Color Samples (TCS)** - For CRI calculation

### Tristimulus Calculation

```python
def calculate_XYZ(
    spectrum: np.ndarray,       # Measured spectrum
    wavelengths: np.ndarray,    # Wavelength array
    observer: str = "10deg",    # "2deg" or "10deg"
) -> tuple[float, float, float]:
    """Calculate CIE XYZ tristimulus values.
    
    X = k * Σ S(λ) * x̄(λ) * Δλ
    Y = k * Σ S(λ) * ȳ(λ) * Δλ
    Z = k * Σ S(λ) * z̄(λ) * Δλ
    
    where k = 100 / Σ S_ref(λ) * ȳ(λ) * Δλ
    """
```

### CRI Calculation

Color Rendering Index calculation per CIE 13.3:
1. Calculate u, v chromaticity of test source
2. Calculate correlated color temperature (CCT)
3. Generate reference illuminant (Planckian or D-illuminant)
4. Calculate color shift for 8 (or 14) test color samples
5. Ra = average of R1-R8

### GUI Controls

| Row | Controls |
|-----|----------|
| 1 | [Trans] [Refl] [Illum] │ [D65 Ref] [XYZ Ref] │ [Normalize] │ Mode: Trans |
| 2 | [Save] [Load] [Black] [White] │ [Lamp] │ CRI: -- │ CCT: -- K |

### Display Overlays
- Toggle D65 reference curve overlay
- Toggle X, Y, Z color matching function overlays
- Show calculated XYZ values
- Show CRI and CCT for illuminance mode

---

## Data Files Required

### Calibration References
```
data/references/
├── FL_CFL_spectrum.csv        # Compact fluorescent lines
├── Hg_low_pressure.csv        # Mercury lamp lines  
├── Solar_fraunhofer.csv       # Solar Fraunhofer lines
└── reference_lines.json       # Combined reference line database
```

### Color Science Data
```
data/colorscience/
├── CIE_D65_1nm.csv            # D65 daylight illuminant
├── CIE_xyz_1931_2deg_1nm.csv  # 2° observer CMFs
├── CIE_xyz_1964_10deg_1nm.csv # 10° observer CMFs
├── CIE_TCS_reflectance.csv    # Test color samples for CRI
└── planckian_locus.csv        # Planckian radiator data
```

---

## Module Structure

```
src/pyspectrometer/
├── modes/
│   ├── __init__.py
│   ├── base.py                # BaseMode abstract class
│   ├── calibration.py         # CalibrationMode
│   ├── measurement.py         # MeasurementMode
│   ├── raman.py               # RamanMode
│   └── colorscience.py        # ColorScienceMode
├── features/
│   ├── __init__.py
│   ├── auto_gain.py           # Auto gain/exposure
│   ├── dark_spectrum.py       # Dark/black reference
│   ├── white_reference.py     # White reference
│   ├── averaging.py           # Frame averaging
│   ├── gpio_control.py        # GPIO lamp control
│   └── spectrum_matching.py   # Peak matching algorithm
├── colorscience/
│   ├── __init__.py
│   ├── tristimulus.py         # XYZ calculation
│   ├── cri.py                 # CRI calculation
│   ├── cct.py                 # CCT calculation
│   └── data_loader.py         # Load CIE data files
└── ...existing modules...
```

---

## Command Line Interface

```bash
# Launch specific mode
python -m pyspectrometer --mode calibration
python -m pyspectrometer --mode measurement
python -m pyspectrometer --mode raman
python -m pyspectrometer --mode colorscience

# Additional options
python -m pyspectrometer --mode colorscience --submode transmittance
python -m pyspectrometer --mode raman --laser 785

# Waveshare display (default for install-link)
python -m pyspectrometer --waveshare --mode measurement
```

---

## Desktop Links (make install-link)

### Installation
```bash
make install-link
```

Creates executable desktop files in `~/.local/share/applications/`:

| Link | Command | Description |
|------|---------|-------------|
| pyspec-calibration.desktop | `--waveshare --mode calibration` | Calibration Mode |
| pyspec-measurement.desktop | `--waveshare --mode measurement` | Measurement Mode |
| pyspec-raman.desktop | `--waveshare --mode raman` | Raman Mode |
| pyspec-colorscience.desktop | `--waveshare --mode colorscience` | Color Science Mode |
| pyspec.desktop | `--waveshare` | Default (Measurement) |

### Desktop File Template
```ini
[Desktop Entry]
Name=PySpectrometer - Measurement
Comment=Spectrum Measurement Mode
Exec=/home/pi/PySpectrometer3/venv/bin/python -m pyspectrometer --waveshare --mode measurement
Icon=/home/pi/PySpectrometer3/assets/icon.png
Terminal=false
Type=Application
Categories=Science;Education;
```

### Shell Scripts
Also create executable shell scripts in `/usr/local/bin/`:
```bash
#!/bin/bash
# /usr/local/bin/pyspec-measurement
cd /home/pi/PySpectrometer3
./venv/bin/python -m pyspectrometer --waveshare --mode measurement
```

---

## Implementation Phases

### Phase 1: Mode Framework
1. Create `modes/base.py` with `BaseMode` abstract class
2. Define common interface for all modes
3. Update `Spectrometer` to use mode system
4. Update CLI to accept `--mode` argument

### Phase 2: Common Features
1. Implement `features/dark_spectrum.py`
2. Implement `features/white_reference.py`
3. Implement `features/averaging.py`
4. Implement `features/auto_gain.py`
5. Implement `features/gpio_control.py`

### Phase 3: Calibration Mode
1. Implement `modes/calibration.py`
2. Add reference spectrum database
3. Implement peak matching algorithm
4. Update calibration to 4-point minimum
5. Add auto-level, auto-center, auto-calibrate

### Phase 4: Measurement Mode
1. Implement `modes/measurement.py`
2. Integrate common features
3. Add reference normalization

### Phase 5: Raman Mode
1. Implement `modes/raman.py`
2. Add wavenumber conversion
3. Implement laser line detection

### Phase 6: Color Science Mode
1. Implement `colorscience/` module
2. Load CIE data files
3. Implement XYZ calculation
4. Implement CRI calculation
5. Implement `modes/colorscience.py`

### Phase 7: Desktop Integration
1. Update Makefile with `install-link` target
2. Create desktop files
3. Create shell scripts
4. Test all modes

---

## Open Questions

1. **Exposure control**: Does Picamera2 support separate exposure control, or only gain?
   - Need to verify camera capabilities

2. **CRI calculation**: Full CRI (R1-R14) or basic (R1-R8)?
   - Recommendation: Implement R1-R8 first, extend later

3. **Raman baseline correction**: Which algorithm?
   - Options: Polynomial fit, adaptive iterative reweighted penalized least squares (airPLS)
   - Recommendation: Start with polynomial, add airPLS later

4. **Mode switching at runtime**: Allow switching modes without restart?
   - Recommendation: Yes, via GUI button or keyboard shortcut

---

## Implementation Status

### ✅ Calibration Mode - IMPLEMENTED

**Files Created/Modified**:
- `src/pyspectrometer/modes/__init__.py` - Mode module exports
- `src/pyspectrometer/modes/base.py` - Base mode class with common functionality
- `src/pyspectrometer/modes/calibration.py` - Calibration mode implementation
- `src/pyspectrometer/data/__init__.py` - Data module exports
- `src/pyspectrometer/data/reference_spectra.py` - Reference spectra (FL, Hg, Sun, LED)
- `src/pyspectrometer/gui/control_bar.py` - Mode-specific button layouts
- `src/pyspectrometer/display/renderer.py` - Mode overlay rendering
- `src/pyspectrometer/core/calibration.py` - Added recalibrate(), cal_pixels, cal_wavelengths
- `src/pyspectrometer/spectrometer.py` - Calibration mode integration

**GUI Controls**:
| Button | Action | Description |
|--------|--------|-------------|
| FL | source_fl | Select Fluorescent reference |
| Hg | source_hg | Select Mercury reference |
| Sun | source_sun | Select Solar reference |
| LED | source_led | Select White LED reference |
| Overlay | toggle_overlay | Toggle reference overlay visibility |
| AutoLvl | auto_level | Toggle auto-level (gain adjustment) |
| AutoCal | auto_calibrate | Automatic peak matching calibration |
| SaveCal | save_cal | Save calibration to file |
| LoadCal | load_cal | Load calibration from file |
| Freeze | freeze | Freeze spectrum for calibration |
| Peak | capture_peak | Peak hold mode |
| Avg | toggle_averaging | Toggle frame averaging |
| AutoG | auto_gain | Toggle automatic gain control |
| Gain+/- | gain_up/down | Manual gain adjustment |
| Clear | clear_points | Clear calibration points |
| Quit | quit | Exit application |

**Calibration Workflow**:
1. `make calibrate` to start calibration mode
2. FL source is selected by default with overlay visible
3. Point spectrometer at matching light source
4. AutoGain is enabled by default - spectrum should auto-scale
5. Click `Freeze` to lock the current spectrum
6. Click `AutoCal` to match peaks and compute calibration
7. Verify overlay aligns with measured peaks
8. Click `SaveCal` to save, or clear and retry

**Technical Notes**:
- 4-point minimum for cubic polynomial fit
- Calibration saved as pixel→wavelength pairs (human-editable)
- Reference sources include characteristic spectral lines
- White LED has blue peak (~450nm) and broad phosphor (~550nm)

### ✅ Measurement Mode - IMPLEMENTED

**Files Created/Modified**:
- `src/pyspectrometer/modes/measurement.py` - Measurement mode implementation
- `src/pyspectrometer/modes/__init__.py` - Added MeasurementMode export
- `src/pyspectrometer/gui/control_bar.py` - Added MEASUREMENT_BUTTONS layout
- `src/pyspectrometer/spectrometer.py` - Measurement mode integration

**GUI Controls**:
| Button | Action | Description |
|--------|--------|-------------|
| Capture | capture | Capture current spectrum as reference |
| Peak | capture_peak | Peak hold mode |
| Avg | toggle_averaging | Toggle spectrum averaging |
| Dark | set_dark | Set dark/black reference |
| White | set_white | Set white reference |
| ClrRef | clear_refs | Clear all references |
| Save | save | Save spectrum to file |
| Load | load | Load spectrum from file |
| ShowRef | show_reference | Toggle reference overlay |
| Norm | normalize | Toggle normalization to reference |
| AutoG | auto_gain | Toggle automatic gain control |
| Gain+/- | gain_up/down | Manual gain adjustment |
| Lamp | lamp_toggle | Toggle GPIO 22 lamp |
| Quit | quit | Exit application |

**Measurement Workflow**:
1. `make measure` to start measurement mode
2. Optional: Click `Dark` with covered sensor to set dark reference
3. Optional: Click `White` with white reference to set baseline
4. Point spectrometer at sample
5. Click `Capture` to capture and set as reference
6. Enable `ShowRef` to overlay reference spectrum
7. Enable `Norm` to normalize measured spectrum to reference
8. Click `Save` to save current spectrum to CSV

**Features**:
- Dark subtraction: Automatically subtracts dark spectrum if set
- White normalization: Normalizes to white reference for reflectance/transmittance
- Reference overlay: Display captured reference as gray overlay
- Reference normalization: Compute ratio to reference spectrum
- Averaging: Accumulate multiple spectra for noise reduction
- Auto-gain: Automatic gain adjustment to keep spectrum in range

**Technical Notes**:
- Dark and white references are subtracted/normalized before display
- Captured spectrum is automatically set as reference for overlay/normalization
- Status bar shows "Dark", "White", "Ref" when references are set
- Status bar shows averaging count when enabled

### ⏳ Raman Mode - PENDING

### ⏳ Color Science Mode - PENDING

---

*Document Version: 1.2*
*Updated: 2026-03-08*
*Author: AI Assistant*
