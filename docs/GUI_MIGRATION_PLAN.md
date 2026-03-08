# GUI Migration Plan: Keyboard Controls to GUI-Based Controls

## Overview

This document outlines the phased migration of PySpectrometer3 from keyboard-focused controls to a comprehensive GUI-based control system using OpenCV's `cv2.createTrackbar`, `cv2.createButton` (Qt backend), and mouse-based interactions.

### Goals
1. Replace all keyboard shortcuts with GUI controls
2. Add new features: spectrum capture modes, reference/dark spectra, XYZ fitting, GPIO
3. Maintain backwards compatibility with keyboard shortcuts during transition
4. Keep OpenCV as the UI framework (no additional dependencies like Tk, Qt standalone)

### Current State Summary

| Feature | Current Implementation |
|---------|----------------------|
| UI Framework | OpenCV (`cv2.namedWindow`, `cv2.imshow`) |
| Input | `cv2.waitKey()` → 20 keyboard actions |
| Calibration | Click + keyboard workflow |
| Gain | Keyboard up/down (0-50 range) |
| Save | CSV export only |
| Reference | Not implemented (spec exists) |
| Dark spectrum | Not implemented |
| XYZ fitting | Not implemented |
| GPIO | Not implemented |

---

## Architecture Design

### Control Panel Approach

OpenCV supports two approaches for GUI controls:
1. **Trackbars** (`cv2.createTrackbar`) - Built-in, works everywhere
2. **Buttons** (`cv2.createButton`) - Requires Qt backend (`-DWITH_QT=ON`)

**Recommendation**: Use a **hybrid approach**:
- Control Panel window with trackbars for numeric controls
- Mouse click regions for buttons/actions (custom rendering)
- Keep keyboard shortcuts as secondary input

### Module Structure

```
src/pyspectrometer/
├── gui/                          # NEW module
│   ├── __init__.py
│   ├── control_panel.py          # Main control panel window
│   ├── buttons.py                # Clickable button rendering
│   ├── trackbars.py              # Trackbar wrappers
│   ├── dialogs.py                # File dialogs (save/load)
│   └── state.py                  # GUI state management
├── features/                     # NEW module
│   ├── __init__.py
│   ├── capture_modes.py          # Current/Peak/Average/Sum capture
│   ├── dark_spectrum.py          # Dark spectrum subtraction
│   ├── xyz_fitting.py            # XYZ curve fitting
│   └── gpio_control.py           # GPIO light control
├── reference/                    # As per REFERENCE_OVERLAY_SPEC.md
│   ├── spectrum.py
│   ├── loader.py
│   ├── interpolator.py
│   └── manager.py
└── ...existing modules...
```

### Control Panel Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│  PySpectrometer 3 - Control Panel                                   │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  CALIBRATION                                                 │   │
│  │  [Start Calibration] [Clear Points] [Save Cal]               │   │
│  │  Mode: ○ Pixel  ○ Measure  ○ Off                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  CAPTURE                                                     │   │
│  │  [Capture Current] [Capture Peak] [Capture Avg] [Capture Sum]│   │
│  │  [Save Spectrum] [Load Reference]                            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  SPECTRUM PROCESSING                                         │   │
│  │  Reference: [None ▼] [Use as Ref] [Clear Ref]                │   │
│  │  Dark:      [None]   [Use as Dark] [Clear Dark]              │   │
│  │  [Fit XYZ Curves]                                            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  CAMERA                                                      │   │
│  │  Gain: ═══════════●═══════════ 25  [Auto]                    │   │
│  │  Extraction: [MED] [WGT] [GAU]                               │   │
│  │  Angle:  ═══════●═══════════ 0.0°  [Auto Detect]             │   │
│  │  Width:  ═══════●═══════════ 20                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  SIGNAL PROCESSING                                           │   │
│  │  SavGol:    ═══════●═══════════ 3                            │   │
│  │  Peak Width:═══════●═══════════ 10                           │   │
│  │  Threshold: ═══════●═══════════ 20                           │   │
│  │  Hold Peaks: [ ]                                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  LIGHT CONTROL (GPIO 22)                                     │   │
│  │  [ON] [OFF]  PWM: ═══════●═══════════ 50%                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  [Quit]                                                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Feature Requirements

### 1. Calibration (GUI-ified)
- **Start Calibration**: Enter pixel mode, show instructions overlay
- **Clear Points**: Clear clicked calibration points
- **Save Calibration**: Save current calibration to file
- **Mode Toggle**: Radio buttons for Pixel/Measure/Off modes

### 2. Spectrum Capture Modes

| Mode | Description | Implementation |
|------|-------------|----------------|
| **Current** | Single snapshot | Copy current intensity array |
| **Peak** | Peak hold over time | `np.maximum(held, current)` |
| **Average** | Running average | Accumulate and divide |
| **Sum** | Accumulate intensity | Sum frames until stopped |

**Data Flow**:
```
Capture Button → CaptureManager.capture(mode) → CapturedSpectrum
                                              ↓
                                    Store in capture buffer
                                              ↓
                          Available for: Save, Use as Ref, Use as Dark
```

### 3. Reference Spectrum System

**Extends REFERENCE_OVERLAY_SPEC.md with:**

| Feature | Description |
|---------|-------------|
| Load from file | File dialog to select CSV/PRN |
| Load from data/ | Dropdown of available references |
| Use captured as reference | Set current capture as reference |
| Reference subtraction | `display = current - reference` |
| Reference ratio | `display = current / reference` |

**Reference Processing Mode**:
```python
class ReferenceMode(Enum):
    OVERLAY = "overlay"      # Display reference as overlay line
    SUBTRACT = "subtract"    # Subtract reference from live
    RATIO = "ratio"          # Divide live by reference
    NONE = "none"           # No reference processing
```

### 4. Dark Spectrum Subtraction

**Purpose**: Remove sensor noise and ambient light.

```python
@dataclass
class DarkSpectrum:
    intensity: np.ndarray
    capture_time: datetime
    frames_averaged: int = 1
    
    @classmethod
    def capture_average(cls, frames: list[np.ndarray]) -> "DarkSpectrum":
        """Capture dark by averaging multiple frames."""
        return cls(
            intensity=np.mean(frames, axis=0),
            capture_time=datetime.now(),
            frames_averaged=len(frames),
        )
```

**Processing Pipeline**:
```
Raw Intensity → Dark Subtraction → Reference Processing → Filter → Display
                     ↓
            max(0, raw - dark)  # Clamp to avoid negatives
```

### 5. Gain Control with Auto Mode

**Current**: Manual up/down (0-50)

**Enhanced**:
- Trackbar for manual control
- Auto-gain checkbox: Adjusts gain to keep peak intensity in target range
- Target range config (e.g., 200-240 out of 255)

```python
class AutoGainController:
    def __init__(
        self,
        target_min: int = 200,
        target_max: int = 240,
        gain_step: float = 0.5,
    ):
        self.enabled = False
        self.target_min = target_min
        self.target_max = target_max
        self.gain_step = gain_step
    
    def adjust(self, current_max: int, current_gain: float) -> float:
        """Calculate gain adjustment based on current max intensity."""
        if not self.enabled:
            return current_gain
        
        if current_max > self.target_max:
            return max(0, current_gain - self.gain_step)
        elif current_max < self.target_min:
            return min(50, current_gain + self.gain_step)
        return current_gain
```

### 6. Save/Load Spectrum

**Save**:
- Current capture → CSV with wavelength, intensity, metadata
- Optional PNG of graph
- Optional JSON metadata (calibration, settings)

**Load**:
- File dialog to select saved spectrum
- Load as reference spectrum
- Display as overlay

**File Format Enhancement**:
```csv
# PySpectrometer3 Spectrum Export
# Timestamp: 2026-03-08T14:30:00
# Calibration: caldata.txt
# Gain: 25
# Mode: Peak Hold
Wavelength (nm),Intensity
380.0,12.5
380.5,13.2
...
```

### 7. XYZ Curve Fitting

**Purpose**: Fit measured spectrum to CIE XYZ color matching functions.

**Algorithm**:
1. Load CIE 1931 2° observer data (x̄, ȳ, z̄ curves)
2. Interpolate to match spectrometer wavelengths
3. Calculate tristimulus values:
   ```
   X = Σ S(λ) × x̄(λ) × Δλ
   Y = Σ S(λ) × ȳ(λ) × Δλ
   Z = Σ S(λ) × z̄(λ) × Δλ
   ```
4. Convert to chromaticity coordinates: `x = X/(X+Y+Z)`, `y = Y/(X+Y+Z)`
5. Display on chromaticity diagram overlay

**Data Files Required**:
- `data/CIE_xyz_1931_2deg.csv` (already in EasyRobot/colorscience)

### 8. GPIO Light Control (Pin 22)

**Requirements**:
- On/Off toggle
- Software PWM for intensity control
- Works on Raspberry Pi only (graceful fallback on other platforms)

```python
class GPIOLightControl:
    """Control external light via GPIO pin 22."""
    
    def __init__(self, pin: int = 22, pwm_freq: int = 100):
        self._pin = pin
        self._pwm_freq = pwm_freq
        self._pwm = None
        self._enabled = False
        self._duty_cycle = 100  # 0-100%
        self._gpio_available = self._init_gpio()
    
    def _init_gpio(self) -> bool:
        """Initialize GPIO, return False if not available."""
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self._pin, GPIO.OUT)
            self._pwm = GPIO.PWM(self._pin, self._pwm_freq)
            return True
        except (ImportError, RuntimeError):
            return False
    
    def set_enabled(self, enabled: bool) -> None:
        """Turn light on/off."""
        if not self._gpio_available:
            return
        
        self._enabled = enabled
        if enabled:
            self._pwm.start(self._duty_cycle)
        else:
            self._pwm.stop()
    
    def set_duty_cycle(self, duty: int) -> None:
        """Set PWM duty cycle (0-100%)."""
        self._duty_cycle = max(0, min(100, duty))
        if self._enabled and self._gpio_available:
            self._pwm.ChangeDutyCycle(self._duty_cycle)
    
    def cleanup(self) -> None:
        """Clean up GPIO on exit."""
        if self._gpio_available:
            import RPi.GPIO as GPIO
            GPIO.cleanup(self._pin)
```

---

## Implementation Phases

### Phase 1: GUI Infrastructure (Foundation)
**Duration**: ~3-4 hours
**Priority**: Critical

1. Create `gui/` module structure
2. Implement `ControlPanel` class with OpenCV window
3. Implement custom button rendering with click detection
4. Implement trackbar wrappers
5. Wire control panel to `Spectrometer` main loop
6. Maintain keyboard as fallback

**Deliverables**:
- [ ] `gui/__init__.py`
- [ ] `gui/control_panel.py` - Main control panel window
- [ ] `gui/buttons.py` - Clickable button system
- [ ] `gui/trackbars.py` - Trackbar utilities
- [ ] `gui/state.py` - GUI state management
- [ ] Updated `spectrometer.py` with control panel integration

### Phase 2: Capture Modes
**Duration**: ~2 hours
**Priority**: High

1. Create `CaptureManager` class
2. Implement Current/Peak/Average/Sum capture modes
3. Add capture buttons to control panel
4. Create `CapturedSpectrum` data structure
5. Display captured spectrum indicator

**Deliverables**:
- [ ] `features/capture_modes.py`
- [ ] Capture buttons in control panel
- [ ] Visual indicator of active capture

### Phase 3: Gain Control Enhancement
**Duration**: ~1 hour
**Priority**: High

1. Add gain trackbar to control panel
2. Implement `AutoGainController`
3. Add auto-gain toggle checkbox
4. Wire to camera gain

**Deliverables**:
- [ ] Gain trackbar
- [ ] `features/auto_gain.py`
- [ ] Auto-gain toggle

### Phase 4: Save/Load Spectrum
**Duration**: ~2 hours
**Priority**: High

1. Enhance CSV export with metadata header
2. Implement file save dialog (or filename input)
3. Implement file load for reference
4. Add save/load buttons

**Deliverables**:
- [ ] `gui/dialogs.py` - File dialogs
- [ ] Enhanced `export/csv_exporter.py`
- [ ] Save/Load buttons

### Phase 5: Reference Spectrum System
**Duration**: ~4 hours
**Priority**: Medium-High

1. Implement `reference/` module per REFERENCE_OVERLAY_SPEC.md
2. Add reference dropdown/selector to control panel
3. Add "Use as Reference" button
4. Implement reference overlay rendering
5. Add reference processing modes (subtract, ratio)

**Deliverables**:
- [ ] `reference/spectrum.py`
- [ ] `reference/loader.py`
- [ ] `reference/interpolator.py`
- [ ] `reference/manager.py`
- [ ] Reference controls in panel
- [ ] Overlay rendering in `DisplayManager`

### Phase 6: Dark Spectrum Subtraction
**Duration**: ~2 hours
**Priority**: Medium

1. Create `DarkSpectrum` class
2. Implement dark capture (single + averaged)
3. Add dark buttons to control panel
4. Integrate dark subtraction in pipeline
5. Add dark status indicator

**Deliverables**:
- [ ] `features/dark_spectrum.py`
- [ ] Dark controls in panel
- [ ] Pipeline integration

### Phase 7: XYZ Curve Fitting
**Duration**: ~3 hours
**Priority**: Medium

1. Load CIE observer data
2. Implement tristimulus calculation
3. Implement chromaticity coordinates
4. Add XYZ fit button
5. Display results (XYZ values, chromaticity on diagram)

**Deliverables**:
- [ ] `features/xyz_fitting.py`
- [ ] `data/CIE_xyz_1931_2deg.csv`
- [ ] Fit button and results display

### Phase 8: GPIO Light Control
**Duration**: ~1 hour
**Priority**: Low

1. Implement `GPIOLightControl` class
2. Add on/off buttons
3. Add PWM trackbar
4. Graceful fallback for non-Pi platforms

**Deliverables**:
- [ ] `features/gpio_control.py`
- [ ] Light controls in panel

### Phase 9: Calibration GUI Enhancement
**Duration**: ~2 hours
**Priority**: Medium

1. Replace keyboard calibration with GUI workflow
2. Add calibration wizard overlay
3. Wavelength input via GUI (not console)
4. Mode toggles as radio buttons

**Deliverables**:
- [ ] `gui/calibration_wizard.py`
- [ ] Calibration controls in panel

### Phase 10: Polish & Integration
**Duration**: ~2 hours
**Priority**: Medium

1. Remove console dependencies (wavelength input)
2. Clean up keyboard handlers (make optional)
3. Add tooltips/help text
4. Save/restore control panel state
5. Documentation update

**Deliverables**:
- [ ] Full GUI-only operation capability
- [ ] Updated README
- [ ] User guide

---

## Implementation Priority Order

```
Phase 1 (GUI Infrastructure) ─────────────────────────────────────────►
            │
            ├──► Phase 2 (Capture Modes) ─────────────────────────────►
            │
            ├──► Phase 3 (Gain Control) ──────────────────────────────►
            │
            └──► Phase 4 (Save/Load) ─────────────────────────────────►
                        │
                        └──► Phase 5 (Reference System) ──────────────►
                                    │
                                    └──► Phase 6 (Dark Spectrum) ─────►
                                                │
                                                └──► Phase 7 (XYZ Fit) ►
                                                │
                                                └──► Phase 8 (GPIO) ──►
                                                            │
Phase 9 (Calibration GUI) can run in parallel ──────────────────────────►
                                                            │
                                                Phase 10 (Polish) ─────►
```

---

## Technical Considerations

### OpenCV GUI Limitations
- No native buttons in standard OpenCV (requires Qt backend)
- Trackbars only support integer values
- No native dropdown menus
- No file dialogs

**Workarounds**:
1. **Custom buttons**: Render rectangles with text, detect clicks in regions
2. **Float trackbars**: Use integer trackbar, divide by scale factor
3. **Dropdowns**: Cycle through options on click, or use separate window
4. **File dialogs**: Use `tkinter.filedialog` (minimal Tk dependency) or path input

### Thread Safety
- OpenCV GUI must run in main thread
- Camera capture in main thread (Picamera2 requirement)
- Control panel callbacks should be quick (no blocking)

### State Management
```python
@dataclass
class GUIState:
    """Central GUI state management."""
    
    # Capture
    capture_mode: CaptureMode = CaptureMode.NONE
    captured_spectrum: Optional[CapturedSpectrum] = None
    
    # Reference
    reference: Optional[ReferenceSpectrum] = None
    reference_mode: ReferenceMode = ReferenceMode.OVERLAY
    
    # Dark
    dark_spectrum: Optional[DarkSpectrum] = None
    dark_enabled: bool = False
    
    # Camera
    auto_gain_enabled: bool = False
    
    # GPIO
    light_enabled: bool = False
    light_pwm: int = 100
```

---

## Testing Strategy

### Unit Tests
- `test_capture_modes.py` - Capture mode logic
- `test_auto_gain.py` - Auto-gain controller
- `test_dark_spectrum.py` - Dark subtraction math
- `test_xyz_fitting.py` - Tristimulus calculations
- `test_gpio_control.py` - GPIO mock tests

### Integration Tests
- Control panel renders correctly
- Button clicks trigger correct actions
- Trackbar changes update state
- End-to-end capture → save → load as reference

### Manual Testing Checklist
- [ ] All buttons clickable and responsive
- [ ] Trackbars update values in real-time
- [ ] Spectrum display updates correctly
- [ ] File save/load works
- [ ] Reference overlay displays
- [ ] Dark subtraction visible
- [ ] XYZ values reasonable for known light sources
- [ ] GPIO works on Raspberry Pi (if available)

---

## Migration Checklist

### Pre-Migration
- [ ] Backup current working state
- [ ] Create feature branch
- [ ] Set up test fixtures

### Per-Phase
- [ ] Implement feature
- [ ] Write tests
- [ ] Test manually
- [ ] Update documentation
- [ ] Commit with clear message

### Post-Migration
- [ ] Full integration test
- [ ] Performance check (frame rate)
- [ ] Update README with new controls
- [ ] Create user guide
- [ ] Remove deprecated keyboard-only code

---

## Open Questions

1. **Tk dependency for file dialogs?**
   - Option A: Minimal Tk (`tkinter.filedialog` only)
   - Option B: Type filename in OpenCV input
   - Option C: Use command-line argument for output dir
   - **Recommendation**: Option A (Tk already available in most Python installs)

2. **Control panel as separate window or embedded?**
   - Separate window allows flexible positioning
   - Embedded keeps everything in one place
   - **Recommendation**: Separate window (easier to implement, standard pattern)

3. **Keyboard shortcuts during transition?**
   - Keep all keyboard shortcuts working
   - Display key hints on buttons (e.g., "[S] Save")
   - **Recommendation**: Keep as secondary input method

4. **Auto-save captured spectrum?**
   - Auto-save all captures to temp directory
   - Manual save only
   - **Recommendation**: Manual save, but keep last N captures in memory

---

*Document Version: 1.0*
*Created: 2026-03-08*
*Author: AI Assistant*
