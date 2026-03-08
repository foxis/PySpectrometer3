# PySpectrometer3 Refactoring Guide

This document explains how to refactor the PySpectrometer3 codebase for best practices: **SOLID principles**, **Single Responsibility**, **DRY**, and maintainability. It is a guide for incremental improvement, not a mandate to refactor everything at once.

---

## Table of Contents

1. [Current Architecture Overview](#current-architecture-overview)
2. [Single Responsibility Principle (SRP)](#single-responsibility-principle-srp)
3. [Open/Closed Principle (OCP)](#openclosed-principle-ocp)
4. [Liskov Substitution Principle (LSP)](#liskov-substitution-principle-lsp)
5. [Interface Segregation Principle (ISP)](#interface-segregation-principle-isp)
6. [Dependency Inversion Principle (DIP)](#dependency-inversion-principle-dip)
7. [DRY (Don't Repeat Yourself)](#dry-dont-repeat-yourself)
8. [Refactoring Priorities](#refactoring-priorities)

---

## Current Architecture Overview

```
pyspectrometer/
├── spectrometer.py      # Main orchestrator (~800 lines) – GOD CLASS
├── config.py            # Configuration (well-structured)
├── core/                # Spectrum, Calibration, ReferenceSpectrumManager
├── capture/             # Camera abstraction (base + picamera)
├── processing/          # Pipeline, filters, peak detection, extraction
├── display/             # Renderer (~930 lines), graticule, waterfall
├── gui/                 # Control bar, buttons, sliders
├── modes/               # Calibration, Measurement (BaseMode)
├── data/                # Reference spectra
├── export/              # CSV exporter
└── input/               # Keyboard handler
```

---

## Single Responsibility Principle (SRP)

> Each class/module should have one reason to change.

### 1. `Spectrometer` (spectrometer.py) – God Class

**Problem:** `Spectrometer` handles:

- Camera lifecycle
- Calibration loading/saving
- Extraction, pipeline, peak detection
- Display setup and rendering coordination
- Keyboard and button callback registration
- Mode-specific logic (calibration vs measurement)
- Auto-gain/exposure
- Freeze, peak hold, averaging
- Save/export
- Status updates

**Refactoring direction:**

| Responsibility | Extract To | Notes |
|----------------|------------|-------|
| Main loop orchestration | `Spectrometer` (slim) | Keep only: run loop, wire components |
| Event/callback registration | `EventHandlerRegistry` or `CallbackBinder` | Centralize `_register_*_callbacks` |
| Auto-gain/exposure logic | `AutoGainController` | Move `_handle_auto_gain_exposure` |
| Mode coordination | `ModeCoordinator` | Switch modes, delegate to mode handlers |
| Freeze/peak hold | Keep in modes or `SpectrumStateManager` | Already partly in modes |

**Example – extract AutoGainController:**

```python
# processing/auto_gain.py
class AutoGainController:
    """Adjusts gain/exposure to keep spectrum peak in target range."""
    
    def __init__(self, target_high: float = 0.95, target_low: float = 0.50):
        self.target_high = target_high
        self.target_low = target_low
    
    def adjust(
        self,
        data: SpectrumData,
        camera: CameraInterface,
        gain_callback: Callable[[float], None],
        exposure_callback: Callable[[int], None],
        auto_gain: bool,
        auto_exposure: bool,
    ) -> None:
        """Adjust gain/exposure if peak is outside target range."""
        # Move logic from _handle_auto_gain_exposure here
        ...
```

### 2. `DisplayManager` (display/renderer.py) – Too Many Responsibilities

**Problem:** `DisplayManager` handles:

- Window creation and lifecycle
- Mouse events (control bar, sliders, click points)
- Graticule, spectrum, peaks, cursor, overlays
- Preview modes (window, full, none)
- Status bar, control bar, sliders
- Autolevel overlay
- Waterfall coordination

**Refactoring direction:**

| Responsibility | Extract To | Notes |
|----------------|------------|-------|
| Graph rendering | `SpectrumGraphRenderer` | `_render_spectrum`, `_render_peaks`, `_render_cursor`, `_render_click_points` |
| Overlay rendering | `OverlayRenderer` | `_render_mode_overlay`, `_render_sensitivity_overlay`, `_render_reference_spectrum` – share common polyline logic |
| Preview composition | `PreviewComposer` | `render()` preview-mode switch, `_draw_rotated_crop_box` |
| Status display | `StatusDisplay` | `_draw_status_bar`, `_draw_status_overlay`, `_update_control_bar_status` |
| Mouse routing | Keep in DisplayManager or `InputRouter` | Route to control bar, sliders, graph |

**Example – shared overlay rendering:**

```python
# display/overlay_renderer.py
def render_polyline_overlay(
    graph: np.ndarray,
    intensity: np.ndarray,
    color: tuple[int, int, int],
    resample_to_width: Optional[int] = None,
) -> None:
    """Render intensity as polyline on graph. Reusable for mode, sensitivity, reference."""
    height = graph.shape[0]
    if resample_to_width and len(intensity) != resample_to_width:
        x_old = np.linspace(0, resample_to_width - 1, num=len(intensity), dtype=np.float32)
        x_new = np.arange(resample_to_width, dtype=np.float32)
        intensity = np.interp(x_new, x_old, np.asarray(intensity, dtype=np.float32))
    scale = float(height - 1) if height > 1 else 1.0
    points = [(i, height - int(min(float(v) * scale, height - 1))) for i, v in enumerate(intensity)]
    if len(points) > 1:
        pts = np.array(points, dtype=np.int32)
        cv2.polylines(graph, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)
```

### 3. `CalibrationMode` (modes/calibration.py)

**Problem:** Calibration mode handles:

- Reference source selection
- Overlay visibility
- Sensitivity correction (load CSV, apply, overlay)
- Auto-calibrate (correlation optimization)
- Peak-matching calibration
- Calibration point management

**Refactoring direction:**

| Responsibility | Extract To | Notes |
|----------------|------------|-------|
| Correlation-based calibration | `CorrelationCalibrator` | `_auto_calibrate_correlation` |
| Peak-matching calibration | `PeakMatchCalibrator` | `calibrate_from_peaks`, `_match_peaks_to_reference` |
| CMOS sensitivity | `SensitivityCorrector` | Load CSV, `apply_sensitivity_correction`, `get_sensitivity_curve` |
| Overlay generation | Keep in mode or `CalibrationOverlayProvider` | `get_overlay`, `get_sensitivity_curve` |

---

## Open/Closed Principle (OCP)

> Open for extension, closed for modification.

### 1. Button Layouts

**Problem:** `control_bar.py` uses hardcoded `CALIBRATION_BUTTONS` and `MEASUREMENT_BUTTONS`. Adding a new mode requires editing this file.

**Refactoring:** Modes already define `get_buttons()` returning `ButtonDefinition`. Use that as the single source of truth:

```python
# control_bar.py – use mode's get_buttons()
def _setup_buttons_for_mode(self, mode: str, mode_instance: Optional[BaseMode] = None) -> None:
    if mode_instance is not None:
        buttons = [self._button_def_to_control_def(b) for b in mode_instance.get_buttons()]
    else:
        buttons = DEFAULT_BUTTONS  # fallback
```

**Action:** Align `control_bar.py` button definitions with `modes/*.py` `get_buttons()` so there is one source of truth (prefer mode definitions).

### 2. Processing Pipeline

**Current:** Pipeline is already extensible – add processors without changing `ProcessingPipeline`. Good OCP.

### 3. Modes

**Current:** New modes extend `BaseMode` and implement abstract methods. Good OCP. Ensure Raman and Color Science modes follow the same pattern.

---

## Liskov Substitution Principle (LSP)

> Subtypes must be substitutable for their base types.

### 1. `BaseMode.process_spectrum` vs Implementations

**Problem:** `CalibrationMode.process_spectrum` returns intensity; `MeasurementMode.process_spectrum` returns intensity scaled 0–255. Inconsistent output range.

**Refactoring:** Define a clear contract:

- Either all modes return normalized 0–1, and display handles scaling.
- Or document that measurement mode returns 0–255 for legacy compatibility, and add a `normalized_output: bool` contract.

**Recommendation:** Standardize on 0–1 float for all modes; display/renderer does final scaling.

### 2. `BaseMode.apply_references` vs `MeasurementMode.process_spectrum`

**Problem:** `BaseMode` has `apply_references()` (dark/white correction) but `MeasurementMode` reimplements this in `process_spectrum` with slightly different logic (e.g. `* 255`).

**Refactoring:** Use `BaseMode.apply_references()` in measurement mode, or make it the single implementation:

```python
# In MeasurementMode.process_spectrum
result = self.accumulate_spectrum(intensity) if self.state.averaging_enabled else result
result = self.apply_references(result)  # Reuse base implementation
if self.meas_state.normalize_to_reference and self.meas_state.reference_spectrum is not None:
    result = self._normalize_to_reference(result)
return np.clip(result, 0, 1).astype(np.float32)  # Standard 0-1 output
```

---

## Interface Segregation Principle (ISP)

> Clients should not depend on methods they don't use.

### 1. DisplayManager Exposure

**Problem:** `Spectrometer` uses `self._display._control_bar.set_status()` – reaching into internals.

**Refactoring:** Add a narrow interface on `DisplayManager`:

```python
def set_status(self, key: str, value: str) -> None:
    """Set status for control bar. Preferred over accessing _control_bar."""
    self._control_bar.set_status(key, value)
```

Spectrometer should call `display.set_status()`, not `display._control_bar.set_status()`.

### 2. Camera Interface

**Current:** `CameraInterface` in `capture/base.py` is a good abstraction. Ensure all camera operations go through it.

### 3. Mode Interface

**Current:** `BaseMode` defines `process_spectrum`, `get_overlay`, `get_buttons`. Modes that don't need overlays can return `None`. Interface is already reasonably small.

---

## Dependency Inversion Principle (DIP)

> Depend on abstractions, not concretions.

### 1. Spectrometer Construction

**Problem:** `Spectrometer` directly constructs `PicameraCapture`, `DisplayManager`, `CSVExporter`, etc. Hard to test and swap implementations.

**Refactoring:** Accept dependencies via constructor (optional for backward compatibility):

```python
def __init__(
    self,
    config: Optional[Config] = None,
    camera: Optional[CameraInterface] = None,
    display_factory: Optional[Callable[[Config, Calibration, str], DisplayInterface]] = None,
    exporter: Optional[ExportInterface] = None,
    mode: str = "measurement",
    ...
):
    self._camera = camera or PicameraCapture(...)
    self._display = (display_factory or DisplayManager)(config, calibration, mode)
    self._exporter = exporter or CSVExporter(...)
```

### 2. Display Interface

**Refactoring:** Define `DisplayInterface` with the methods Spectrometer needs:

```python
# display/base.py
from abc import ABC, abstractmethod

class DisplayInterface(ABC):
    @abstractmethod
    def setup_windows(self) -> None: ...
    
    @abstractmethod
    def render(self, data: SpectrumData, **kwargs) -> None: ...
    
    def set_status(self, key: str, value: str) -> None: ...
    
    def register_button_callback(self, action_name: str, callback: Callable[[], None]) -> bool: ...
    
    def set_button_active(self, action_name: str, active: bool) -> bool: ...
```

`DisplayManager` implements `DisplayInterface`.

---

## DRY (Don't Repeat Yourself)

### 1. Overlay Rendering

**Problem:** `_render_mode_overlay`, `_render_sensitivity_overlay`, and `_render_reference_spectrum` in `renderer.py` repeat the same pattern: intensity → points → polylines/lines.

**Refactoring:** Use a shared helper (see `render_polyline_overlay` example under DisplayManager).

### 2. Dark/White Reference Logic

**Problem:** `BaseMode.apply_references` and `MeasurementMode.process_spectrum` both implement dark subtraction and white normalization.

**Refactoring:** Use `BaseMode.apply_references` everywhere, or extract to a `ReferenceCorrector`:

```python
# processing/reference_correction.py
def apply_dark_white_correction(
    intensity: np.ndarray,
    dark: Optional[np.ndarray],
    white: Optional[np.ndarray],
) -> np.ndarray:
    result = intensity.astype(np.float64)
    if dark is not None:
        result = np.maximum(result - dark, 0)
    if white is not None:
        w = np.maximum(white - (dark or 0), 1)
        result = (result / w)
    return np.clip(result, 0, 1).astype(np.float32)
```

### 3. Accumulate Spectrum

**Problem:** `CalibrationMode.process_spectrum` calls `self.accumulate_frame(intensity)` but `BaseMode` only defines `accumulate_spectrum` – `accumulate_frame` does not exist. This is a bug (would raise `AttributeError` if `process_spectrum` were ever called for calibration mode). The main loop calls `accumulate_spectrum` directly for calibration mode, so `CalibrationMode.process_spectrum` is currently unused.

**Refactoring:** In `CalibrationMode.process_spectrum`, change `accumulate_frame` → `accumulate_spectrum`. Use `BaseMode.accumulate_spectrum` consistently.

### 4. Button Definitions Duplication

**Problem:** `modes/calibration.py` `get_buttons()` and `control_bar.py` `CALIBRATION_BUTTONS` define similar buttons but are not identical. Two sources of truth.

**Refactoring:** Use mode's `get_buttons()` as the single source. Map `ButtonDefinition` to `ButtonDef` in the control bar. Remove hardcoded `CALIBRATION_BUTTONS` / `MEASUREMENT_BUTTONS` in favor of mode-driven setup.

### 5. Intensity Scaling to Graph Height

**Problem:** Multiple places scale intensity (0–1) to graph Y: `scaled = intensity * (graph_height - 10)` or similar.

**Refactoring:** Centralize in a helper:

```python
# display/scale_utils.py
def scale_intensity_to_graph(intensity: np.ndarray, graph_height: int, margin: int = 10) -> np.ndarray:
    return intensity * (graph_height - margin)
```

---

## Refactoring Priorities

### High impact, lower risk

1. **DRY – overlay rendering:** Extract `render_polyline_overlay` and reuse. Low risk, clear win.
2. **ISP – DisplayManager encapsulation:** Add `set_status()` and avoid `_control_bar` access from Spectrometer.
3. **DRY – reference correction:** Unify dark/white logic in `BaseMode.apply_references` or `ReferenceCorrector`.

### Medium impact, medium effort

4. **SRP – AutoGainController:** Extract auto-gain logic from Spectrometer.
5. **DIP – constructor injection:** Allow injecting camera, display, exporter for tests.
6. **Button source of truth:** Use mode `get_buttons()` in control bar.

### Higher effort

7. **SRP – Spectrometer split:** Extract `EventHandlerRegistry`, `ModeCoordinator`.
8. **SRP – DisplayManager split:** Extract `SpectrumGraphRenderer`, `OverlayRenderer`, `PreviewComposer`.
9. **SRP – CalibrationMode split:** Extract `CorrelationCalibrator`, `SensitivityCorrector`.

---

## Summary Checklist

| Principle | Area | Action |
|-----------|------|--------|
| SRP | Spectrometer | Extract AutoGainController, consider EventHandlerRegistry |
| SRP | DisplayManager | Extract overlay/graph rendering helpers |
| SRP | CalibrationMode | Extract SensitivityCorrector, CorrelationCalibrator |
| OCP | Buttons | Use mode get_buttons() as single source |
| LSP | process_spectrum | Standardize 0–1 output; reuse apply_references |
| ISP | DisplayManager | Add set_status(), hide _control_bar |
| DIP | Spectrometer | Optional constructor injection for camera, display, exporter |
| DRY | Overlays | Shared render_polyline_overlay |
| DRY | Dark/white | Single apply_references or ReferenceCorrector |
| DRY | Buttons | One source: mode get_buttons() |

---

*Document Version: 1.0*  
*Created: 2026-03-08*
