# PySpectrometer3 Refactoring Guide — Task List

This document is a **task list** for refactoring the codebase to improve **expandability**, **maintainability**, and **reliability** by centralizing important controllers/logic and enforcing **Single Responsibility Principle (SRP)**. Use it as a step-by-step guide; complete tasks in order where dependencies are noted.

**References:** [ARCHITECTURE.md](ARCHITECTURE.md) is the **single source of truth** for modes, button mappings, spectrum extraction, CLI, module map, and implementation status. MODES_SPEC.md and SPEC_SPECTRUM_EXTRACTION.md may be archived once ARCHITECTURE is adopted.

---

## Goals

| Goal | How the task list addresses it |
|------|--------------------------------|
| **Expandability** | Single source of truth for buttons/modes; interfaces and extension points instead of editing god classes. |
| **Maintainability** | Smaller, focused modules; one reason to change per class; clear boundaries. |
| **Fewer mistakes** | Centralized logic (reference correction, overlay rendering, event wiring) so fixes and behavior live in one place. |
| **SRP** | Extract responsibilities from `Spectrometer` and `DisplayManager` into dedicated components. |

---

## Current Hotspots (from ARCHITECTURE.md + code)

- **`spectrometer.py`** — Orchestrator; also owns callback registration, auto-gain wiring, mode branching, freeze/peak hold, status updates. *Target: slim orchestrator + central controllers (Controller in a Controller/Viewer/Data split).*
- **`display/renderer.py`** — Window lifecycle, graph, overlays, status, control bar, sliders, mouse routing. *Target: delegate rendering and status to focused components (Viewer).*
- **`gui/control_bar.py`** — Button layout from **hardcoded** `CALIBRATION_BUTTONS` / `MEASUREMENT_BUTTONS`; modes define `get_buttons()` but control bar does **not** use it. *Target: one source of truth (mode `get_buttons()`), aligned with ARCHITECTURE §3 button tables.*
- **Modes** — Five modes per ARCHITECTURE: Calibration, Measurement, **Waterfall**, Raman, Color Science. `process_spectrum` output range inconsistent (0–1 vs 0–255); dark/white logic centralized in `processing/reference_correction.py` — keep single path. *Target: standardize mode output contract; no duplicate reference logic; new modes (Waterfall, Raman, Color Science) additive via BaseMode.*

---

## Phase 1: Single Source of Truth & DRY

*Reduces mistakes by centralizing definitions and shared logic.*

### 1.1 Button layout — mode as single source

- [ ] **1.1.1** In `gui/control_bar.py`, change `_setup_buttons_for_mode(mode, ...)` to accept an optional `mode_instance: Optional[BaseMode] = None`. When `mode_instance` is provided, call `mode_instance.get_buttons()` and map each `ButtonDefinition` to `ButtonDef` (label, action_name, is_toggle, shortcut, row; add `icon_type` from mode if needed). When `mode_instance` is None, keep current fallback to `CALIBRATION_BUTTONS` / `MEASUREMENT_BUTTONS` for backward compatibility.
- [ ] **1.1.2** Where `ControlBar` is constructed (e.g. `DisplayManager`), pass the current mode instance so the control bar can call `get_buttons()`. Update `DisplayManager.__init__` and any `set_mode`/mode-switch path to pass the mode instance into the control bar.
- [ ] **1.1.3** Align each mode’s `get_buttons()` with (a) the actions implemented in Spectrometer (e.g. `_register_calibration_callbacks` / `_register_measurement_callbacks`) and (b) the button → action mapping in **ARCHITECTURE.md §3** (e.g. Calibration: FL12, HG, LED, D65, SaveCal, LoadCal, AutoCal, Overlay, Freeze, etc.; Measurement: Capture, Dark, White, ClrRef, Save, Load, ShowRef, Norm, LED, I2C…). Remove or reconcile duplicate definitions so control bar and modes stay in sync with ARCHITECTURE.
- [ ] **1.1.4** After verification, deprecate or remove hardcoded `CALIBRATION_BUTTONS` and `MEASUREMENT_BUTTONS` in favor of mode-driven layout (or keep as fallback only when no mode instance is available).

### 1.2 Reference correction — already centralized

- [x] **1.2.1** Dark/white correction lives in `processing/reference_correction.py` (`apply_dark_white_correction`). `BaseMode.apply_references` and `MeasurementMode` use it. **No duplicate logic.** Ensure any new mode that needs dark/white uses `apply_references` or `apply_dark_white_correction` only — no new in-mode implementations.
- [ ] **1.2.2** If any remaining code path still does manual dark subtract / white divide outside this module, refactor it to call `apply_dark_white_correction` (or `BaseMode.apply_references`).

### 1.3 Overlay rendering — already centralized

- [x] **1.3.1** `display/overlay_utils.py` provides `render_polyline_overlay`. Use it everywhere a polyline overlay is drawn (mode overlay, sensitivity overlay, reference spectrum). Audit `renderer.py` and ensure no duplicate “intensity → points → polylines” logic remains.
- [ ] **1.3.2** If any overlay path still implements its own polyline drawing, replace it with a call to `render_polyline_overlay`.

### 1.4 Intensity scaling to graph

- [x] **1.4.1** Add `scale_intensity_to_graph(intensity, graph_height, margin=10)` in `display/overlay_utils.py`; use it in calibration (overlay, sensitivity), measurement (overlay), and renderer (reference spectrum, spectrum line, peaks).

---

## Phase 2: Encapsulation & Narrow Interfaces

*Reduces mistakes by hiding internals and depending on stable APIs.*

### 2.1 Display — hide control bar and slider panel internals

- [x] **2.1.1** `DisplayManager` already exposes `set_status(key, value)`. Spectrometer should use only `display.set_status()`, not `display._control_bar`.
- [x] **2.1.2** register_slider_callbacks(gain_cb, exposure_cb, led_intensity_cb); LED slider added. — design the API to support gain, exposure, and LED intensity (or a generic “register slider callback by name”.
- [x] **2.1.3** Public `slider_panel` property removed; external code uses `register_slider_callbacks`, `toggle_*_slider`, `set_*_value`, `show_*_slider`.

### 2.2 Display interface (optional but recommended)

- [ ] **2.2.1** Define `DisplayInterface` (e.g. in `display/base.py`) with the methods Spectrometer needs: `setup_windows`, `render`, `set_status`, `register_button_callback`, `set_button_active`, `set_gain_value`, `set_exposure_value`, slider registration (gain, exposure, and LED intensity for Measurement/Color Science), and any other required. Make `DisplayManager` implement this interface. This prepares for dependency injection and testing.

---

## Phase 3: Centralize Controllers (Spectrometer slim-down)

*Reduces mistakes by having one place for event wiring and mode coordination.*

### 3.1 Event / callback registration

- [ ] **3.1.1** Introduce an `EventHandlerRegistry` or `CallbackBinder` (e.g. in `input/` or a small `core/` module). Responsibility: register keyboard actions, register button callbacks, and optionally slider callbacks from a single configuration (e.g. mode + spectrometer refs). Spectrometer passes this registry the callbacks it provides; the registry wires them to `KeyboardHandler` and `DisplayManager` (and sliders via display API). Spectrometer no longer contains long `_register_key_callbacks` / `_register_button_callbacks` / `_register_calibration_callbacks` / `_register_measurement_callbacks` blocks — instead it builds a list of (action, callback) and the registry applies them.
- [ ] **3.1.2** Move registration logic from Spectrometer into this component. Spectrometer only: creates the registry, supplies callbacks, and calls something like `registry.bind_for_mode(mode, self)` so that adding a new mode or new action does not require editing Spectrometer’s private methods.

### 3.2 Mode coordination

- [ ] **3.2.1** Introduce a `ModeCoordinator` (or use the same module as the registry). Responsibility: hold current mode, switch mode, and delegate “current mode” queries (e.g. which mode object, which status fields to show). Spectrometer asks the coordinator for the current mode and delegates mode-specific behavior (e.g. overlay, status, process_spectrum) to it, instead of branching on `self.mode == "calibration"` / `"measurement"` / `"waterfall"` / etc. Supports all five modes (Calibration, Measurement, Waterfall, Raman, Color Science) per ARCHITECTURE.
- [ ] **3.2.2** Spectrometer’s main loop and high-level flow use only the coordinator and the current mode interface; no duplicated if/else on mode name in multiple places.

### 3.3 Auto-gain / auto-exposure

- [x] **3.3.1** Auto-gain and auto-exposure are already in `processing/auto_controls.py` (`AutoGainController`, `AutoExposureController`). Spectrometer uses them. Keep this; ensure all auto-adjustment logic stays in these controllers and Spectrometer only invokes them (no in-orchestrator gain math).

---

## Phase 4: DisplayManager — Split Responsibilities (SRP)

*One reason to change per component.*

### 4.1 Graph and overlay rendering

- [ ] **4.1.1** Extract `SpectrumGraphRenderer`: responsibilities = draw spectrum line, peaks, cursor, click points on the graph. `DisplayManager.render()` delegates graph drawing to this class (pass graph buffer, spectrum data, display state). Move `_render_spectrum`, `_render_peaks`, `_render_cursor`, `_render_click_points` (or equivalent) into it.
- [ ] **4.1.2** Ensure overlay drawing (mode overlay, sensitivity, reference) uses `overlay_utils.render_polyline_overlay` and, if useful, an `OverlayRenderer` that only composes overlays (no window or control bar logic).

### 4.2 Preview and status

- [ ] **4.2.1** Extract `PreviewComposer`: responsibility = compose the preview image (window/full/none, rotated crop box, etc.). `DisplayManager` calls it with frame and config and gets back the preview image to show.
- [ ] **4.2.2** Extract or isolate `StatusDisplay`: responsibility = produce status bar text and any status overlay. `DisplayManager` gets status from control bar and/or this component and draws it in one place.

### 4.3 Mouse routing

- [ ] **4.3.1** Keep mouse routing in `DisplayManager` or move to a thin `InputRouter` that routes (x, y) to control bar, slider panel, or graph. Single place for “where did the click go?” so behavior is consistent and easy to extend.

---

## Phase 5: Mode Layer — Contracts & Calibration Split

*Consistent behavior and SRP for modes.*

### 5.1 Mode output contract (LSP)

- [ ] **5.1.1** Define and document the contract: all modes’ `process_spectrum` return intensity in **normalized 0–1** float. Display/renderer does final scaling to 0–255 or graph height. Audit `CalibrationMode.process_spectrum` and `MeasurementMode.process_spectrum`; if either returns 0–255, change it to 0–1 and move scaling to the display layer.
- [ ] **5.1.2** Ensure `MeasurementMode` uses `BaseMode.apply_references` (or `apply_dark_white_correction`) for dark/white and does not reimplement it. Remove any duplicate `* 255` or scaling that belongs in the display.

### 5.2 CalibrationMode — extract calibrators (SRP)

- [ ] **5.2.1** Extract `CorrelationCalibrator`: move correlation-based auto-calibrate logic (e.g. `_auto_calibrate_correlation`) from `CalibrationMode` into this class. It takes spectrum/reference data and returns updated calibration or fit result. CalibrationMode calls it instead of containing the algorithm.
- [ ] **5.2.2** Extract `PeakMatchCalibrator` (or keep in `processing/peak_detection.py`): peak-matching calibration (`calibrate_from_peaks`, `match_peaks_to_reference`) in one place. CalibrationMode delegates to it. Respect ARCHITECTURE rules: **4 calibration points minimum** (cubic fit); matching algorithm as in ARCHITECTURE §3.2.
- [ ] **5.2.3** Extract `SensitivityCorrector` (or a dedicated helper): load CMOS sensitivity CSV (or derive from reference spectrum per ARCHITECTURE §1.3), apply sensitivity correction, and provide the curve for overlay. CalibrationMode uses it for “sensitivity” feature; no CSV loading or curve logic inside the mode class.

### 5.3 Accumulate spectrum naming

- [ ] **5.3.1** In `CalibrationMode.process_spectrum` (if it ever uses accumulation), use `accumulate_spectrum` consistently. The refactoring guide previously noted a possible `accumulate_frame` typo; ensure only `BaseMode.accumulate_spectrum` is used and that both modes behave consistently for averaging.

---

## Phase 6: Dependency Injection (Testability & Swap)

*Depend on abstractions; optional for backward compatibility.*

### 6.1 Spectrometer constructor

- [ ] **6.1.1** Extend `Spectrometer.__init__` to accept optional `camera: Optional[CameraInterface] = None`, `display_factory: Optional[Callable[..., DisplayInterface]] = None`, `exporter: Optional[ExportInterface] = None`. If None, construct default (picamera.Capture, DisplayManager, CSVExporter). This allows tests and alternative UIs to inject mocks or stubs.
- [ ] **6.1.2** Use the injected display type only via the narrow interface (e.g. `DisplayInterface`); no reliance on `DisplayManager`-specific attributes outside that interface.

### 6.2 Export and camera

- [ ] **6.2.1** Ensure `ExportInterface` (if not already) defines the methods Spectrometer needs (e.g. save spectrum, path). `CSVExporter` implements it. Spectrometer depends on `ExportInterface`, not `CSVExporter`.
- [x] **6.2.2** Camera: `CameraInterface` is the abstraction; Spectrometer accepts `camera: Optional[CameraInterface] = None`.

### 6.3 OpenCV capture backend (ARCHITECTURE §5.1)

- [x] **6.3.1** Add `capture/opencv.py` — `Capture` implementing `CameraInterface`. Source: webcam (int index), v4l path, or rtsp/http URL, from CLI `--camera SOURCE`.
- [x] **6.3.2** Convert frames to **10-bit grayscale**: 2D `uint16`, shape `(height, width)`, values 0–1023. BGR/grayscale from OpenCV → `(gray * 1023 / 255).astype(uint16)`.
- [x] **6.3.3** **Gain / exposure:** No-op (setters accept but have no effect).
- [x] **6.3.4** **List cameras:** `--list-cameras` enumerates available devices; log index/path; exit after listing.
- [x] **6.3.5** **Capabilities on start:** On `start()`, log camera info (resolution, fps); gain/exposure noted as no-op.
- [x] **6.3.6** Add `--camera SOURCE` and `--list-cameras` to `__main__.py`. When `--camera` set, create `opencv.Capture` and pass to `Spectrometer(camera=...)`. No other code changes.

---

## Verification Checklist

After each phase (or at the end), verify:

| Check | How |
|-------|-----|
| No duplicate dark/white logic | Grep for manual dark subtract / white divide; only `reference_correction.apply_dark_white_correction` or `BaseMode.apply_references`. |
| Single button source | Control bar layout comes from mode `get_buttons()` when mode instance is available; no divergent hardcoded list. |
| Spectrometer does not touch display internals | No `_display._control_bar` or `_display.slider_panel` in Spectrometer; use `set_status`, `register_*_callback`, `set_gain_value`, etc. |
| Mode output 0–1 | All `process_spectrum` return 0–1 float; display does scaling. |
| New mode is additive | Adding Waterfall/Raman/Color Science mode = new file under `modes/`, implement `BaseMode`, add to mode list; no edits to Spectrometer’s long callback lists if Phase 3 is done. |
| Tests | Run existing tests; add or adjust tests for extracted components (e.g. `EventHandlerRegistry`, `ReferenceCorrector`, calibrators) as needed. |

---

## Summary Task List (Quick Reference)

| # | Task | Phase |
|---|------|--------|
| 1 | Button layout from mode `get_buttons()`; control bar accepts mode instance | 1.1 |
| 2 | No duplicate dark/white logic; all paths use `reference_correction` / `apply_references` | 1.2 |
| 3 | All overlay polylines via `overlay_utils.render_polyline_overlay` | 1.3 |
| 4 | Central helper for intensity → graph Y scaling | 1.4 |
| 5 | Display: no direct `_control_bar`/`slider_panel` from Spectrometer; add register_slider_callbacks | 2.1 |
| 6 | Optional: `DisplayInterface` and implement on DisplayManager | 2.2 |
| 7 | `EventHandlerRegistry` / `CallbackBinder` centralizes key and button registration | 3.1 |
| 8 | `ModeCoordinator` holds current mode and delegates mode-specific behavior | 3.2 |
| 9 | Extract `SpectrumGraphRenderer` (and optionally `OverlayRenderer`, `PreviewComposer`, `StatusDisplay`) | 4.1–4.3 |
| 10 | Mode contract: `process_spectrum` 0–1; use `apply_references` in MeasurementMode | 5.1–5.2 |
| 11 | Extract `CorrelationCalibrator`, `PeakMatchCalibrator`, `SensitivityCorrector` from CalibrationMode (4-pt min per ARCHITECTURE) | 5.2 |
| 12 | Optional: constructor injection for camera, display factory, exporter | 6.1–6.2 |
| 13 | opencv.Capture: webcam/v4l/rtsp, 10-bit grayscale, --camera/--list-cameras; gain/exposure no-op; log capabilities on start | 6.3 |

---

---

## Alignment with ARCHITECTURE.md

- **Modes:** Refactoring supports all **five** modes (Calibration, Measurement, Waterfall, Raman, Color Science). Button layouts and actions follow ARCHITECTURE §3 button tables (e.g. FL12, SaveCal, LoadCal, LED, I2C…).
- **Layers:** Controller (Spectrometer + EventHandlerRegistry + ModeCoordinator), Viewer (display + gui), Data/Logic (core, processing, modes) — refactor keeps these boundaries clear.
- **Calibration:** 4-point minimum, `match_peaks_to_reference`, Auto-Level, Auto-Center Y as in ARCHITECTURE §3.2.
- **Module map:** ARCHITECTURE §8 is authoritative; new modules (e.g. `modes/waterfall.py`, `capture/opencv.py`) appear there when planned/implemented.
- **Capture:** opencv.Capture (webcam, v4l, rtsp) in `capture/`; selected via `--camera`; outputs 10-bit grayscale; substitutes via `Spectrometer(camera=...)` with no other code changes (ARCHITECTURE §5.1).

---

*Document Version: 2.1 — Task list aligned with ARCHITECTURE.md (five modes, button tables, Controller/Viewer/Data).*  
*Last updated: 2026-03-09*
