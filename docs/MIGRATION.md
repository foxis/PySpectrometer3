# PySpectrometer3 Migration Guide

This document describes the refactoring from the monolithic `PySpectrometer2-Picam2-v1.0.py` to the modular `pyspectrometer` package.

## Feature Parity

| Feature | Original | Refactored | Notes |
|---------|----------|------------|-------|
| Camera capture (Picamera2) | ✅ | ✅ | `capture/picamera.py` |
| Frame cropping & averaging | ✅ | ✅ | `CameraInterface.extract_spectrum_region()` |
| Wavelength calibration | ✅ | ✅ | `core/calibration.py` |
| 2nd/3rd order polynomial fit | ✅ | ✅ | Auto-selects based on point count |
| Calibration file I/O | ✅ | ✅ | `Calibration.load()/.save()` |
| Savitzky-Golay filter | ✅ | ✅ | `processing/filters.py` |
| Peak detection | ✅ | ✅ | `processing/peak_detection.py` |
| Peak hold mode | ✅ | ✅ | `DisplayManager.state.hold_peaks` |
| Spectrum graph display | ✅ | ✅ | `display/renderer.py` |
| Graticule lines (10nm/50nm) | ✅ | ✅ | `display/graticule.py` |
| Waterfall display | ✅ | ✅ | `display/waterfall.py` |
| Fullscreen mode | ✅ | ✅ | `--fullscreen` argument |
| Measurement cursor | ✅ | ✅ | `DisplayManager.state.cursor` |
| Pixel mode (calibration) | ✅ | ✅ | Toggle with 'p' key |
| CSV export | ✅ | ✅ | `export/csv_exporter.py` |
| PNG snapshot | ✅ | ✅ | `Spectrometer._save_snapshot()` |
| Keyboard shortcuts | ✅ | ✅ | `input/keyboard.py` |
| Camera gain control | ✅ | ✅ | 't'/'g' keys |
| Status messages | ✅ | ✅ | Rendered in display |
| **Rotated spectrum extraction** | ❌ | ✅ | `processing/extraction.py` |
| **Multiple extraction methods** | ❌ | ✅ | Median, Weighted Sum, Gaussian |
| **Auto angle detection** | ❌ | ✅ | Hough transform based |

## New Architecture Benefits

### Extensibility

Adding new features is now straightforward:

1. **New processors**: Implement `ProcessorInterface` and add to pipeline
2. **New export formats**: Implement `ExporterInterface`
3. **New input methods**: Implement `InputHandler` (GPIO, etc.)
4. **New camera backends**: Implement `CameraInterface`

### Example: Adding a Reference Spectrum Matcher

```python
from pyspectrometer.processing.base import ProcessorInterface
from pyspectrometer.core.spectrum import SpectrumData

class ReferenceMatcherProcessor(ProcessorInterface):
    def __init__(self, reference_library_path: str):
        self.library = load_reference_library(reference_library_path)
    
    @property
    def name(self) -> str:
        return "Reference Matcher"
    
    def process(self, data: SpectrumData) -> SpectrumData:
        matches = self.find_matches(data.intensity, self.library)
        # Add matches to data (extend SpectrumData if needed)
        return data
```

Then add to the pipeline:

```python
spectrometer.pipeline.add(ReferenceMatcherProcessor("references/"))
```

### Example: Adding GPIO Input

```python
from pyspectrometer.input.keyboard import Action

class GPIOHandler:
    def __init__(self, spectrometer):
        self.spectrometer = spectrometer
        # Setup GPIO pins...
    
    def on_button_press(self, pin):
        if pin == SAVE_BUTTON:
            self.spectrometer._on_save()
        elif pin == HOLD_BUTTON:
            self.spectrometer._on_toggle_hold()
```

## Running the Refactored Version

```bash
# From the src directory
python -m pyspectrometer

# With options
python -m pyspectrometer --fullscreen
python -m pyspectrometer --waterfall
python -m pyspectrometer --gain 15

# For Waveshare 3.5" touchscreen (480x320)
python -m pyspectrometer --waveshare
python -m pyspectrometer --waveshare --fullscreen
```

## Raspberry Pi Installation

Use the Makefile for easy installation on Raspberry Pi:

```bash
# Full installation (deps + desktop shortcut)
make install

# Or step by step:
make install-system-deps   # Install system packages (requires sudo)
make install-deps          # Install Python packages
make install-desktop       # Create desktop shortcut

# Run commands
make run                   # Standard 800x480 mode
make run-fullscreen        # Fullscreen mode
make run-waveshare         # Waveshare 3.5" (480x320)
make run-waveshare-fullscreen  # Waveshare fullscreen

# Uninstall
make uninstall             # Remove desktop shortcut
```

## Display Configurations

| Display | Resolution | Command |
|---------|------------|---------|
| Standard RPi LCD | 800x480 | `make run` or `--fullscreen` |
| Waveshare 3.5" | 480x320 | `make run-waveshare` or `--waveshare` |
| Custom | Any | `--width X --height Y` |

The Waveshare preset optimizes:
- Smaller graph height (200px vs 320px)
- Smaller preview height (50px vs 80px)
- Smaller fonts (0.3 vs 0.4 scale)
- Adjusted status text positions
- Smaller peak detection distance (30 vs 50)

## Project Structure

```
pyspectrometer/
├── __init__.py          # Package exports
├── __main__.py          # Entry point (argparse)
├── config.py            # Configuration dataclasses
├── spectrometer.py      # Main orchestrator
├── core/
│   ├── spectrum.py      # SpectrumData, Peak
│   └── calibration.py   # Wavelength calibration
├── capture/
│   ├── base.py          # CameraInterface ABC
│   └── picamera.py      # Picamera2 implementation
├── processing/
│   ├── base.py          # ProcessorInterface ABC
│   ├── pipeline.py      # ProcessingPipeline
│   ├── filters.py       # SavitzkyGolayFilter
│   ├── peak_detection.py # PeakDetector
│   └── extraction.py    # SpectrumExtractor (rotation, methods)
├── display/
│   ├── renderer.py      # DisplayManager
│   ├── graticule.py     # GraticuleRenderer
│   └── waterfall.py     # WaterfallDisplay
├── export/
│   ├── base.py          # ExporterInterface ABC
│   └── csv_exporter.py  # CSVExporter
├── input/
│   └── keyboard.py      # KeyboardHandler, Action enum
└── utils/
    └── color.py         # wavelength_to_rgb
```

## Configuration

All configuration is now centralized in `Config` dataclass:

```python
from pyspectrometer import Config, Spectrometer

config = Config()
config.camera.gain = 15.0
config.display.fullscreen = True
config.processing.savgol_poly = 9

spectrometer = Spectrometer(config)
spectrometer.run()
```

## Breaking Changes

1. **No global state**: All state is encapsulated in classes
2. **Module structure**: Import paths changed (e.g., `from pyspectrometer.utils.color import wavelength_to_rgb`)
3. **Entry point**: Now `python -m pyspectrometer` instead of running the script directly

## Spectrum Extraction

The new extraction system handles rotated spectra and vertical coupling structure.

### Extraction Methods

| Method | Key | Best For |
|--------|-----|----------|
| Weighted Sum | `e` (cycle) | General use, low-medium signals (default) |
| Median | `e` (cycle) | Noisy conditions, high signals, hot pixels |
| Gaussian | `e` (cycle) | Precision measurements, publication quality |

### Extraction Controls

| Key | Action |
|-----|--------|
| `e` | Cycle extraction method (Median → Weighted Sum → Gaussian) |
| `E` (Shift+e) | Auto-detect rotation angle using Hough transform |
| `]` | Increase perpendicular sampling width |
| `[` | Decrease perpendicular sampling width |
| `w` | Save extraction parameters to calibration file |

### Calibration File Format (Extended)

The calibration file now stores extraction parameters:

```
100,300,500,700          # pixel positions
400.0,500.0,600.0,700.0  # wavelengths
7.5                      # rotation angle (degrees)
240                      # spectrum Y center (pixels)
25                       # perpendicular width (pixels)
```

## Future Extensions (Planned)

- [ ] JSON exporter
- [ ] MQTT streaming exporter
- [ ] GPIO input handler
- [ ] Reference spectra library and matcher
- [ ] USB webcam capture backend
- [ ] Headless mode (data logging only)
- [ ] Web interface for remote viewing
