# PySpectrometer3 Makefile for Raspberry Pi
# Supports installation, running, and desktop integration

PYTHON := python3
PIP := pip3
VENV_DIR := .venv
SRC_DIR := src
PACKAGE := pyspectrometer
DESKTOP_DIR := $(HOME)/Desktop
DESKTOP_FILE := PySpectrometer3.desktop
ICON_FILE := pyspectrometer.png

# Detect if running in virtual environment
ifdef VIRTUAL_ENV
	PIP_INSTALL := $(PIP) install
else
	PIP_INSTALL := $(PIP) install --user
endif

.PHONY: all install install-deps install-system-deps install-desktop \
        install-waveshare install-waveshare-fullscreen \
        install-waveshare-link install-waveshare-fullscreen-link \
        install-link uninstall run run-fullscreen run-waterfall run-waveshare \
        run-calibration run-measurement run-raman run-colorscience \
        calibrate measure raman colors \
        setup-pi setup-overlays setup-config \
        stream clean help venv

# Raspberry Pi setup (Waveshare 3.5" DPI LCD + OV9281 camera)
WAVESHARE_OVERLAY_URL := https://files.waveshare.com/wiki/3.5inch%20DPI%20LCD/3.5DPI-dtbo.zip
WAVESHARE_OVERLAY_ZIP := .cache/waveshare-35dpi-dtbo.zip
WAVESHARE_OVERLAY_DIR := .cache/waveshare-35dpi-dtbo

all: help

## Installation targets

install: install-system-deps install-deps install-desktop
	@echo "Installation complete!"
	@echo "Run 'make run' or double-click the desktop icon to start."

install-system-deps:
	@echo "Installing system dependencies..."
	sudo apt-get update
	sudo apt-get install -y \
		python3-pip \
		python3-venv \
		python3-opencv \
		python3-numpy \
		libatlas-base-dev \
		libcamera-dev \
		python3-picamera2

install-deps:
	@echo "Installing Python dependencies..."
	$(PIP_INSTALL) -r requirements.txt

install-desktop: create-icon create-run-script
	@echo "Creating desktop shortcut..."
	@mkdir -p $(DESKTOP_DIR)
	@echo "[Desktop Entry]" > $(DESKTOP_DIR)/$(DESKTOP_FILE)
	@echo "Name=PySpectrometer3" >> $(DESKTOP_DIR)/$(DESKTOP_FILE)
	@echo "Comment=Spectrometer Application" >> $(DESKTOP_DIR)/$(DESKTOP_FILE)
	@echo "Exec=$(shell pwd)/run.sh" >> $(DESKTOP_DIR)/$(DESKTOP_FILE)
	@echo "Icon=$(shell pwd)/$(SRC_DIR)/resources/$(ICON_FILE)" >> $(DESKTOP_DIR)/$(DESKTOP_FILE)
	@echo "Terminal=false" >> $(DESKTOP_DIR)/$(DESKTOP_FILE)
	@echo "Type=Application" >> $(DESKTOP_DIR)/$(DESKTOP_FILE)
	@echo "Categories=Science;Education;" >> $(DESKTOP_DIR)/$(DESKTOP_FILE)
	@chmod +x $(DESKTOP_DIR)/$(DESKTOP_FILE)
	@echo "Desktop shortcut created at $(DESKTOP_DIR)/$(DESKTOP_FILE)"

install-waveshare: create-icon create-waveshare-script
	@echo "Creating Waveshare 3.5\" desktop shortcut..."
	@mkdir -p $(DESKTOP_DIR)
	@echo "[Desktop Entry]" > $(DESKTOP_DIR)/PySpectrometer3-Waveshare.desktop
	@echo "Name=PySpectrometer3 (Waveshare)" >> $(DESKTOP_DIR)/PySpectrometer3-Waveshare.desktop
	@echo "Comment=Spectrometer for Waveshare 3.5\" Display" >> $(DESKTOP_DIR)/PySpectrometer3-Waveshare.desktop
	@echo "Exec=$(shell pwd)/run-waveshare.sh" >> $(DESKTOP_DIR)/PySpectrometer3-Waveshare.desktop
	@echo "Icon=$(shell pwd)/$(SRC_DIR)/resources/$(ICON_FILE)" >> $(DESKTOP_DIR)/PySpectrometer3-Waveshare.desktop
	@echo "Terminal=false" >> $(DESKTOP_DIR)/PySpectrometer3-Waveshare.desktop
	@echo "Type=Application" >> $(DESKTOP_DIR)/PySpectrometer3-Waveshare.desktop
	@echo "Categories=Science;Education;" >> $(DESKTOP_DIR)/PySpectrometer3-Waveshare.desktop
	@chmod +x $(DESKTOP_DIR)/PySpectrometer3-Waveshare.desktop
	@echo "Desktop shortcut created at $(DESKTOP_DIR)/PySpectrometer3-Waveshare.desktop"

install-waveshare-fullscreen: create-icon
	@echo "Creating Waveshare 3.5\" fullscreen desktop shortcut..."
	@mkdir -p $(DESKTOP_DIR)
	@echo '#!/bin/bash' > run-waveshare-fullscreen.sh
	@echo 'cd "$(shell pwd)"' >> run-waveshare-fullscreen.sh
	@echo 'export DISPLAY=:0' >> run-waveshare-fullscreen.sh
	@echo '$(PYTHON) -m $(PACKAGE) --waveshare --fullscreen "$$@"' >> run-waveshare-fullscreen.sh
	@chmod +x run-waveshare-fullscreen.sh
	@echo "[Desktop Entry]" > $(DESKTOP_DIR)/PySpectrometer3-Waveshare-FS.desktop
	@echo "Name=PySpectrometer3 (Waveshare FS)" >> $(DESKTOP_DIR)/PySpectrometer3-Waveshare-FS.desktop
	@echo "Comment=Spectrometer Fullscreen for Waveshare 3.5\" Display" >> $(DESKTOP_DIR)/PySpectrometer3-Waveshare-FS.desktop
	@echo "Exec=$(shell pwd)/run-waveshare-fullscreen.sh" >> $(DESKTOP_DIR)/PySpectrometer3-Waveshare-FS.desktop
	@echo "Icon=$(shell pwd)/$(SRC_DIR)/resources/$(ICON_FILE)" >> $(DESKTOP_DIR)/PySpectrometer3-Waveshare-FS.desktop
	@echo "Terminal=false" >> $(DESKTOP_DIR)/PySpectrometer3-Waveshare-FS.desktop
	@echo "Type=Application" >> $(DESKTOP_DIR)/PySpectrometer3-Waveshare-FS.desktop
	@echo "Categories=Science;Education;" >> $(DESKTOP_DIR)/PySpectrometer3-Waveshare-FS.desktop
	@chmod +x $(DESKTOP_DIR)/PySpectrometer3-Waveshare-FS.desktop
	@echo "Desktop shortcut created at $(DESKTOP_DIR)/PySpectrometer3-Waveshare-FS.desktop"

install-waveshare-link: create-waveshare-script
	@echo "Creating Waveshare 3.5\" desktop symlink..."
	@mkdir -p $(DESKTOP_DIR)
	@ln -sf $(shell pwd)/run-waveshare.sh $(DESKTOP_DIR)/PySpectrometer3-Waveshare.sh
	@chmod +x $(DESKTOP_DIR)/PySpectrometer3-Waveshare.sh
	@echo "Desktop symlink created at $(DESKTOP_DIR)/PySpectrometer3-Waveshare.sh"

install-waveshare-fullscreen-link: create-waveshare-fullscreen-script
	@echo "Creating Waveshare 3.5\" fullscreen desktop symlink..."
	@mkdir -p $(DESKTOP_DIR)
	@ln -sf $(shell pwd)/run-waveshare-fullscreen.sh $(DESKTOP_DIR)/PySpectrometer3-Waveshare-FS.sh
	@chmod +x $(DESKTOP_DIR)/PySpectrometer3-Waveshare-FS.sh
	@echo "Desktop symlink created at $(DESKTOP_DIR)/PySpectrometer3-Waveshare-FS.sh"

## Mode-specific installation (install-link)
## Creates executable scripts and desktop files for each mode

install-link: create-icon create-mode-scripts create-mode-desktop-files
	@echo ""
	@echo "=============================================="
	@echo "Mode-specific launchers installed!"
	@echo "=============================================="
	@echo ""
	@echo "Desktop shortcuts created:"
	@echo "  - PySpec-Calibration"
	@echo "  - PySpec-Measurement (default)"
	@echo "  - PySpec-Raman"
	@echo "  - PySpec-ColorScience"
	@echo ""
	@echo "Executable scripts in $(DESKTOP_DIR)/:"
	@echo "  - pyspec-calibration.sh"
	@echo "  - pyspec-measurement.sh"
	@echo "  - pyspec-raman.sh"
	@echo "  - pyspec-colorscience.sh"
	@echo ""

create-mode-scripts:
	@echo "Creating mode-specific run scripts..."
	@# Calibration mode
	@echo '#!/bin/bash' > pyspec-calibration.sh
	@echo '# PySpectrometer3 - Calibration Mode' >> pyspec-calibration.sh
	@echo 'cd "$(shell pwd)"' >> pyspec-calibration.sh
	@echo 'export DISPLAY=:0' >> pyspec-calibration.sh
	@echo '$(PYTHON) -m $(PACKAGE) --waveshare --mode calibration "$$@"' >> pyspec-calibration.sh
	@chmod +x pyspec-calibration.sh
	@# Measurement mode
	@echo '#!/bin/bash' > pyspec-measurement.sh
	@echo '# PySpectrometer3 - Measurement Mode' >> pyspec-measurement.sh
	@echo 'cd "$(shell pwd)"' >> pyspec-measurement.sh
	@echo 'export DISPLAY=:0' >> pyspec-measurement.sh
	@echo '$(PYTHON) -m $(PACKAGE) --waveshare --mode measurement "$$@"' >> pyspec-measurement.sh
	@chmod +x pyspec-measurement.sh
	@# Raman mode
	@echo '#!/bin/bash' > pyspec-raman.sh
	@echo '# PySpectrometer3 - Raman Mode' >> pyspec-raman.sh
	@echo 'cd "$(shell pwd)"' >> pyspec-raman.sh
	@echo 'export DISPLAY=:0' >> pyspec-raman.sh
	@echo '$(PYTHON) -m $(PACKAGE) --waveshare --mode raman "$$@"' >> pyspec-raman.sh
	@chmod +x pyspec-raman.sh
	@# Color Science mode
	@echo '#!/bin/bash' > pyspec-colorscience.sh
	@echo '# PySpectrometer3 - Color Science Mode' >> pyspec-colorscience.sh
	@echo 'cd "$(shell pwd)"' >> pyspec-colorscience.sh
	@echo 'export DISPLAY=:0' >> pyspec-colorscience.sh
	@echo '$(PYTHON) -m $(PACKAGE) --waveshare --mode colorscience "$$@"' >> pyspec-colorscience.sh
	@chmod +x pyspec-colorscience.sh
	@echo "Mode scripts created."

create-mode-desktop-files:
	@echo "Creating mode-specific desktop files..."
	@mkdir -p $(DESKTOP_DIR)
	@# Calibration desktop file
	@echo "[Desktop Entry]" > $(DESKTOP_DIR)/PySpec-Calibration.desktop
	@echo "Name=PySpec-Calibration" >> $(DESKTOP_DIR)/PySpec-Calibration.desktop
	@echo "Comment=PySpectrometer3 Calibration Mode" >> $(DESKTOP_DIR)/PySpec-Calibration.desktop
	@echo "Exec=$(shell pwd)/pyspec-calibration.sh" >> $(DESKTOP_DIR)/PySpec-Calibration.desktop
	@echo "Icon=$(shell pwd)/$(SRC_DIR)/resources/$(ICON_FILE)" >> $(DESKTOP_DIR)/PySpec-Calibration.desktop
	@echo "Terminal=false" >> $(DESKTOP_DIR)/PySpec-Calibration.desktop
	@echo "Type=Application" >> $(DESKTOP_DIR)/PySpec-Calibration.desktop
	@echo "Categories=Science;Education;" >> $(DESKTOP_DIR)/PySpec-Calibration.desktop
	@chmod +x $(DESKTOP_DIR)/PySpec-Calibration.desktop
	@# Measurement desktop file (default)
	@echo "[Desktop Entry]" > $(DESKTOP_DIR)/PySpec-Measurement.desktop
	@echo "Name=PySpec-Measurement" >> $(DESKTOP_DIR)/PySpec-Measurement.desktop
	@echo "Comment=PySpectrometer3 Measurement Mode (Default)" >> $(DESKTOP_DIR)/PySpec-Measurement.desktop
	@echo "Exec=$(shell pwd)/pyspec-measurement.sh" >> $(DESKTOP_DIR)/PySpec-Measurement.desktop
	@echo "Icon=$(shell pwd)/$(SRC_DIR)/resources/$(ICON_FILE)" >> $(DESKTOP_DIR)/PySpec-Measurement.desktop
	@echo "Terminal=false" >> $(DESKTOP_DIR)/PySpec-Measurement.desktop
	@echo "Type=Application" >> $(DESKTOP_DIR)/PySpec-Measurement.desktop
	@echo "Categories=Science;Education;" >> $(DESKTOP_DIR)/PySpec-Measurement.desktop
	@chmod +x $(DESKTOP_DIR)/PySpec-Measurement.desktop
	@# Raman desktop file
	@echo "[Desktop Entry]" > $(DESKTOP_DIR)/PySpec-Raman.desktop
	@echo "Name=PySpec-Raman" >> $(DESKTOP_DIR)/PySpec-Raman.desktop
	@echo "Comment=PySpectrometer3 Raman Mode" >> $(DESKTOP_DIR)/PySpec-Raman.desktop
	@echo "Exec=$(shell pwd)/pyspec-raman.sh" >> $(DESKTOP_DIR)/PySpec-Raman.desktop
	@echo "Icon=$(shell pwd)/$(SRC_DIR)/resources/$(ICON_FILE)" >> $(DESKTOP_DIR)/PySpec-Raman.desktop
	@echo "Terminal=false" >> $(DESKTOP_DIR)/PySpec-Raman.desktop
	@echo "Type=Application" >> $(DESKTOP_DIR)/PySpec-Raman.desktop
	@echo "Categories=Science;Education;" >> $(DESKTOP_DIR)/PySpec-Raman.desktop
	@chmod +x $(DESKTOP_DIR)/PySpec-Raman.desktop
	@# Color Science desktop file
	@echo "[Desktop Entry]" > $(DESKTOP_DIR)/PySpec-ColorScience.desktop
	@echo "Name=PySpec-ColorScience" >> $(DESKTOP_DIR)/PySpec-ColorScience.desktop
	@echo "Comment=PySpectrometer3 Color Science Mode" >> $(DESKTOP_DIR)/PySpec-ColorScience.desktop
	@echo "Exec=$(shell pwd)/pyspec-colorscience.sh" >> $(DESKTOP_DIR)/PySpec-ColorScience.desktop
	@echo "Icon=$(shell pwd)/$(SRC_DIR)/resources/$(ICON_FILE)" >> $(DESKTOP_DIR)/PySpec-ColorScience.desktop
	@echo "Terminal=false" >> $(DESKTOP_DIR)/PySpec-ColorScience.desktop
	@echo "Type=Application" >> $(DESKTOP_DIR)/PySpec-ColorScience.desktop
	@echo "Categories=Science;Education;" >> $(DESKTOP_DIR)/PySpec-ColorScience.desktop
	@chmod +x $(DESKTOP_DIR)/PySpec-ColorScience.desktop
	@# Copy scripts to desktop for easy access
	@ln -sf $(shell pwd)/pyspec-calibration.sh $(DESKTOP_DIR)/pyspec-calibration.sh
	@ln -sf $(shell pwd)/pyspec-measurement.sh $(DESKTOP_DIR)/pyspec-measurement.sh
	@ln -sf $(shell pwd)/pyspec-raman.sh $(DESKTOP_DIR)/pyspec-raman.sh
	@ln -sf $(shell pwd)/pyspec-colorscience.sh $(DESKTOP_DIR)/pyspec-colorscience.sh
	@echo "Desktop files created."

create-icon:
	@echo "Creating application icon..."
	@mkdir -p $(SRC_DIR)/resources
	@$(PYTHON) -c "import base64; \
icon_data = 'iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH5goFFDgj33B8iQAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAADsElEQVR42u2bTW7bMBCFnwKvkC7SLrJIF10kq6yDLrpKFllmkXWWWSXrLLvKKuugi6zSRVdBl9k0i4heGJSoH1uyZFl+AwMKKZHk8M0MSVEEYDAYDAaDwWAwGAwGg8FgOGUQAJydnZHP59nv9/Q/cjY7o+VyyWq1opKSEvr9+w9RCQktLy+xbZuKi4uppqaG6urqKC0t7dR0jPz+fWZmhjw7O0sr4+NUVFhIBQUFlJGRQQqFgrq7u+nxkyeUl5dHVVVVTHHPnz8nm83GFJaamkp1dXXU09NDK8vLbP4RUAoA4OzslC5evEjp6ek0NDRE5y9coLm5OTYtPz+fhoeHafCffxiHV65cob6+Pjp//jwRAyD4y8tLmpqaYuqYmJigsrIyUqvVzHRYPuLly5csb0dHB42Pj1NxcTFpaWkBLQHe3NxkCrhy5QoNDAwwNWhpaUmt11dXVVWx5dPT0+ny5cu0sLBAs7OzVFlZyUxmfHycaSQ5OZny8/PpypUrNDw0xOZZWVlJcrmc9u3bRysrK6y8zMxM0mg0lJKSQoPXrtH09DTNzMxQVVUVJScn09jYGNu+np4e6u7uplsDA2x5xQUFpFKpqLCwkNQ7O8nMzAxNTk5SRkYGlZeXs23Z0tJC69evU3t7O7W3t9PAQD+lpKQQpjExMUFFRUW0Y8cOtrIXy6C+vp4mJyepoKCAdDodffjwgV3Rq1evsjVz9epV9hxdXV2sTPDqzQ0NDfTixQsqLi5mGnzy5AlTR319PU1MTFBRURFrnLt27SKc6PT0ND14+JCqqqpIo9EwF/3x4wfb88Pp4PR3795RWVkZ5efn0+XBQaqqrGRu4vy580StrbS4uEhDQ0PU0tJChYWFTANv3rxh1dbW1kZjY2OUl5dHhYWFdPPmTWpoaGAl1Wg07EpsbGyky1euMC1dvXqVysrL2frz8vJ4MxQVFdG5c+eYVu7evUuVlZWUn59PjY2NzJWAGhoaaOTRI1KpVKTRaOjatWvU1dVFSqWSNBoN3b9/nyorK5kWJiYmaG5ujsrLy0mv19OjR4+ourqarYfe3l4aHh6m0tJSdm0/e/aMOjo6mLu4desWOzc5OUl6vZ5SUlJoaGiIrl27RioVwuFwO4F/vnrlSgsOh8NJYIBwOJxMIA+F5x6HAc8G4VkJPF+A50JwXv8NfDYE54b+Hp4Nw3MR+GxYAQifDMF5Jfy/wrMReE4Mnw0roOCTIQVMfDaigIvPhhRw8cmQAjY+GVLAxidDCvj4bEgBH58NKRLgsyFFQnwypEiIT4YUifHJkOIvfDKk+A94NqT4C58MKVLikyFFanwypEiNT4YU/wAHfz8o8Tj0GAAAAABJRU5ErkJggg=='; \
open('$(SRC_DIR)/resources/$(ICON_FILE)', 'wb').write(base64.b64decode(icon_data))" 2>/dev/null || \
	$(PYTHON) -c "print('Icon creation skipped - will use default icon')"

create-run-script:
	@echo "Creating run script..."
	@echo '#!/bin/bash' > run.sh
	@echo 'cd "$(shell pwd)"' >> run.sh
	@echo 'export DISPLAY=:0' >> run.sh
	@echo '$(PYTHON) -m $(PACKAGE) "$$@"' >> run.sh
	@chmod +x run.sh

create-waveshare-script:
	@echo "Creating Waveshare 3.5\" run script..."
	@echo '#!/bin/bash' > run-waveshare.sh
	@echo 'cd "$(shell pwd)"' >> run-waveshare.sh
	@echo 'export DISPLAY=:0' >> run-waveshare.sh
	@echo '$(PYTHON) -m $(PACKAGE) --waveshare "$$@"' >> run-waveshare.sh
	@chmod +x run-waveshare.sh

create-waveshare-fullscreen-script:
	@echo "Creating Waveshare 3.5\" fullscreen run script..."
	@echo '#!/bin/bash' > run-waveshare-fullscreen.sh
	@echo 'cd "$(shell pwd)"' >> run-waveshare-fullscreen.sh
	@echo 'export DISPLAY=:0' >> run-waveshare-fullscreen.sh
	@echo '$(PYTHON) -m $(PACKAGE) --waveshare --fullscreen "$$@"' >> run-waveshare-fullscreen.sh
	@chmod +x run-waveshare-fullscreen.sh

## Uninstall target

uninstall:
	@echo "Removing desktop shortcuts..."
	@rm -f $(DESKTOP_DIR)/$(DESKTOP_FILE)
	@rm -f $(DESKTOP_DIR)/PySpectrometer3-Waveshare.desktop
	@rm -f $(DESKTOP_DIR)/PySpectrometer3-Waveshare-FS.desktop
	@rm -f $(DESKTOP_DIR)/PySpectrometer3-Waveshare.sh
	@rm -f $(DESKTOP_DIR)/PySpectrometer3-Waveshare-FS.sh
	@rm -f $(DESKTOP_DIR)/PySpec-Calibration.desktop
	@rm -f $(DESKTOP_DIR)/PySpec-Measurement.desktop
	@rm -f $(DESKTOP_DIR)/PySpec-Raman.desktop
	@rm -f $(DESKTOP_DIR)/PySpec-ColorScience.desktop
	@rm -f $(DESKTOP_DIR)/pyspec-calibration.sh
	@rm -f $(DESKTOP_DIR)/pyspec-measurement.sh
	@rm -f $(DESKTOP_DIR)/pyspec-raman.sh
	@rm -f $(DESKTOP_DIR)/pyspec-colorscience.sh
	@rm -f run.sh run-waveshare.sh run-waveshare-fullscreen.sh
	@rm -f pyspec-calibration.sh pyspec-measurement.sh pyspec-raman.sh pyspec-colorscience.sh
	@echo "Uninstall complete. Dependencies not removed."

## Raspberry Pi setup (Waveshare 3.5" DPI LCD + OV9281 camera)
## Run on the Pi: make setup-pi && sudo reboot

setup-pi: setup-overlays setup-config
	@echo ""
	@echo "Setup complete! Reboot to apply: sudo reboot"

setup-overlays:
	@echo "Downloading Waveshare 3.5\" DPI LCD overlays..."
	@mkdir -p .cache
	@if [ ! -f $(WAVESHARE_OVERLAY_ZIP) ]; then \
		curl -sL -o $(WAVESHARE_OVERLAY_ZIP) "$(WAVESHARE_OVERLAY_URL)" || \
		wget -q -O $(WAVESHARE_OVERLAY_ZIP) "$(WAVESHARE_OVERLAY_URL)" || \
		{ echo "Failed to download. Get manually: $(WAVESHARE_OVERLAY_URL)"; exit 1; }; \
	fi
	@echo "Extracting overlays..."
	@rm -rf $(WAVESHARE_OVERLAY_DIR)
	@unzip -q -o $(WAVESHARE_OVERLAY_ZIP) -d $(WAVESHARE_OVERLAY_DIR)
	@echo "Copying overlays to boot..."
	@if [ -d /boot/firmware/overlays ]; then \
		OVERLAYS_DIR=/boot/firmware/overlays; \
	elif [ -d /boot/overlays ]; then \
		OVERLAYS_DIR=/boot/overlays; \
	else \
		echo "Overlays directory not found (expected /boot/firmware/overlays or /boot/overlays)"; exit 1; \
	fi; \
	for f in $$(find $(WAVESHARE_OVERLAY_DIR) -name "*.dtbo" 2>/dev/null); do \
		sudo cp "$$f" "$$OVERLAYS_DIR/" && echo "  Installed $$(basename $$f)"; \
	done
	@echo "Overlays installed."

setup-config:
	@echo "Updating boot config for Waveshare 3.5\" DPI LCD + OV9281..."
	@if [ -f /boot/firmware/config.txt ]; then \
		CONFIG=/boot/firmware/config.txt; \
	elif [ -f /boot/config.txt ]; then \
		CONFIG=/boot/config.txt; \
	else \
		echo "Config not found (expected /boot/firmware/config.txt or /boot/config.txt)"; exit 1; \
	fi; \
	if grep -q "# PySpectrometer3 setup" "$$CONFIG" 2>/dev/null; then \
		echo "  Config already contains PySpectrometer3 setup (skipping)"; \
	else \
		{ echo ""; echo "# PySpectrometer3 setup - Waveshare 3.5\" DPI LCD + OV9281"; \
		  echo "camera_auto_detect=0"; echo "display_auto_detect=0"; \
		  echo "dtoverlay=vc4-kms-v3d"; echo "dtoverlay=waveshare-35dpi"; \
		  echo "dtoverlay=waveshare-touch-35dpi"; echo "max_framebuffers=2"; \
		  echo "dtoverlay=ov9281,arducam"; echo "enable_uart=1"; echo "gpio=22=op,dl"; } | sudo tee -a "$$CONFIG" > /dev/null; \
		echo "  Config updated."; \
	fi

## Run targets

run: create-run-script
	@echo "Starting PySpectrometer3..."
	cd $(SRC_DIR) && $(PYTHON) -m $(PACKAGE)

run-fullscreen: create-run-script
	@echo "Starting PySpectrometer3 in fullscreen mode..."
	cd $(SRC_DIR) && $(PYTHON) -m $(PACKAGE) --fullscreen

run-waterfall: create-run-script
	@echo "Starting PySpectrometer3 with waterfall display..."
	cd $(SRC_DIR) && $(PYTHON) -m $(PACKAGE) --waterfall

run-waveshare: create-waveshare-script
	@echo "Starting PySpectrometer3 for Waveshare 3.5\" display..."
	cd $(SRC_DIR) && $(PYTHON) -m $(PACKAGE) --waveshare

run-waveshare-fullscreen: create-waveshare-script
	@echo "Starting PySpectrometer3 for Waveshare 3.5\" display (fullscreen)..."
	cd $(SRC_DIR) && $(PYTHON) -m $(PACKAGE) --waveshare --fullscreen

## Mode-specific run targets

run-calibration:
	@echo "Starting PySpectrometer3 in Calibration mode..."
	cd $(SRC_DIR) && $(PYTHON) -m $(PACKAGE) --waveshare --mode calibration

run-measurement:
	@echo "Starting PySpectrometer3 in Measurement mode..."
	cd $(SRC_DIR) && $(PYTHON) -m $(PACKAGE) --waveshare --mode measurement

run-raman:
	@echo "Starting PySpectrometer3 in Raman mode..."
	cd $(SRC_DIR) && $(PYTHON) -m $(PACKAGE) --waveshare --mode raman

run-colorscience:
	@echo "Starting PySpectrometer3 in Color Science mode..."
	cd $(SRC_DIR) && $(PYTHON) -m $(PACKAGE) --waveshare --mode colorscience

## Short aliases for modes

calibrate:
	@echo "Starting PySpectrometer3 in Calibration mode..."
	cd $(SRC_DIR) && $(PYTHON) -m $(PACKAGE) --waveshare --mode calibration

measure:
	@echo "Starting PySpectrometer3 in Measurement mode..."
	cd $(SRC_DIR) && $(PYTHON) -m $(PACKAGE) --waveshare --mode measurement

raman:
	@echo "Starting PySpectrometer3 in Raman mode..."
	cd $(SRC_DIR) && $(PYTHON) -m $(PACKAGE) --waveshare --mode raman

colors:
	@echo "Starting PySpectrometer3 in Color Science mode..."
	cd $(SRC_DIR) && $(PYTHON) -m $(PACKAGE) --waveshare --mode colorscience

## Remote development - MJPEG stream (run on Pi, connect from desktop)

stream:
	@echo "Starting MJPEG stream on port 8000..."
	@echo "On desktop, run: python -m $(PACKAGE) --camera http://<Pi-IP>:8000/stream.mjpg"
	$(PYTHON) scripts/stream_camera.py --port 8000

## Development targets

venv:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "Activate with: source $(VENV_DIR)/bin/activate"

lint:
	@echo "Running linter..."
	cd $(SRC_DIR) && $(PYTHON) -m flake8 $(PACKAGE) || true

typecheck:
	@echo "Running type checker..."
	cd $(SRC_DIR) && $(PYTHON) -m mypy $(PACKAGE) || true

test:
	@echo "Running tests..."
	cd $(SRC_DIR) && $(PYTHON) -m pytest $(PACKAGE) -v || true

## Clean targets

clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache 2>/dev/null || true

## Help

help:
	@echo "PySpectrometer3 Makefile"
	@echo ""
	@echo "Installation:"
	@echo "  make install                   - Full installation (deps + standard desktop shortcut)"
	@echo "  make install-system-deps       - Install system dependencies (requires sudo)"
	@echo "  make install-deps              - Install Python dependencies"
	@echo "  make install-desktop           - Create standard desktop shortcut (800x480)"
	@echo "  make install-waveshare         - Create Waveshare 3.5\" desktop shortcut"
	@echo "  make install-waveshare-fullscreen - Create Waveshare fullscreen shortcut"
	@echo "  make install-link              - Create mode-specific desktop shortcuts (RECOMMENDED)"
	@echo "  make uninstall                 - Remove all desktop shortcuts"
	@echo ""
	@echo "Raspberry Pi setup (Waveshare 3.5\" + OV9281):"
	@echo "  make setup-pi                  - Download overlays, update config (run on Pi, then reboot)"
	@echo "  make setup-overlays            - Download and install Waveshare DPI LCD overlays"
	@echo "  make setup-config              - Append display+camera config to boot config.txt"
	@echo ""
	@echo "Running (General):"
	@echo "  make run                       - Run in windowed mode (800x480)"
	@echo "  make run-fullscreen            - Run in fullscreen mode"
	@echo "  make run-waterfall             - Run with waterfall display"
	@echo "  make run-waveshare             - Run for Waveshare 3.5\" display (640x480)"
	@echo "  make run-waveshare-fullscreen  - Run fullscreen on Waveshare 3.5\""
	@echo ""
	@echo "Running (Modes):"
	@echo "  make calibrate                 - Run Calibration mode (alias)"
	@echo "  make measure                   - Run Measurement mode (alias)"
	@echo "  make raman                     - Run Raman mode (alias)"
	@echo "  make colors                    - Run Color Science mode (alias)"
	@echo "  make run-calibration           - Run Calibration mode"
	@echo "  make run-measurement           - Run Measurement mode"
	@echo "  make run-raman                 - Run Raman mode"
	@echo "  make run-colorscience          - Run Color Science mode"
	@echo ""
	@echo "Remote development:"
	@echo "  make stream                    - Start MJPEG HTTP stream (run on Pi)"
	@echo "                                 Use --camera http://<Pi-IP>:8000/stream.mjpg on desktop"
	@echo ""
	@echo "Development:"
	@echo "  make venv                      - Create virtual environment"
	@echo "  make lint                      - Run linter"
	@echo "  make typecheck                 - Run type checker"
	@echo "  make test                      - Run tests"
	@echo "  make clean                     - Remove cached files"
	@echo ""
	@echo "Operating Modes:"
	@echo "  Calibration    - Wavelength calibration with FL/Hg/Sun references"
	@echo "  Measurement    - General spectrum measurement"
	@echo "  Raman          - Raman spectroscopy (785nm laser)"
	@echo "  ColorScience   - Transmittance/Reflectance/CRI/XYZ"
	@echo ""
	@echo "Display Modes:"
	@echo "  Default (800x480)              - Standard RPi LCD"
	@echo "  Waveshare 3.5\" (640x480)       - Small touchscreen"
	@echo "  Fullscreen                     - Native display resolution"
