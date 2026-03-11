# PySpectrometer3 Makefile for Raspberry Pi
# Initial setup: packages, partitions, safe-shutdown, camera/display
# Poetry venv in /home (writable). System packages (picamera2) from /usr via system-site-packages.

# Raspberry Pi setup (Waveshare 3.5" DPI LCD + OV9281 camera)
WAVESHARE_OVERLAY_URL := https://files.waveshare.com/wiki/3.5inch%20DPI%20LCD/3.5DPI-dtbo.zip
WAVESHARE_OVERLAY_ZIP := .cache/waveshare-35dpi-dtbo.zip
WAVESHARE_OVERLAY_DIR := .cache/waveshare-35dpi-dtbo

.PHONY: all setup-packages setup-packages-recreate setup-partitions setup-partitions-plan setup-safe-shutdown \
        setup-display setup-performance setup-overlays setup-config help

all: help

## Setup targets (run on Raspberry Pi, in order)

setup-packages:
	@echo "Installing system dependencies..."
	sudo apt-get update
	sudo apt-get install -y \
		python3-pip \
		python3-venv \
		libcamera-dev \
		python3-libcamera \
		python3-picamera2 \
		curl \
		unzip \
		parted \
		e2fsprogs
	@echo "Installing Poetry (official installer, apt version is too old for system-site-packages)..."
	curl -sSL https://install.python-poetry.org | python3 -
	@echo "Configuring Poetry: venv in project, system-site-packages..."
	$${HOME}/.local/bin/poetry config virtualenvs.in-project true --local
	$${HOME}/.local/bin/poetry config virtualenvs.options.system-site-packages true --local
	@echo "Installing Python deps via Poetry..."
	$${HOME}/.local/bin/poetry install --no-interaction
	@echo "Packages complete. Add to PATH: export PATH=\"$$HOME/.local/bin:$$PATH\""

setup-packages-recreate:
	@echo "Recreate venv with Poetry..."
	poetry config virtualenvs.in-project true --local
	poetry config virtualenvs.options.system-site-packages true --local
	poetry env remove --all 2>/dev/null || true
	poetry install --no-interaction
	@echo "Done. Run: poetry run calibrate"

setup-partitions:
	@echo "Resizing SD partitions (run when booted from USB)..."
	@chmod +x scripts/setup-partitions-resize.sh
	sudo ./scripts/setup-partitions-resize.sh

setup-partitions-migrate:
	@echo "Migrate /home (run when booted from SD, after manual resize)..."
	@chmod +x scripts/setup-partitions.sh
	./scripts/setup-partitions.sh

setup-safe-shutdown:
	@echo "Configuring logs to RAM and read-only root/boot..."
	@chmod +x scripts/setup-safe-shutdown.sh
	./scripts/setup-safe-shutdown.sh

setup-display: setup-overlays setup-config
	@echo ""
	@echo "Display + camera setup complete. Reboot to apply: sudo reboot"

setup-performance:
	@echo "Configuring performance (CPU governor, overclock, minimal services)..."
	@chmod +x scripts/setup-performance.sh
	./scripts/setup-performance.sh

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
		echo "Overlays directory not found"; exit 1; \
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
		echo "Config not found"; exit 1; \
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

## Help

help:
	@echo "PySpectrometer3 - Raspberry Pi Setup"
	@echo ""
	@echo "Setup order (run on Pi):"
	@echo "  1. make setup-packages      - apt + poetry (venv in /home, reads picamera2 from /usr)"
	@echo "     make setup-packages-recreate - If Picamera2 fails: recreate venv"
	@echo "  2. make setup-partitions     - Resize SD when booted from USB (no GParted)"
	@echo "  3. make setup-safe-shutdown - Logs to RAM, root/boot read-only"
	@echo "  4. make setup-display       - Waveshare 3.5\" + OV9281 camera"
	@echo "  5. make setup-performance   - CPU governor, overclock, disable Bluetooth/avahi (WiFi kept)"
	@echo ""
	@echo "Then: sudo reboot"
	@echo ""
	@echo "Run: poetry run python -m pyspectrometer --waveshare --mode measurement"
	@echo ""
