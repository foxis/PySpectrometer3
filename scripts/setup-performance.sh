#!/bin/bash
# Performance tuning + minimal services (WiFi only).
# CPU governor=performance, light overclock, disable Bluetooth/avahi/etc.

set -e

BOOT_DIR="/boot/firmware"
[ -d "$BOOT_DIR" ] || BOOT_DIR="/boot"
CONFIG="$BOOT_DIR/config.txt"

# 1. CPU governor: performance (persistent)
echo "Setting CPU governor to performance..."
sudo tee /etc/systemd/system/cpu-performance.service << 'EOF'
[Unit]
Description=Set CPU governor to performance
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/bin/sh -c 'for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > $f 2>/dev/null || true; done'
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable cpu-performance.service
sudo systemctl start cpu-performance.service 2>/dev/null || true

# 2. config.txt: overclock + gpu_mem (Pi Zero 2 W friendly)
if ! grep -q "# PySpectrometer3 performance" "$CONFIG" 2>/dev/null; then
    echo "Adding performance config to config.txt..."
    {
        echo ""
        echo "# PySpectrometer3 performance"
        echo "arm_freq=1100"
        echo "over_voltage=2"
        echo "gpu_mem=64"
    } | sudo tee -a "$CONFIG" > /dev/null
    echo "  arm_freq=1100, over_voltage=2, gpu_mem=64"
fi

# 3. Disable services (keep WiFi only)
echo "Disabling unnecessary services..."
for svc in bluetooth hciuart avahi-daemon avahi-daemon.socket \
    triggerhappy geoclue geoclue-provider-mlsdb \
    cups cups-browsed; do
    sudo systemctl disable "$svc" 2>/dev/null && echo "  Disabled $svc" || true
    sudo systemctl stop "$svc" 2>/dev/null || true
done

echo ""
echo "Performance setup complete."
echo "  - CPU governor: performance"
echo "  - Overclock: 1100 MHz (Pi Zero 2 W)"
echo "  - Disabled: Bluetooth, avahi, printing, etc. (WiFi kept)"
echo "Reboot to apply config: sudo reboot"
