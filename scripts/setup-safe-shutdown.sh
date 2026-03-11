#!/bin/bash
# Configure Raspberry Pi for safe power-off: logs to RAM, root and boot read-only.
# /home must be a separate writable partition (run setup-partitions first).
# Ref: https://www.dzombak.com/blog/2024/03/Running-a-Raspberry-Pi-with-a-read-only-root-filesystem.html

set -e

# Ensure /home is a separate partition before making root read-only
ROOT_SRC=$(findmnt -n -o SOURCE / 2>/dev/null)
HOME_SRC=$(findmnt -n -o SOURCE /home 2>/dev/null)
if [ -z "$HOME_SRC" ] || [ "$ROOT_SRC" = "$HOME_SRC" ]; then
    echo "Error: /home is not a separate partition (it's on root)."
    echo "Run 'make setup-partitions' first. Making root read-only would lock you out of /home."
    exit 1
fi

BOOT_DIR="/boot/firmware"
[ -d "$BOOT_DIR" ] || BOOT_DIR="/boot"

need_reboot=0

# 1. Disable swap
echo "Disabling swap..."
sudo dphys-swapfile swapoff 2>/dev/null || true
sudo systemctl disable dphys-swapfile 2>/dev/null || true

# 2. Add fsck.mode=skip noswap to cmdline
CMDLINE="$BOOT_DIR/cmdline.txt"
if ! grep -q "fsck.mode=skip" "$CMDLINE" 2>/dev/null; then
    echo "Updating cmdline.txt..."
    sudo sed -i 's/$/ fsck.mode=skip noswap/' "$CMDLINE"
    need_reboot=1
fi

# 3. Add tmpfs mounts for logs and temp (logs to RAM)
FSTAB="/etc/fstab"
if ! grep -q "# PySpectrometer3 tmpfs" "$FSTAB" 2>/dev/null; then
    echo "Adding tmpfs mounts to fstab..."
    sudo tee -a "$FSTAB" << 'FSTAB_EOF'

# PySpectrometer3 tmpfs - logs and temp in RAM (safe power-off)
tmpfs  /tmp               tmpfs  defaults,noatime,nosuid,nodev,size=64m   0  0
tmpfs  /var/tmp           tmpfs  defaults,noatime,nosuid,nodev,size=32m   0  0
tmpfs  /var/log           tmpfs  defaults,noatime,nosuid,nodev,noexec,size=64m  0  0
tmpfs  /var/lib/logrotate tmpfs  defaults,noatime,nosuid,nodev,size=1m,mode=0755  0  0
tmpfs  /var/lib/sudo      tmpfs  defaults,noatime,nosuid,nodev,size=1m,mode=0700  0  0
FSTAB_EOF
    need_reboot=1
fi

# 4. Limit journald to fit in /var/log tmpfs
JOURNALD_CONF="/etc/systemd/journald.conf"
if ! grep -q "SystemMaxUse=25M" "$JOURNALD_CONF" 2>/dev/null; then
    echo "Limiting journald size..."
    sudo sed -i 's/^#\?SystemMaxUse=.*/SystemMaxUse=25M/' "$JOURNALD_CONF"
fi

# 5. Add ro to cmdline for read-only root
if ! grep -q " ro$" "$CMDLINE" 2>/dev/null && ! grep -q " ro " "$CMDLINE" 2>/dev/null; then
    echo "Adding read-only root to cmdline..."
    sudo sed -i 's/$/ ro/' "$CMDLINE"
    need_reboot=1
fi

# 6. Update fstab: root and boot as read-only
echo "Updating fstab for read-only root and boot..."
if ! grep -q "defaults,noatime,ro" "$FSTAB" 2>/dev/null; then
    sudo cp "$FSTAB" "${FSTAB}.bak"
    # Root: add noatime,ro (match line with " / " as mount point)
    sudo sed -i '/[[:space:]]\/[[:space:]]/s/defaults/defaults,noatime,ro/' "$FSTAB"
    sudo sed -i '/[[:space:]]\/[[:space:]]/s/,ro,ro/,ro/' "$FSTAB"
    # Boot: add ro (match /boot/firmware or /boot)
    sudo sed -i '/\/boot\/firmware\|[[:space:]]\/boot[[:space:]]/s/defaults/defaults,ro/' "$FSTAB"
    sudo sed -i '/\/boot\/firmware\|[[:space:]]\/boot[[:space:]]/s/,ro,ro/,ro/' "$FSTAB"
    need_reboot=1
fi

# 7. Add rw/ro aliases for maintenance
BASHRC="/etc/bash.bashrc"
if ! grep -q "alias ro=" "$BASHRC" 2>/dev/null; then
    echo "Adding rw/ro aliases..."
    sudo tee -a "$BASHRC" << 'BASHRC_EOF'

# PySpectrometer3 - remount root for maintenance
alias ro='sudo mount -o remount,ro / ; sudo mount -o remount,ro /boot/firmware 2>/dev/null || sudo mount -o remount,ro /boot'
alias rw='sudo mount -o remount,rw / ; sudo mount -o remount,rw /boot/firmware 2>/dev/null || sudo mount -o remount,rw /boot'
BASHRC_EOF
fi

# 8. Disable apt timers (optional - system is frozen)
sudo systemctl mask apt-daily.timer 2>/dev/null || true
sudo systemctl mask apt-daily-upgrade.timer 2>/dev/null || true

echo ""
echo "Safe-shutdown setup complete."
echo "  - Logs in RAM (/var/log tmpfs)"
echo "  - Root and boot read-only"
echo "  - Use 'rw' to remount for updates, 'ro' when done"
echo "  - Note: Clock may reset on power loss; NTP will correct on next boot"
if [ "$need_reboot" = "1" ]; then
    echo ""
    echo "Reboot to apply: sudo reboot"
fi
