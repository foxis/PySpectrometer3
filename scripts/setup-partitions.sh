#!/bin/bash
# Create separate writable /home partition. Root and boot will be read-only.
#
# Layout: root = used + 4GB (+ 256MB buffer), home = remainder
# No swap partition (swap on SD causes wear; we disable swap in setup-safe-shutdown)
# Logs/temp: tmpfs in RAM (setup-safe-shutdown), not on disk
#
# Run with --plan to get sizes, then use GParted to resize. Run without --plan
# after creating the home partition to migrate.

set -e

ROOT_BUFFER_MB=256
ROOT_EXTRA_GB=4

plan() {
    echo "=== Partition plan (root = used + ${ROOT_EXTRA_GB}GB + ${ROOT_BUFFER_MB}MB buffer) ==="
    echo ""

    ROOT_DEV=$(findmnt -n -o SOURCE / 2>/dev/null | head -1)
    [ -n "$ROOT_DEV" ] || { echo "Could not find root device"; exit 1; }

    USED_KB=$(df -B1K / | tail -1 | awk '{print $3}')
    TOTAL_KB=$(df -B1K / | tail -1 | awk '{print $2}')
    # root = used + 4GB + 256MB buffer
    ROOT_SIZE_KB=$((USED_KB + ROOT_EXTRA_GB * 1024 * 1024 + ROOT_BUFFER_MB * 1024))
    ROOT_SIZE_MB=$((ROOT_SIZE_KB / 1024))
    HOME_SIZE_KB=$((TOTAL_KB - ROOT_SIZE_KB))
    HOME_SIZE_MB=$((HOME_SIZE_KB / 1024))
    USED_GB=$((USED_KB / 1024 / 1024))
    TOTAL_GB=$((TOTAL_KB / 1024 / 1024))

    echo "Root device: $ROOT_DEV"
    echo "Root used:   ${USED_GB} GB"
    echo "Card total:  ${TOTAL_GB} GB"
    echo ""
    echo "Target sizes:"
    echo "  Root: ${ROOT_SIZE_MB} MB (used + ${ROOT_EXTRA_GB}GB + ${ROOT_BUFFER_MB}MB buffer)"
    echo "  Home: ${HOME_SIZE_MB} MB (remainder)"
    echo ""
    echo "Steps:"
    echo "  1. Boot from GParted (e.g. Raspberry Pi Imager → Other → GParted)"
    echo "  2. Shrink root partition to ${ROOT_SIZE_MB} MB"
    echo "  3. Create new ext4 partition in freed space (label: home)"
    echo "  4. Reboot into Raspberry Pi OS"
    echo "  5. Run: make setup-partitions"
    echo ""
    echo "Note: No swap partition (bad for SD wear). Logs use tmpfs (RAM) via setup-safe-shutdown."
    exit 0
}

[ "$1" = "--plan" ] && plan

# --- Migration (run after creating home partition) ---

# Detect home partition - mmcblk0p3 or sda1
HOME_PART=""
for p in /dev/mmcblk0p3 /dev/sda1; do
    if [ -b "$p" ]; then
        if findmnt -n "$p" 2>/dev/null | grep -q /home; then
            echo "/home is already a separate partition ($p). Nothing to do."
            exit 0
        fi
        if ! findmnt -n "$p" 2>/dev/null; then
            HOME_PART="$p"
            break
        fi
    fi
done

if [ -z "$HOME_PART" ]; then
    echo "No unallocated home partition found."
    echo ""
    echo "Run with --plan first: ./scripts/setup-partitions.sh --plan"
    echo "Then use GParted to shrink root and create home partition."
    exit 1
fi

echo "Found home partition: $HOME_PART"
echo "This will format it and migrate /home. Existing /home will be backed up to /home.bak.old"
read -p "Continue? [y/N] " -n 1 -r
echo
[[ $REPLY =~ ^[Yy]$ ]] || exit 1

# Format if not already ext4
FSTYPE=$(blkid -o value -s TYPE "$HOME_PART" 2>/dev/null || true)
if [ "$FSTYPE" != "ext4" ]; then
    echo "Formatting $HOME_PART as ext4..."
    sudo mkfs.ext4 -L home "$HOME_PART"
fi

# Backup existing /home (it's on root partition)
if [ -d /home ] && [ "$(findmnt -n -o SOURCE /home 2>/dev/null)" != "$HOME_PART" ]; then
    echo "Backing up /home to /home.bak.old..."
    sudo mv /home /home.bak.old
fi

# Create mount point and mount
sudo mkdir -p /home
sudo mount "$HOME_PART" /home

# Restore home contents
if [ -d /home.bak.old ]; then
    echo "Migrating home contents..."
    sudo rsync -a /home.bak.old/ /home/
    sudo chown -R --reference=/home.bak.old /home
fi

# Add to fstab (use PARTUUID for reliability)
PARTUUID=$(blkid -o value -s PARTUUID "$HOME_PART" 2>/dev/null || true)
if [ -n "$PARTUUID" ]; then
    FSTAB_ENTRY="PARTUUID=$PARTUUID  /home  ext4  defaults,noatime  0  2"
else
    FSTAB_ENTRY="$HOME_PART  /home  ext4  defaults,noatime  0  2"
fi

if ! grep -q "/home" /etc/fstab; then
    echo "Adding /home to fstab..."
    echo "$FSTAB_ENTRY" | sudo tee -a /etc/fstab
fi

echo ""
echo "Partition setup complete. /home is now on $HOME_PART"
echo "Reboot to verify: sudo reboot"
