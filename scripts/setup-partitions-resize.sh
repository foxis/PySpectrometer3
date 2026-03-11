#!/bin/bash
# Resize SD card partitions from command line. No GParted needed.
#
# Must run when booted from USB (so SD root is unmounted). If booted from SD,
# script prints instructions to boot from USB and retry.
#
# Prereq: sudo apt install parted e2fsprogs
#
# Layout: root = used + 4GB + 256MB buffer, home = remainder
# Requires: parted, e2fsprogs (resize2fs, e2fsck)

set -e

ROOT_BUFFER_MB=256
ROOT_EXTRA_GB=4
SD_DEV="/dev/mmcblk0"
ROOT_PART="${SD_DEV}p2"

# Check we're NOT booted from SD
ROOT_SRC=$(findmnt -n -o SOURCE / 2>/dev/null | head -1)
if [[ "$ROOT_SRC" == *"mmcblk0"* ]]; then
    echo "You are booted from SD ($ROOT_SRC). Cannot resize mounted root."
    echo ""
    echo "Boot from USB instead:"
    echo "  1. Copy this Raspberry Pi OS to a USB drive"
    echo "  2. Boot the Pi with USB (remove SD or set USB boot priority)"
    echo "  3. SSH in and run: sudo ./scripts/setup-partitions-resize.sh"
    echo ""
    echo "Or run from another Linux (e.g. laptop with SD reader):"
    echo "  Insert SD, run this script as root."
    exit 1
fi

# Ensure SD partitions are unmounted
for part in ${SD_DEV}p1 ${SD_DEV}p2 ${SD_DEV}p3; do
    [ -b "$part" ] || continue
    if findmnt -n "$part" 2>/dev/null; then
        echo "Unmounting $part..."
        sudo umount "$part" 2>/dev/null || true
    fi
done

# Get current sizes (used = total blocks - free blocks)
DUMP=$(sudo dumpe2fs -h "$ROOT_PART" 2>/dev/null)
BLOCK_COUNT=$(echo "$DUMP" | grep "Block count:" | awk '{print $3}')
FREE_BLOCKS=$(echo "$DUMP" | grep "Free blocks:" | awk '{print $3}')
BLOCK_SIZE=$(echo "$DUMP" | grep "Block size:" | awk '{print $3}')
USED_KB=$(((BLOCK_COUNT - FREE_BLOCKS) * BLOCK_SIZE / 1024))
TOTAL_KB=$(sudo blockdev --getsize64 "$ROOT_PART" 2>/dev/null)
TOTAL_KB=$((TOTAL_KB / 1024))

ROOT_SIZE_KB=$((USED_KB + ROOT_EXTRA_GB * 1024 * 1024 + ROOT_BUFFER_MB * 1024))
ROOT_SIZE_MB=$((ROOT_SIZE_KB / 1024))
HOME_SIZE_MB=$(((TOTAL_KB - ROOT_SIZE_KB) / 1024))

echo "=== SD partition resize (root = used + ${ROOT_EXTRA_GB}GB + ${ROOT_BUFFER_MB}MB) ==="
echo "Root used:   $((USED_KB / 1024 / 1024)) GB"
echo "Root target: ${ROOT_SIZE_MB} MB"
echo "Home size:   ${HOME_SIZE_MB} MB"
echo ""
read -p "Resize now? [y/N] " -n 1 -r
echo
[[ $REPLY =~ ^[Yy]$ ]] || exit 0

# Shrink filesystem first (must be unmounted)
echo "Checking filesystem..."
sudo e2fsck -f -y "$ROOT_PART" || true
echo "Shrinking filesystem to ${ROOT_SIZE_MB}M..."
sudo resize2fs "$ROOT_PART" "${ROOT_SIZE_MB}M"

# Shrink partition, create home
echo "Resizing partition..."
sudo parted -s "$SD_DEV" resizepart 2 "${ROOT_SIZE_MB}MB"
echo "Creating home partition..."
sudo parted -s "$SD_DEV" mkpart primary ext4 "${ROOT_SIZE_MB}MB" 100%

# Format home
HOME_PART="${SD_DEV}p3"
echo "Formatting $HOME_PART..."
sudo mkfs.ext4 -L home "$HOME_PART"

# Mount root, add fstab, migrate /home
MNT=$(mktemp -d)
sudo mount "$ROOT_PART" "$MNT"

# Add fstab entry
PARTUUID=$(sudo blkid -o value -s PARTUUID "$HOME_PART" 2>/dev/null || true)
if [ -n "$PARTUUID" ]; then
    FSTAB_ENTRY="PARTUUID=$PARTUUID  /home  ext4  defaults,noatime  0  2"
else
    FSTAB_ENTRY="$HOME_PART  /home  ext4  defaults,noatime  0  2"
fi
if ! grep -q "/home" "$MNT/etc/fstab"; then
    echo "$FSTAB_ENTRY" | sudo tee -a "$MNT/etc/fstab"
fi

# Migrate /home to new partition
sudo mkdir -p "${MNT}/home.new"
sudo mount "$HOME_PART" "${MNT}/home.new"
if [ -d "${MNT}/home" ] && [ "$(ls -A ${MNT}/home 2>/dev/null)" ]; then
    echo "Migrating /home..."
    sudo rsync -a "${MNT}/home/" "${MNT}/home.new/"
fi
sudo umount "${MNT}/home.new"
sudo rmdir "${MNT}/home.new"
sudo mv "${MNT}/home" "${MNT}/home.bak.old" 2>/dev/null || true
sudo mkdir "${MNT}/home"

sudo umount "$MNT"
rmdir "$MNT"

echo ""
echo "Done. Remove USB, boot from SD. /home will mount from new partition."
echo "Old /home backed up to /home.bak.old on root (delete when verified)."
