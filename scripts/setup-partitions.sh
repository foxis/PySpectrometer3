#!/bin/bash
# Create separate writable /home partition. Root and boot will be read-only.
# Prerequisite: You must have unallocated space or shrink root first.
#
# Option A - Fresh install: Use Raspberry Pi Imager "Customize" to create 3 partitions
#   (boot, root, home) before first boot.
#
# Option B - Existing install: Boot from GParted (or another Linux live USB),
#   shrink root partition, create new ext4 partition for /home, reboot.
#   Then run this script.
#
# This script: formats the home partition (if new), adds fstab entry, migrates /home.

set -e

# Detect home partition - typically mmcblk0p3 if you created it after root (p2)
HOME_PART=""
for p in /dev/mmcblk0p3 /dev/sda1; do
    if [ -b "$p" ]; then
        # Check if already mounted at /home
        if findmnt -n "$p" 2>/dev/null | grep -q /home; then
            echo "/home is already a separate partition ($p). Nothing to do."
            exit 0
        fi
        # Check if not mounted (candidate for home)
        if ! findmnt -n "$p" 2>/dev/null; then
            HOME_PART="$p"
            break
        fi
    fi
done

if [ -z "$HOME_PART" ]; then
    echo "No unallocated home partition found."
    echo ""
    echo "To create a separate /home partition:"
    echo "  1. Boot from GParted (or Raspberry Pi Imager with GParted)"
    echo "  2. Shrink the root partition (ext4) to free space at the end"
    echo "  3. Create new ext4 partition in the freed space (e.g. /dev/mmcblk0p3)"
    echo "  4. Reboot into Raspberry Pi OS"
    echo "  5. Run this script again"
    echo ""
    echo "Or use Raspberry Pi Imager 'Customize' to set 3-partition layout before first boot."
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
