"""Поиск USB-флешки (Orange Pi / Linux)."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

_USB_FS = frozenset({
    "vfat",
    "exfat",
    "ntfs",
    "ntfs-3g",
    "fuseblk",
    "msdos",
    "fat",
    "fat32",
    "ext4",
})

# Куда обычно монтируется флешка.
_MOUNT_PREFIXES = (
    "/media/",
    "/run/media/",
    "/mnt/",
)


def _try_automount_usb() -> Path | None:
    """Если флешка вставлена, но не смонтирована — смонтировать через udisksctl."""
    try:
        out = subprocess.check_output(
            ["lsblk", "-o", "NAME,TYPE,FSTYPE,MOUNTPOINT", "-P"],
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).decode()
    except Exception:
        return None

    for line in out.splitlines():
        fields = dict(re.findall(r'(\w+)="([^"]*)"', line))
        if fields.get("TYPE") != "part":
            continue
        if fields.get("MOUNTPOINT"):
            continue
        name = fields.get("NAME")
        if not name or name.startswith("mmcblk"):
            continue
        if not fields.get("FSTYPE"):
            continue
        try:
            result = subprocess.run(
                ["udisksctl", "mount", "-b", f"/dev/{name}"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            m = re.search(r"at (/[\S]+)", result.stdout or "")
            if not m:
                continue
            path = Path(m.group(1).rstrip("."))
            if path.is_dir() and os.access(path, os.W_OK):
                return path
        except Exception:
            continue
    return None


def find_usb_mount() -> Path | None:
    """Каталог флешки или None."""
    override = os.environ.get("USB_EXPORT_MOUNT", "").strip()
    if override:
        path = Path(override)
        if path.is_dir() and os.access(path, os.W_OK):
            return path

    candidates: list[Path] = []
    try:
        mounts_text = Path("/proc/mounts").read_text(encoding="utf-8", errors="ignore")
    except OSError:
        mounts_text = ""

    for line in mounts_text.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        mount_point, fs_type = parts[1], parts[2]
        if fs_type not in _USB_FS:
            continue
        if not any(mount_point.startswith(prefix) for prefix in _MOUNT_PREFIXES):
            continue
        if mount_point in ("/", "/boot", "/boot/firmware"):
            continue
        path = Path(mount_point)
        if path.is_dir() and os.access(path, os.W_OK):
            candidates.append(path)

    if not candidates:
        # Нет /proc/mounts — ищем по путям.
        for pattern in ("/media/*/*", "/run/media/*/*", "/mnt/usb*", "/mnt/USB*"):
            for path in sorted(Path("/").glob(pattern.lstrip("/"))):
                if path.is_dir() and os.access(path, os.W_OK):
                    candidates.append(path)

    if not candidates:
        auto = _try_automount_usb()
        if auto is not None:
            return auto
        return None

    # Берём самый длинный путь (часто /media/user/LABEL).
    candidates.sort(key=lambda p: (len(p.parts), str(p)))
    return candidates[-1]


def safe_export_filename(suffix: str = "") -> str:
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"objects_export_{stamp}"
    if suffix:
        base += suffix
    return re.sub(r"[^\w.\-]+", "_", base) + ".zip"
