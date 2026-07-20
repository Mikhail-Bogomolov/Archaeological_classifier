"""Поиск смонтированной USB-флешки (Orange Pi / Linux)."""

from __future__ import annotations

import os
import re
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
})

# Типичные точки монтирования на Armbian / Debian (один USB-A).
_MOUNT_PREFIXES = (
    "/media/",
    "/run/media/",
    "/mnt/",
)


def find_usb_mount() -> Path | None:
    """Возвращает каталог флешки или None, если не найдена."""
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
        # Запасной обход известных путей (если /proc/mounts недоступен).
        for pattern in ("/media/*/*", "/run/media/*/*", "/mnt/usb*", "/mnt/USB*"):
            for path in sorted(Path("/").glob(pattern.lstrip("/"))):
                if path.is_dir() and os.access(path, os.W_OK):
                    candidates.append(path)

    if not candidates:
        return None

    # Одна флешка — берём самый «глубокий» путь (обычно label внутри /media/user/).
    candidates.sort(key=lambda p: (len(p.parts), str(p)))
    return candidates[-1]


def safe_export_filename(suffix: str = "") -> str:
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"objects_export_{stamp}"
    if suffix:
        base += suffix
    return re.sub(r"[^\w.\-]+", "_", base) + ".zip"
