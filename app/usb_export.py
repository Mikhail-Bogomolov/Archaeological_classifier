"""Поиск USB-флешки (Orange Pi / Linux / Windows / macOS)."""

from __future__ import annotations

import os
import platform
import re
import subprocess
from pathlib import Path

_USB_FS = frozenset({
    "vfat", "exfat", "ntfs", "ntfs-3g", "fuseblk",
    "msdos", "fat", "fat32", "ext4", "ext3", "ext2",
    "btrfs", "xfs", "drvfs", "9p", "hfs", "apfs",
})

_MOUNT_PREFIXES = (
    "/media/",
    "/run/media/",
    "/mnt/",
)


def _windows_removable_drives() -> list[Path]:
    """Ищет съёмные диски на Windows."""
    candidates: list[Path] = []
    try:
        out = subprocess.check_output(
            [
                "powershell",
                "-Command",
                "Get-CimInstance Win32_LogicalDisk | Where-Object {$_.DriveType -eq 2} | ForEach-Object {$_.DeviceID}",
            ],
            text=True,
            timeout=10,
        )
        for line in out.strip().splitlines():
            d = line.strip().rstrip("\\") + "\\"
            p = Path(d)
            if p.exists() and os.access(p, os.W_OK):
                candidates.append(p)
    except Exception:
        pass
    return candidates


def find_usb_mount() -> Path | None:
    """Каталог флешки или None."""
    # 1. Переменная окружения — приоритет №1
    override = os.environ.get("USB_EXPORT_MOUNT", "").strip()
    if override:
        path = Path(override)
        if path.is_dir() and os.access(path, os.W_OK):
            return path
        print(f"[USB] WARNING: USB_EXPORT_MOUNT={override} недоступен для записи")

    # 2. Windows
    if platform.system() == "Windows":
        candidates = _windows_removable_drives()
        if candidates:
            return candidates[0]
        # Fallback: любой несистемный диск
        for letter in "DEFGHIJKLMNOPQRSTUVWXYZ":
            p = Path(f"{letter}:\\")
            if p.exists() and os.access(p, os.W_OK):
                if not (p / "Windows").exists():
                    return p
        return None

    # 3. Linux / macOS — /proc/mounts
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
        # Fallback на glob, если /proc/mounts недоступен
        for pattern in ("/media/*/*", "/run/media/*/*", "/mnt/usb*", "/mnt/USB*"):
            for path in sorted(Path("/").glob(pattern.lstrip("/"))):
                if path.is_dir() and os.access(path, os.W_OK):
                    candidates.append(path)

    if not candidates:
        return None

    # Берём самый длинный путь (обычно /media/user/LABEL)
    candidates.sort(key=lambda p: (len(p.parts), str(p)))
    return candidates[-1]


def safe_export_filename(suffix: str = "") -> str:
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"objects_export_{stamp}"
    if suffix:
        base += suffix
    return re.sub(r"[^\w.\-]+", "_", base) + ".zip"
