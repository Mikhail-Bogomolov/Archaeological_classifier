"""Поиск и автомонтирование USB-флешки (Orange Pi / Linux)."""

from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path

logger = logging.getLogger("usb_export")

_USB_FS = frozenset({
    "vfat", "exfat", "ntfs", "ntfs-3g", "ntfs3", "fuseblk",
    "msdos", "fat", "fat32", "ext4", "ext3", "ext2",
    "btrfs", "xfs",
})

_MOUNT_PREFIXES = ("/media/", "/run/media/", "/mnt/")
_MOUNT_POINT = Path("/mnt/usb")


def _run(cmd: list[str], timeout: int = 10) -> subprocess.CompletedProcess:
    """Вспомогательный запуск subprocess с логированием."""
    logger.debug("Run: %s", " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _list_mounted() -> list[Path]:
    """Уже смонтированные подходящие точки."""
    candidates: list[Path] = []
    try:
        for line in Path("/proc/mounts").read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            mp, fst = parts[1], parts[2]
            if fst not in _USB_FS:
                continue
            if not any(mp.startswith(p) for p in _MOUNT_PREFIXES):
                continue
            if mp in ("/", "/boot", "/boot/firmware"):
                continue
            p = Path(mp)
            if p.is_dir() and os.access(p, os.W_OK):
                candidates.append(p)
    except OSError:
        pass

    if not candidates:
        for pattern in ("/media/*/*", "/run/media/*/*", "/mnt/usb*", "/mnt/USB*"):
            for p in sorted(Path("/").glob(pattern.lstrip("/"))):
                if p.is_dir() and os.access(p, os.W_OK):
                    candidates.append(p)

    candidates.sort(key=lambda p: (len(p.parts), str(p)))
    return candidates


def _find_unmounted_usb() -> list[tuple[str, str]]:
    """Вернуть [(device_name, fstype)] для не смонтированных USB-накопителей."""
    result: list[tuple[str, str]] = []

    # Пробуем с TRAN (современный lsblk)
    try:
        out = subprocess.check_output(
            ["lsblk", "-o", "NAME,TYPE,FSTYPE,MOUNTPOINT,TRAN", "-P", "-n"],
            stderr=subprocess.DEVNULL, timeout=5,
        ).decode("utf-8", errors="ignore")
        for line in out.strip().splitlines():
            fields = dict(re.findall(r'(\w+)="([^"]*)"', line))
            name, blktype, fstype, mp, tran = (
                fields.get("NAME", ""),
                fields.get("TYPE", ""),
                fields.get("FSTYPE", ""),
                fields.get("MOUNTPOINT", ""),
                fields.get("TRAN", ""),
            )
            if mp or blktype not in ("part", "disk"):
                continue
            if name.startswith(("mmcblk", "nvme", "loop", "zram")):
                continue
            if tran and tran != "usb":
                continue
            result.append((name, fstype))
    except Exception:
        pass

    # Fallback: старый lsblk без TRAN
    if not result:
        try:
            out = subprocess.check_output(
                ["lsblk", "-o", "NAME,TYPE,FSTYPE,MOUNTPOINT", "-P", "-n"],
                stderr=subprocess.DEVNULL, timeout=5,
            ).decode("utf-8", errors="ignore")
            for line in out.strip().splitlines():
                fields = dict(re.findall(r'(\w+)="([^"]*)"', line))
                name, blktype, fstype, mp = (
                    fields.get("NAME", ""),
                    fields.get("TYPE", ""),
                    fields.get("FSTYPE", ""),
                    fields.get("MOUNTPOINT", ""),
                )
                if mp or blktype not in ("part", "disk"):
                    continue
                if name.startswith(("mmcblk", "nvme", "loop", "zram")):
                    continue
                result.append((name, fstype))
        except Exception:
            pass

    return result


def _guess_fstype(device: str) -> str | None:
    """Определить ФС через blkid, если lsblk не смог."""
    try:
        out = subprocess.check_output(
            ["blkid", "-s", "TYPE", "-o", "value", f"/dev/{device}"],
            stderr=subprocess.DEVNULL, timeout=5,
        ).decode().strip()
        return out or None
    except Exception:
        return None


def _what_is_mounted_at(mount_point: Path) -> str | None:
    """Вернуть устройство, смонтированное в mount_point, или None."""
    try:
        for line in Path("/proc/mounts").read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1] == str(mount_point):
                return parts[0]
    except OSError:
        pass
    return None


def _try_mount(device: str, fstype: str) -> Path | None:
    """Смонтировать /dev/{device} в /mnt/usb через sudo."""
    try:
        _MOUNT_POINT.mkdir(parents=True, exist_ok=True)

        # Проверим, не занята ли точка другой флешкой
        mounted_dev = _what_is_mounted_at(_MOUNT_POINT)
        if mounted_dev:
            if mounted_dev == f"/dev/{device}":
                logger.info("Устройство %s уже смонтировано в %s", device, _MOUNT_POINT)
                if os.access(_MOUNT_POINT, os.W_OK):
                    return _MOUNT_POINT
                return None
            logger.warning("Точка %s занята %s, размонтируем", _MOUNT_POINT, mounted_dev)
            _run(["sudo", "-n", "umount", str(_MOUNT_POINT)])

        cmd = ["sudo", "-n", "mount", "-o", "umask=000,uid=1000,gid=1000"]
        if fstype:
            cmd.extend(["-t", fstype])
        cmd.extend([f"/dev/{device}", str(_MOUNT_POINT)])

        res = _run(cmd, timeout=15)
        if res.returncode != 0:
            logger.warning("sudo mount ошибка: %s", res.stderr.strip())
            return None

        if _MOUNT_POINT.is_dir() and os.access(_MOUNT_POINT, os.W_OK):
            logger.info("Смонтировано %s в %s", device, _MOUNT_POINT)
            return _MOUNT_POINT
    except Exception as e:
        logger.warning("Mount исключение: %s", e)
    return None


def find_usb_mount() -> Path | None:
    """Каталог флешки или None."""
    # 1. Переменная окружения
    override = os.environ.get("USB_EXPORT_MOUNT", "").strip()
    if override:
        p = Path(override)
        if p.is_dir() and os.access(p, os.W_OK):
            return p
        logger.warning("USB_EXPORT_MOUNT=%s недоступен", override)

    # 2. Уже смонтированные
    mounted = _list_mounted()
    if mounted:
        logger.info("Найдена смонтированная флешка: %s", mounted[-1])
        return mounted[-1]

    # 3. Автомонтирование
    logger.info("Ищем флешку для монтирования...")
    import time
    devices: list[tuple[str, str]] = []
    for attempt in range(5):
        try:
            _run(["udevadm", "settle", "--timeout=2"], timeout=3)
        except Exception:
            pass
        devices = _find_unmounted_usb()
        if devices:
            break
        time.sleep(0.5)
    if not devices:
        logger.error("Не найдено не смонтированных USB-накопителей")
        return None

    for dev, fstype in devices:
        if not fstype:
            fstype = _guess_fstype(dev) or ""
        logger.info("Пробуем смонтировать %s (fs=%s)...", dev, fstype or "auto")
        path = _try_mount(dev, fstype)
        if path:
            return path

    logger.error("Не удалось смонтировать ни одно устройство")
    return None


def safe_export_filename(suffix: str = "") -> str:
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"objects_export_{stamp}"
    if suffix:
        base += suffix
    return re.sub(r"[^\w.\-]+", "_", base) + ".zip"
