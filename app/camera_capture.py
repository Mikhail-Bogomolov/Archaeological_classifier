"""Съёмка одного кадра с USB-камеры (OpenCV)."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from time import time

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAMERA_PHOTOS_DIR = PROJECT_ROOT / "data" / "photos_from_camera"

# Предпочтительный числовой индекс OpenCV (CAMERA_INDEX=N).
# Если задан CAMERA_DEVICE=/dev/videoN, он используется вместо индекса.
DEFAULT_CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "1"))
CAMERA_DEVICE = os.environ.get("CAMERA_DEVICE", "").strip()

# Ключевые слова USB id нашей камеры: lsusb = 2b16:6689 SunplusIT Inc OPEN AICAM
_AICAM_HINTS = ("2b16", "6689", "aicam", "sunplusit", "open aicam", "sunplus")

_FALLBACK_INDICES = (1, 4, 0, 2, 3, 5, 6, 7, 8, 9, 10)
WARMUP_FRAMES = 8
MAX_CAPTURE_ATTEMPTS = 25
MIN_FRAME_MEAN = 8.0
JPEG_QUALITY = 92


class CameraCaptureError(RuntimeError):
    pass


# ─── вспомогательные функции ───────────────────────────────────────────────────

def _frame_mean_brightness(frame_bgr) -> float:
    if frame_bgr is None or frame_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def _read_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").strip().lower()
    except OSError:
        return ""


def _run_cmd(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL,
                                       timeout=3).decode("utf-8", errors="ignore").lower()
    except Exception:
        return ""


# ─── поиск правильного /dev/videoN ─────────────────────────────────────────────

def _sysfs_blob_for_video(name: str) -> str:
    """Всё из sysfs для /dev/videoN в одну строку."""
    base = Path("/sys/class/video4linux") / name
    # Поднимаемся вдоль device/, пока не найдём idVendor или uevent с PRODUCT=.
    parts = [_read_file(base / "name")]
    dev = base / "device"
    for _ in range(5):
        if not dev.is_dir():
            break
        for attr in ("idVendor", "idProduct", "uevent", "product"):
            parts.append(_read_file(dev / attr))
        dev = dev / ".." / ".." / "device"
        dev = dev.resolve()
    return " ".join(parts)


def _video_matches_aicam(node: Path) -> bool:
    blob = _sysfs_blob_for_video(node.name)
    return any(h in blob for h in _AICAM_HINTS)


def _v4l2ctl_find_aicam() -> list[int]:
    """v4l2-ctl --list-devices — надёжнее sysfs."""
    out = _run_cmd(["v4l2-ctl", "--list-devices"])
    if not out:
        return []
    indices: list[int] = []
    is_our = False
    for line in out.splitlines():
        stripped = line.strip()
        if any(h in stripped for h in _AICAM_HINTS):
            is_our = True
        elif not stripped.startswith("/dev/video") and stripped:
            is_our = False
        if is_our:
            m = re.search(r"/dev/video(\d+)", stripped)
            if m:
                idx = int(m.group(1))
                if idx not in indices:
                    indices.append(idx)
    return sorted(indices)


def _sysfs_find_aicam() -> list[int]:
    root = Path("/sys/class/video4linux")
    if not root.is_dir():
        return []
    hits: list[int] = []
    for entry in sorted(root.iterdir()):
        if not entry.name.startswith("video"):
            continue
        dev = Path("/dev") / entry.name
        if not dev.exists():
            continue
        if _video_matches_aicam(dev):
            m = re.search(r"video(\d+)$", entry.name)
            if m:
                idx = int(m.group(1))
                if idx not in hits:
                    hits.append(idx)
    return sorted(hits)


def _all_linux_video_indices() -> list[int]:
    indices: list[int] = []
    for p in sorted(Path("/dev").glob("video*")):
        m = re.search(r"video(\d+)$", p.name)
        if m:
            n = int(m.group(1))
            if n not in indices:
                indices.append(n)
    return indices


def _build_index_order(preferred: int) -> list[int]:
    """Список числовых индексов для перебора: сначала AICAM, потом остальные."""
    known_aicam = _v4l2ctl_find_aicam() or _sysfs_find_aicam()

    order: list[int] = []

    def _add(n: int) -> None:
        if n not in order:
            order.append(n)

    # Сначала явный preferred
    _add(preferred)
    # Потом обнаруженные по USB id
    for n in known_aicam:
        _add(n)
    # Потом фиксированные fallback
    for n in _FALLBACK_INDICES:
        _add(n)
    # Потом все /dev/video* что есть
    for n in _all_linux_video_indices():
        _add(n)

    return order


# ─── открытие камеры ───────────────────────────────────────────────────────────

def _camera_backends() -> list[int | None]:
    if sys.platform == "win32":
        return [None, cv2.CAP_DSHOW]
    # На Linux V4L2 явно, потом default.
    backends: list[int | None] = []
    if hasattr(cv2, "CAP_V4L2"):
        backends.append(cv2.CAP_V4L2)
    backends.append(None)
    return backends


def _configure_camera(cap: cv2.VideoCapture) -> None:
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)


def _try_open_index(idx: int) -> cv2.VideoCapture | None:
    for backend in _camera_backends():
        cap = (
            cv2.VideoCapture(idx, backend)
            if backend is not None
            else cv2.VideoCapture(idx)
        )
        if not cap.isOpened():
            cap.release()
            continue
        _configure_camera(cap)
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            return cap
        cap.release()
    return None


def _open_camera(preferred: int) -> cv2.VideoCapture:
    # Если задан явный путь — пробуем только его.
    if CAMERA_DEVICE:
        cap = _try_open_index(CAMERA_DEVICE)  # type: ignore[arg-type]
        if cap is not None:
            return cap
        raise CameraCaptureError(
            f"CAMERA_DEVICE={CAMERA_DEVICE!r}: не удалось открыть камеру."
        )

    tried: list[str] = []

    if sys.platform.startswith("linux"):
        indices = _build_index_order(preferred)
    else:
        indices = list(dict.fromkeys([preferred] + list(_FALLBACK_INDICES)))

    for idx in indices:
        cap = _try_open_index(idx)
        if cap is not None:
            return cap
        tried.append(str(idx))

    raise CameraCaptureError(
        f"Не удалось открыть камеру. "
        f"Пробовали индексы: {', '.join(tried) or '—'}. "
        "Выполните на Orange Pi: ls /dev/video* && v4l2-ctl --list-devices "
        "и задайте CAMERA_INDEX=N или CAMERA_DEVICE=/dev/videoN."
    )


# ─── захват кадра ──────────────────────────────────────────────────────────────

def _capture_frame_bgr(cap: cv2.VideoCapture):
    for _ in range(WARMUP_FRAMES):
        cap.read()

    best_frame = None
    best_mean = 0.0
    for _ in range(MAX_CAPTURE_ATTEMPTS):
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        mean = _frame_mean_brightness(frame)
        if mean >= MIN_FRAME_MEAN:
            return frame
        if mean > best_mean:
            best_mean = mean
            best_frame = frame

    if best_frame is not None and best_mean > 0:
        return best_frame

    raise CameraCaptureError(
        "Камера вернула только чёрные кадры. Подождите пару секунд и попробуйте снова."
    )


def capture_jpeg_bytes(
    camera_index: int | None = None,
    *,
    save: bool = True,
) -> tuple[bytes, Path | None]:
    """
    Захват кадра → JPEG bytes.
    При save=True файл сохраняется в data/photos_from_camera/
    с суффиксом _field_ (полевая съёмка для препроцессинга).
    """
    idx = DEFAULT_CAMERA_INDEX if camera_index is None else camera_index
    cap = _open_camera(idx)
    try:
        frame_bgr = _capture_frame_bgr(cap)
    finally:
        cap.release()

    stamp = int(time() * 1000)
    filename = f"camera_field_{stamp}.jpg"
    saved_path: Path | None = None

    ok_enc, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok_enc:
        raise CameraCaptureError("Не удалось закодировать снимок в JPEG.")
    jpeg_bytes = buf.tobytes()

    if save:
        saved_path = CAMERA_PHOTOS_DIR / filename
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        saved_path.write_bytes(jpeg_bytes)

    return jpeg_bytes, saved_path
