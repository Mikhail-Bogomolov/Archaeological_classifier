"""Съёмка одного кадра с USB-камеры (OpenCV), как в Новая папка/app.py."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from time import time

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAMERA_PHOTOS_DIR = PROJECT_ROOT / "data" / "photos_from_camera"

# Индекс камеры: в app.py из «Новая папка» использовался 1.
DEFAULT_CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "1"))
WARMUP_FRAMES = 8
MAX_CAPTURE_ATTEMPTS = 25
MIN_FRAME_MEAN = 8.0
JPEG_QUALITY = 92


class CameraCaptureError(RuntimeError):
    pass


def _frame_mean_brightness(frame_bgr) -> float:
    if frame_bgr is None or frame_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def _camera_backends() -> list[int | None]:
    # Как в «Новая папка/app.py» — сначала default backend; DSHOW только запасной.
    if sys.platform == "win32":
        return [None, cv2.CAP_DSHOW]
    return [None]


def _indices_to_try(preferred: int) -> list[int]:
    order = [preferred]
    for idx in (0, 1, 2):
        if idx not in order:
            order.append(idx)
    return order


def _configure_camera(cap: cv2.VideoCapture) -> None:
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)


def _open_camera(camera_index: int) -> cv2.VideoCapture:
    last_error: str | None = None
    for idx in _indices_to_try(camera_index):
        for backend in _camera_backends():
            cap = (
                cv2.VideoCapture(idx, backend)
                if backend is not None
                else cv2.VideoCapture(idx)
            )
            if not cap.isOpened():
                cap.release()
                last_error = f"индекс {idx} не открылся"
                continue
            _configure_camera(cap)
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap
            cap.release()
            last_error = f"индекс {idx}: кадр не прочитан"
    raise CameraCaptureError(
        f"Не удалось открыть камеру ({last_error or 'неизвестная ошибка'}). "
        "Проверьте подключение или задайте CAMERA_INDEX=0."
    )


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
