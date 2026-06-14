"""
Сектор обработки фото.

prepare_img() → тензор (5, TARGET_SHAPE, TARGET_SHAPE):
  [shape_mask, edges, R, G, B].
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from app.ml.config import CV_TARGET_SHAPE, TEST_IMAGE_PATHS

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    import imutils
except ImportError:  # pragma: no cover
    imutils = None


def rgb2gray(rgb: np.ndarray) -> np.ndarray:
    """Преобразование RGB в оттенки серого."""
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray.reshape([len(r), len(r[0]), 1])


def _resize_keep_aspect(img: np.ndarray, target_shape: int) -> np.ndarray:
    height, width = img.shape[:2]
    if imutils is not None:
        if height > width:
            return imutils.resize(img, height=target_shape)
        return imutils.resize(img, width=target_shape)
    scale = target_shape / max(height, width)
    new_h, new_w = int(height * scale), int(width * scale)
    if cv2 is not None:
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    from PIL import Image

    pil = Image.fromarray(img.astype(np.uint8) if img.dtype != np.uint8 else img)
    pil = pil.resize((new_w, new_h), Image.LANCZOS)
    return np.array(pil)


def prepare_img(
    raw_clear_img: np.ndarray,
    white_const: int = 240,
    black_const: int = 25,
    target_shape: int = CV_TARGET_SHAPE,
) -> np.ndarray:
    """
    Очистка фона → контуры → Canny → 5 каналов target_shape×target_shape.
    """
    if cv2 is None:
        raise ImportError(
            "Для обработки фото нужен opencv-python-headless."
        )

    raw_clear_img = raw_clear_img.copy()
    height, width = raw_clear_img.shape[:2]

    for i in range(height):
        for j in range(width):
            if (
                raw_clear_img[i][j][0] < black_const
                and raw_clear_img[i][j][1] < black_const
                and raw_clear_img[i][j][2] < black_const
            ):
                raw_clear_img[i][j] = [255, 255, 255]

    shape_img = raw_clear_img.copy()
    for i in range(height):
        for j in range(width):
            if (
                shape_img[i][j][0] > white_const
                and shape_img[i][j][1] > white_const
                and shape_img[i][j][2] > white_const
            ):
                shape_img[i][j] = [255, 255, 255]
            else:
                shape_img[i][j] = [0, 0, 0]

    gsc_img = rgb2gray(shape_img)
    contours, hierarchy = cv2.findContours(
        image=gsc_img.astype(np.uint8),
        mode=cv2.RETR_TREE,
        method=cv2.CHAIN_APPROX_SIMPLE,
    )
    mask = np.ones_like(gsc_img, dtype=np.uint8)
    if hierarchy is not None:
        for i, cnt in enumerate(contours):
            if hierarchy[0][i][2] == -1 and cv2.contourArea(cnt) < 1000:
                cv2.drawContours(mask, [cnt], 0, (0), -1)

    for i in range(height):
        for j in range(width):
            if 0 in mask[i][j]:
                raw_clear_img[i, j] = [255, 255, 255]
                shape_img[i, j] = [255, 255, 255]

    fin_shape_img = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            fin_shape_img[i, j] = 0 if 0 in shape_img[i, j] else 1

    clear_img = raw_clear_img
    raw_r_img = clear_img[:, :, 0]
    raw_g_img = clear_img[:, :, 1]
    raw_b_img = clear_img[:, :, 2]

    edges_r = cv2.Canny(raw_r_img, 20, 120)
    edges_g = cv2.Canny(raw_g_img, 20, 120)
    edges_b = cv2.Canny(raw_b_img, 20, 120)
    edges = edges_r + edges_g + edges_b
    edges_copy = np.zeros_like(edges)
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            if edges[i][j] > 0:
                edges[i][j] = 1
                edges_copy = cv2.circle(edges_copy, (j, i), 3, (1), -1)
            else:
                edges[i][j] = 0
    edges = edges_copy

    fin_data = []
    for img_ in [fin_shape_img, edges, raw_r_img / 255, raw_g_img / 255, raw_b_img / 255]:
        resized_img = _resize_keep_aspect(img_, target_shape)
        y_size, x_size = resized_img.shape[0], resized_img.shape[1]
        res_img = np.pad(
            resized_img,
            ((0, target_shape - y_size), (0, target_shape - x_size)),
            mode="constant",
        )
        fin_data.append(np.array(res_img).reshape((target_shape, target_shape)))

    return np.array(fin_data, dtype=np.float32)


def load_image_bgr(path: str | Path) -> np.ndarray:
    if cv2 is None:
        raise ImportError("opencv-python-headless required")
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {path}")
    return img


def process_file(path: str | Path, target_shape: int = CV_TARGET_SHAPE) -> np.ndarray:
    """cv2.imread → prepare_img"""
    return prepare_img(load_image_bgr(path), target_shape=target_shape)


def process_bytes(image_bytes: bytes, target_shape: int = CV_TARGET_SHAPE) -> np.ndarray:
    if cv2 is None:
        raise ImportError("opencv-python-headless required")
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Не удалось декодировать изображение")
    return prepare_img(bgr, target_shape=target_shape)


def discover_test_images(extra_paths: Iterable[str] | None = None) -> list[Path]:
    """Находит test_img_02.jpg и test_img_10.jpg"""
    found: list[Path] = []
    seen: set[str] = set()
    candidates = list(TEST_IMAGE_PATHS)
    if extra_paths:
        candidates.extend(extra_paths)

    for p in candidates:
        path = Path(p)
        key = str(path.resolve()) if path.exists() else str(path)
        if path.is_file() and key not in seen:
            seen.add(key)
            found.append(path)

    # Уникальные по имени файла (корень приоритетнее)
    by_name: dict[str, Path] = {}
    for p in found:
        by_name.setdefault(p.name, p)
    return [by_name["test_img_02.jpg"], by_name["test_img_10.jpg"]] if by_name else []
