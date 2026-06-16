"""
Подготовка фото перед классификацией.

classifier_preprocess — обычный путь для сети 1.
cv_preprocess — старый тяжёлый вариант, для сети 2.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms

from app.ml.config import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
    USE_TEXTURE_FEATURES,
)
from app.ml.texture_features import extract_texture_vector

# На фото описи снизу линейка, справа бирка.
KANSK_BOTTOM_CROP_RATIO = 0.14
KANSK_RIGHT_CROP_RATIO = 0.08
# Полный JPEG слишком большой, грузим уменьшенный.
MAX_LOAD_SIDE = 1600


def item_key_from_filename(path: str | Path) -> str:
    """уд85_1-1_а.jpg → 1-1 (один предмет, разные ракурсы)."""
    stem = Path(path).stem.lower()
    if stem.startswith("уд85_"):
        stem = stem[5:]
    parts = stem.rsplit("_", 1)
    return parts[0] if len(parts) == 2 else stem


def crop_kansk_frame(pil: Image.Image) -> Image.Image:
    """Убираем линейку и бирку по краям кадра."""
    w, h = pil.size
    right = max(1, int(w * (1 - KANSK_RIGHT_CROP_RATIO)))
    bottom = max(1, int(h * (1 - KANSK_BOTTOM_CROP_RATIO)))
    if right <= 1 or bottom <= 1:
        return pil
    return pil.crop((0, 0, right, bottom))


def crop_to_artifact(pil: Image.Image, max_side: int = 800) -> Image.Image:
    """Ищем предмет на фото и обрезаем лишний фон."""
    try:
        import cv2
    except ImportError:
        return pil

    w, h = pil.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        sw, sh = int(w * scale), int(h * scale)
        small = pil.resize((sw, sh), Image.LANCZOS)
    else:
        small = pil
        sw, sh = w, h

    arr = np.array(small)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return pil

    img_area = sw * sh
    best = None
    best_area = 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if img_area * 0.005 < area < img_area * 0.8 and area > best_area:
            best_area = area
            best = cnt

    if best is None:
        return pil

    x, y, bw, bh = cv2.boundingRect(best)
    mx, my = int(bw * 0.1), int(bh * 0.1)
    x0 = max(0, x - mx)
    y0 = max(0, y - my)
    x1 = min(sw, x + bw + mx)
    y1 = min(sh, y + bh + my)

    if scale < 1.0:
        inv = 1.0 / scale
        return pil.crop((int(x0 * inv), int(y0 * inv), min(w, int(x1 * inv)), min(h, int(y1 * inv))))

    return Image.fromarray(arr[y0:y1, x0:x1])


def load_classifier_rgb(
    image_bytes: bytes | None = None,
    path: str | Path | None = None,
    max_load_side: int = MAX_LOAD_SIDE,
) -> Image.Image:
    """Поворот, обрезка кадра и выделение предмета."""
    if image_bytes is not None:
        pil = Image.open(io.BytesIO(image_bytes))
    elif path is not None:
        pil = Image.open(path)
    else:
        raise ValueError("Нужен image_bytes или path")

    if pil.format == "JPEG" and max_load_side > 0:
        pil.draft("RGB", (max_load_side, max_load_side))

    pil = ImageOps.exif_transpose(pil)
    if max_load_side > 0 and max(pil.size) > max_load_side:
        pil.thumbnail((max_load_side, max_load_side), Image.LANCZOS)
    pil = pil.convert("RGB")
    pil = crop_kansk_frame(pil)
    pil = crop_to_artifact(pil)
    return pil


def _val_transforms() -> transforms.Compose:
    """Как при проверке на val."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def classifier_preprocess(image_bytes: bytes) -> Tuple[torch.Tensor, Image.Image, dict, torch.Tensor | None]:
    """Фото → тензор для сети 1 + текстура, если включена."""
    pil = load_classifier_rgb(image_bytes=image_bytes)
    texture = None
    if USE_TEXTURE_FEATURES:
        texture = torch.tensor(extract_texture_vector(pil), dtype=torch.float32).unsqueeze(0)
    tensor = _val_transforms()(pil).unsqueeze(0)
    meta = {
        "preprocess": "classifier",
        "frame_cropped": True,
        "artifact_cropped": True,
        "texture_features": USE_TEXTURE_FEATURES,
        "cv_multi_channel_applied": False,
        "mean_brightness": float(np.mean(np.array(pil.convert("L")))),
    }
    return tensor, pil, meta, texture


def _mobilenet_tensor_from_rgb(pil: Image.Image) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return tf(pil).unsqueeze(0)


def five_channel_to_pil_rgb(five_ch: np.ndarray) -> Image.Image:
    """RGB из пятиканального массива (каналы 2–4)."""
    rgb = five_ch[2:5]
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    rgb = np.transpose(rgb, (1, 2, 0))
    return Image.fromarray(rgb, mode="RGB")


def cv_preprocess(image_bytes: bytes) -> Tuple[torch.Tensor, Image.Image, dict]:
    """Старый пайплайн с OpenCV, медленный — для сети 2."""
    meta: dict = {"cv_multi_channel_applied": False, "preprocess": "cv"}
    pil_fallback = Image.open(io.BytesIO(image_bytes))
    pil_fallback = ImageOps.exif_transpose(pil_fallback).convert("RGB")

    try:
        from app.ml.image_processing import process_bytes

        five_ch = process_bytes(image_bytes)
        meta["cv_multi_channel_applied"] = True
        meta["multi_channel_shape"] = list(five_ch.shape)
        pil = five_channel_to_pil_rgb(five_ch)
    except Exception as exc:
        meta["cv_pipeline_error"] = str(exc)
        pil = pil_fallback
        pil = ImageOps.autocontrast(pil, cutoff=1)

    gray = np.array(pil.convert("L"))
    meta["mean_brightness"] = float(np.mean(gray))
    tensor = _mobilenet_tensor_from_rgb(pil)
    return tensor, pil, meta


def five_channel_to_mobilenet_tensor(five_ch: np.ndarray) -> torch.Tensor:
    """Пятиканальный numpy → тензор 224×224."""
    pil = five_channel_to_pil_rgb(five_ch)
    return _mobilenet_tensor_from_rgb(pil)
