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


# Полевая/установочная съёмка 16:9; архив Канск ~3:2 (1.5).
FIELD_ASPECT_MIN = 1.62


def _is_installation_shot(path: str | Path | None, pil: Image.Image) -> bool:
    """Полевая/установочная съёмка: без обрезки линейки и без OpenCV-выделения объекта."""
    if path is not None and "_field_" in Path(path).name.lower():
        return True
    w, h = pil.size
    if h > 0 and (w / h) >= FIELD_ASPECT_MIN:
        return True
    # Тёмный фон установки — Otsu ловит блики, не предмет.
    if float(np.mean(np.array(pil.convert("L")))) < 48:
        return True
    return False


def _should_skip_kansk_frame_crop(path: str | Path | None, pil: Image.Image) -> bool:
    """Не обрезаем линейку/бирку на полевых и установочных кадрах."""
    return _is_installation_shot(path, pil)


def crop_kansk_frame(pil: Image.Image) -> Image.Image:
    """Убираем линейку и бирку по краям кадра (только архив Канск)."""
    w, h = pil.size
    right = max(1, int(w * (1 - KANSK_RIGHT_CROP_RATIO)))
    bottom = max(1, int(h * (1 - KANSK_BOTTOM_CROP_RATIO)))
    if right <= 1 or bottom <= 1:
        return pil
    return pil.crop((0, 0, right, bottom))


def _artifact_bbox_ok(sw: int, sh: int, bw: int, bh: int, *, ui: bool = False) -> bool:
    """Отсекаем полоски от бликов и прочий мусор."""
    min_side = 0.10 if ui else 0.15
    min_area = 0.04 if ui else 0.08
    max_aspect = 2.5 if ui else 3.5
    min_aspect = 0.35 if ui else 0.28
    if bw < sw * min_side or bh < sh * min_side:
        return False
    aspect = bw / max(bh, 1)
    if aspect > max_aspect or aspect < min_aspect:
        return False
    if bw * bh < sw * sh * min_area:
        return False
    return True


def _artifact_contour_score(cnt, sw: int, sh: int, *, ui: bool = False) -> float:
    import cv2

    img_area = sw * sh
    area = cv2.contourArea(cnt)
    min_area = 0.004 if ui else 0.008
    if not (img_area * min_area < area < img_area * 0.80):
        return -1.0
    _x, _y, bw, bh = cv2.boundingRect(cnt)
    if not _artifact_bbox_ok(sw, sh, bw, bh, ui=ui):
        return -1.0
    cx = _x + bw / 2
    cy = _y + bh / 2
    center_dist = ((cx - sw / 2) / sw) ** 2 + ((cy - sh / 2) / sh) ** 2
    return area * (1.0 - 0.75 * min(center_dist, 1.0))


def _crop_from_mask(
    pil: Image.Image,
    arr: np.ndarray,
    mask: np.ndarray,
    scale: float,
    w: int,
    h: int,
    sw: int,
    sh: int,
    *,
    ui: bool = False,
) -> Image.Image:
    import cv2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return pil

    best = None
    best_score = -1.0
    for cnt in contours:
        score = _artifact_contour_score(cnt, sw, sh, ui=ui)
        if score > best_score:
            best_score = score
            best = cnt

    if best is None:
        return pil

    x, y, bw, bh = cv2.boundingRect(best)
    pad = 0.10 if ui else 0.08
    mx, my = int(bw * pad), int(bh * pad)
    x0 = max(0, x - mx)
    y0 = max(0, y - my)
    x1 = min(sw, x + bw + mx)
    y1 = min(sh, y + bh + my)

    if not _artifact_bbox_ok(sw, sh, x1 - x0, y1 - y0, ui=ui):
        return pil

    min_keep = 0.08 if ui else 0.12
    if (x1 - x0) * (y1 - y0) < sw * sh * min_keep:
        return pil

    if scale < 1.0:
        inv = 1.0 / scale
        out = pil.crop((int(x0 * inv), int(y0 * inv), min(w, int(x1 * inv)), min(h, int(y1 * inv))))
    else:
        out = Image.fromarray(arr[y0:y1, x0:x1])

    ow, oh = pil.size
    if out.size[0] * out.size[1] < ow * oh * min_keep:
        return pil
    return out


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
    return _crop_from_mask(pil, arr, mask, scale, w, h, sw, sh, ui=False)


def crop_to_artifact_desk(pil: Image.Image, max_side: int = 800) -> Image.Image:
    """Выделение на светлом столе (фото описи) — только для превью UI."""
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
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        8,
    )
    return _crop_from_mask(pil, arr, mask, scale, w, h, sw, sh, ui=True)


def center_crop_focus(
    pil: Image.Image,
    width_ratio: float = 0.62,
    height_ratio: float = 0.68,
) -> Image.Image:
    """Центральная область кадра — для установки, где предмет по центру."""
    w, h = pil.size
    cw = max(1, min(w, int(w * width_ratio)))
    ch = max(1, min(h, int(h * height_ratio)))
    left = (w - cw) // 2
    top = (h - ch) // 2
    return pil.crop((left, top, left + cw, top + ch))


def _pad_preview_aspect(
    pil: Image.Image,
    target: float = 4 / 3,
    pad_rgb: tuple[int, int, int] = (229, 231, 235),
) -> Image.Image:
    """Вписываем объект в 4:3 с полями — ничего не отрезаем."""
    w, h = pil.size
    if h <= 0:
        return pil
    current = w / h
    if abs(current - target) < 0.03:
        return pil
    if current > target:
        new_h = max(h, int(w / target))
        canvas = Image.new("RGB", (w, new_h), pad_rgb)
        canvas.paste(pil, (0, (new_h - h) // 2))
        return canvas
    new_w = max(w, int(h * target))
    canvas = Image.new("RGB", (new_w, h), pad_rgb)
    canvas.paste(pil, ((new_w - w) // 2, 0))
    return canvas


def _ui_preview_crop_ratios(
    pil: Image.Image,
    installation: bool,
) -> tuple[float, float]:
    """Лёгкий зум по центру — только для установки."""
    w, h = pil.size
    aspect = w / max(h, 1)
    mean_b = float(np.mean(np.array(pil.convert("L"))))

    if installation:
        if mean_b < 55:
            return 0.52, 0.58
        if aspect >= FIELD_ASPECT_MIN:
            return 0.58, 0.64
        return 0.62, 0.68
    return 1.0, 1.0


# Доп. обрезка бирки справа в превью UI (после crop_kansk_frame).
UI_PREVIEW_RIGHT_TRIM = 0.10
UI_PREVIEW_LEFT_TRIM = 0.02
UI_PREVIEW_TOP_TRIM = 0.05
UI_PREVIEW_BOTTOM_TRIM = 0.12


def crop_kansk_preview_trim(pil: Image.Image) -> Image.Image:
    """Убираем зону бирки справа — предмет обычно левее центра."""
    w, h = pil.size
    left = int(w * UI_PREVIEW_LEFT_TRIM)
    right = int(w * (1 - UI_PREVIEW_RIGHT_TRIM))
    top = int(h * UI_PREVIEW_TOP_TRIM)
    bottom = int(h * (1 - UI_PREVIEW_BOTTOM_TRIM))
    if right <= left + 8 or bottom <= top + 8:
        return pil
    return pil.crop((left, top, right, bottom))


def _ui_kansk_preview(framed: Image.Image) -> Image.Image:
    """Геометрическое превью без OpenCV — бирка и линейка не попадают в кадр."""
    trimmed = crop_kansk_preview_trim(framed)
    w, h = trimmed.size
    aspect = w / max(h, 1)
    if aspect > 1.25:
        return center_crop_focus(trimmed, 0.80, 0.62)
    if aspect < 0.85:
        return center_crop_focus(trimmed, 0.92, 0.94)
    return center_crop_focus(trimmed, 0.88, 0.78)


def load_ui_preview_rgb(image_bytes: bytes, max_load_side: int = MAX_LOAD_SIDE) -> Image.Image:
    """Превью для UI: геометрия, без OpenCV (бирки/блики не цепляются)."""
    pil = _load_rgb_base(image_bytes=image_bytes, max_load_side=max_load_side)
    installation = _is_installation_shot(None, pil)

    if not installation:
        out = _ui_kansk_preview(crop_kansk_frame(pil))
    else:
        wr, hr = _ui_preview_crop_ratios(pil, installation)
        out = center_crop_focus(pil, wr, hr)

    return _pad_preview_aspect(out)


def pil_to_jpeg_bytes(pil: Image.Image, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _load_rgb_base(
    image_bytes: bytes | None = None,
    path: str | Path | None = None,
    max_load_side: int = MAX_LOAD_SIDE,
) -> Image.Image:
    """Поворот EXIF и уменьшение без обрезки предмета."""
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
    return pil.convert("RGB")


def _prepare_classifier_pil(
    image_bytes: bytes | None = None,
    path: str | Path | None = None,
    max_load_side: int = MAX_LOAD_SIDE,
) -> tuple[Image.Image, dict]:
    """Общая логика препроцессинга для обучения и инференса."""
    pil = _load_rgb_base(
        image_bytes=image_bytes,
        path=path,
        max_load_side=max_load_side,
    )
    installation = _is_installation_shot(path, pil)
    frame_cropped = False
    artifact_cropped = False

    if not installation:
        pil = crop_kansk_frame(pil)
        frame_cropped = True
        before = pil.size
        pil = crop_to_artifact(pil)
        artifact_cropped = pil.size != before

    return pil, {
        "installation_shot": installation,
        "frame_cropped": frame_cropped,
        "artifact_cropped": artifact_cropped,
    }


def load_classifier_rgb(
    image_bytes: bytes | None = None,
    path: str | Path | None = None,
    max_load_side: int = MAX_LOAD_SIDE,
) -> Image.Image:
    """Поворот, обрезка кадра и выделение предмета (для нейросети)."""
    pil, _ = _prepare_classifier_pil(
        image_bytes=image_bytes,
        path=path,
        max_load_side=max_load_side,
    )
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
    pil, prep_flags = _prepare_classifier_pil(image_bytes=image_bytes)
    texture = None
    if USE_TEXTURE_FEATURES:
        texture = torch.tensor(extract_texture_vector(pil), dtype=torch.float32).unsqueeze(0)
    tensor = _val_transforms()(pil).unsqueeze(0)
    meta = {
        "preprocess": "classifier",
        "frame_cropped": prep_flags["frame_cropped"],
        "artifact_cropped": prep_flags["artifact_cropped"],
        "installation_shot": prep_flags["installation_shot"],
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
