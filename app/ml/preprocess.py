"""
Подготовка фото перед классификацией.

classifier_preprocess — обычный путь для сети 1.
cv_preprocess — старый тяжёлый вариант, для сети 2.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, replace
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
    if _has_led_reflection(pil):
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
    # Больше поля вокруг предмета — лучше видны пропорции относительно фона.
    pad = 0.22 if ui else 0.18
    mx, my = int(bw * pad), int(bh * pad)
    # Минимум в пикселях, чтобы тонкие объекты (ножи) не оставались «впритык».
    mx = max(mx, int(sw * 0.04))
    my = max(my, int(sh * 0.04))
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
    x_bias: float = 0.5,
    y_bias: float = 0.5,
) -> Image.Image:
    """Центральная область кадра; x/y_bias смещают окно (0 — влево/вверх)."""
    w, h = pil.size
    cw = max(1, min(w, int(w * width_ratio)))
    ch = max(1, min(h, int(h * height_ratio)))
    max_left = max(0, w - cw)
    max_top = max(0, h - ch)
    left = int(max_left * max(0.0, min(1.0, x_bias)))
    top = int(max_top * max(0.0, min(1.0, y_bias)))
    return pil.crop((left, top, left + cw, top + ch))


def _estimate_bg_level(gray: np.ndarray) -> float:
    h, w = gray.shape
    strip = max(1, min(h, w) // 40)
    border = np.concatenate([
        gray[:strip, :].ravel(),
        gray[-strip:, :].ravel(),
        gray[:, :strip].ravel(),
        gray[:, -strip:].ravel(),
    ])
    return float(np.median(border))


def _foreground_mask(pil: Image.Image) -> np.ndarray:
    """Пиксели предмета: отличаются от фона, без бирки, линейки и LED."""
    rgb = np.asarray(pil.convert("RGB"), dtype=np.float32)
    gray = rgb.mean(axis=2)
    bg = _estimate_bg_level(gray)
    mask = (np.abs(gray - bg) > 10) | (gray < bg - 16)
    mask &= gray < 228

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    orange_ruler = (r > 145) & (g < 135) & (b < 105) & (r > g + 18)
    led = gray > 212
    white_tag = gray > 238
    mask &= ~(orange_ruler | led | white_tag)
    return mask


def crop_to_foreground_bbox(
    pil: Image.Image,
    pad_ratio: float = 0.14,
    max_coverage: float = 0.90,
    min_coverage: float = 0.0,
) -> Image.Image:
    """Обрезка по bbox предмета; min_coverage не даёт уйти в сильный перезум."""
    mask = _foreground_mask(pil)
    if int(mask.sum()) < 60:
        return pil

    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    h, w = mask.shape
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    elong = bh / max(bw, 1)
    if elong > 2.2:
        pad_ratio = max(pad_ratio, 0.22)
    pad_x = max(2, int(bw * pad_ratio), int(w * 0.03))
    pad_y = max(2, int(bh * pad_ratio), int(h * 0.03))
    left = max(0, x0 - pad_x)
    top = max(0, y0 - pad_y)
    right = min(w, x1 + 1 + pad_x)
    bottom = min(h, y1 + 1 + pad_y)

    if min_coverage > 0:
        min_area = w * h * min_coverage
        crop_area = (right - left) * (bottom - top)
        if crop_area < min_area:
            cx = (left + right) / 2.0
            cy = (top + bottom) / 2.0
            scale = (min_area / max(crop_area, 1)) ** 0.5
            half_w = (right - left) * scale / 2.0
            half_h = (bottom - top) * scale / 2.0
            left = int(max(0, cx - half_w))
            right = int(min(w, cx + half_w))
            top = int(max(0, cy - half_h))
            bottom = int(min(h, cy + half_h))

    if (right - left) * (bottom - top) > w * h * max_coverage:
        return pil
    if (right - left) * (bottom - top) < w * h * 0.02:
        return pil
    return pil.crop((left, top, right, bottom))


def _preview_bbox_settings(
    installation: bool,
    object_class: str | None,
) -> tuple[float, float, float]:
    """pad_ratio, max_coverage, min_coverage."""
    if installation:
        return 0.24, 0.78, 0.38
    if object_class == "ножи":
        return 0.28, 0.88, 0.28
    if object_class == "наконечники стрел":
        return 0.24, 0.86, 0.28
    return 0.18, 0.88, 0.16


def _content_center_fraction(pil: Image.Image) -> tuple[float, float]:
    mask = _foreground_mask(pil)
    if int(mask.sum()) < 80:
        return 0.5, 0.5
    ys, xs = np.where(mask)
    h, w = mask.shape
    return float(xs.mean() / max(w - 1, 1)), float(ys.mean() / max(h - 1, 1))


def _has_led_reflection(pil: Image.Image) -> bool:
    """Горизонтальные блики LED-панели (съёмка на установке)."""
    gray = np.asarray(pil.convert("L"), dtype=np.float32)
    h, _w = gray.shape
    if h < 48:
        return False
    top_half = gray[: h // 2, :]
    row_bright = np.mean(top_half > 208, axis=1)
    stripe_rows = int(np.sum(row_bright > 0.18))
    return stripe_rows >= 4 and float(np.max(row_bright)) > 0.32


def _led_dominates_crop(pil: Image.Image) -> bool:
    """LED мешает превью — только если яркие полосы занимают заметную часть кадра."""
    gray = np.asarray(pil.convert("L"), dtype=np.float32)
    h, _w = gray.shape
    if h < 32:
        return False
    top = gray[: max(1, h // 2), :]
    if float(np.mean(top > 210)) < 0.08:
        return False
    row_bright = np.mean(top > 208, axis=1)
    return int(np.sum(row_bright > 0.22)) >= 3 and float(np.max(row_bright)) > 0.38


def center_crop_on_content(
    pil: Image.Image,
    width_ratio: float,
    height_ratio: float,
    x_bias: float = 0.5,
    y_bias: float = 0.5,
    y_shift: float = 0.0,
) -> Image.Image:
    """Crop вокруг центра масс предмета; иначе — геометрический bias."""
    w, h = pil.size
    cw = max(1, min(w, int(w * width_ratio)))
    ch = max(1, min(h, int(h * height_ratio)))
    cx, cy = _content_center_fraction(pil)
    if cx == 0.5 and cy == 0.5:
        return center_crop_focus(pil, width_ratio, height_ratio, x_bias, y_bias)
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy + y_shift))
    left = int(cx * w - cw / 2)
    top = int(cy * h - ch / 2)
    left = max(0, min(w - cw, left))
    top = max(0, min(h - ch, top))
    return pil.crop((left, top, left + cw, top + ch))


def _fit_preview_cover(
    pil: Image.Image,
    target: float = 4 / 3,
    *,
    min_fill: float = 0.55,
) -> Image.Image:
    """4:3 для превью; при уже крупном crop — мягче, без лишнего перезума."""
    w, h = pil.size
    if h <= 0:
        return pil
    img_aspect = w / h
    if abs(img_aspect - target) < 0.04:
        return pil
    if img_aspect > target:
        new_w = max(1, int(h * target))
        if new_w >= w * min_fill:
            left = max(0, (w - new_w) // 2)
            return pil.crop((left, 0, left + new_w, h))
        return _pad_preview_aspect(pil, target=target)
    new_h = max(1, int(w / target))
    if new_h >= h * min_fill:
        top = max(0, (h - new_h) // 2)
        return pil.crop((0, top, w, top + new_h))
    return _pad_preview_aspect(pil, target=target)


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


@dataclass(frozen=True)
class UiPreviewPreset:
    """Параметры геометрического превью."""

    name: str = "standard"
    left_trim: float = 0.02
    right_trim: float = 0.12
    top_trim: float = 0.05
    bottom_trim: float = 0.13
    width_ratio: float = 0.84
    height_ratio: float = 0.72
    x_bias: float = 0.42
    y_bias: float = 0.46


KANSK_ASPECT_PRESETS: dict[str, UiPreviewPreset] = {
    "wide": UiPreviewPreset(
        name="wide",
        right_trim=0.15,
        bottom_trim=0.15,
        width_ratio=0.78,
        height_ratio=0.60,
        x_bias=0.36,
        y_bias=0.44,
    ),
    "standard": UiPreviewPreset(
        name="standard",
        right_trim=0.12,
        bottom_trim=0.13,
        width_ratio=0.84,
        height_ratio=0.72,
        x_bias=0.42,
        y_bias=0.46,
    ),
    "tall": UiPreviewPreset(
        name="tall",
        left_trim=0.10,
        right_trim=0.10,
        bottom_trim=0.10,
        width_ratio=0.58,
        height_ratio=0.92,
        x_bias=0.50,
        y_bias=0.50,
    ),
}

CLASS_PRESET_OVERRIDES: dict[str, dict[str, float | str]] = {
    "ножи": {
        "name": "class:ножи",
        "left_trim": 0.12,
        "right_trim": 0.22,
        "width_ratio": 0.62,
        "height_ratio": 0.88,
        "x_bias": 0.50,
        "y_bias": 0.50,
    },
    "кельты": {
        "name": "class:кельты",
        "width_ratio": 0.80,
        "height_ratio": 0.68,
        "x_bias": 0.40,
    },
    "наконечники стрел": {
        "name": "class:наконечники",
        "width_ratio": 0.76,
        "height_ratio": 0.62,
        "x_bias": 0.38,
        "y_bias": 0.44,
    },
    "удила": {
        "name": "class:удила",
        "width_ratio": 0.68,
        "height_ratio": 0.68,
        "x_bias": 0.45,
        "y_bias": 0.48,
    },
    "накладки": {
        "name": "class:накладки",
        "width_ratio": 0.72,
        "height_ratio": 0.72,
        "x_bias": 0.40,
        "bottom_trim": 0.11,
    },
}

SOFT_KANSK_PRESET = UiPreviewPreset(
    name="soft",
    left_trim=0.04,
    right_trim=0.06,
    top_trim=0.03,
    bottom_trim=0.08,
    width_ratio=0.78,
    height_ratio=0.88,
    x_bias=0.50,
    y_bias=0.50,
)


def _kansk_aspect_bucket(aspect: float) -> str:
    if aspect > 1.28:
        return "wide"
    if aspect < 0.88:
        return "tall"
    return "standard"


def _resolve_kansk_preset(framed: Image.Image, object_class: str | None) -> UiPreviewPreset:
    aspect = framed.size[0] / max(framed.size[1], 1)
    preset = KANSK_ASPECT_PRESETS[_kansk_aspect_bucket(aspect)]
    if object_class and object_class in CLASS_PRESET_OVERRIDES:
        preset = replace(preset, **CLASS_PRESET_OVERRIDES[object_class])
    return preset


def crop_kansk_preview_trim(
    pil: Image.Image,
    preset: UiPreviewPreset | None = None,
) -> Image.Image:
    """Убираем зону бирки справа — предмет обычно левее центра."""
    p = preset or KANSK_ASPECT_PRESETS["standard"]
    w, h = pil.size
    left = int(w * p.left_trim)
    right = int(w * (1 - p.right_trim))
    top = int(h * p.top_trim)
    bottom = int(h * (1 - p.bottom_trim))
    if right <= left + 8 or bottom <= top + 8:
        return pil
    return pil.crop((left, top, right, bottom))


def _build_kansk_preview(
    framed: Image.Image,
    preset: UiPreviewPreset,
    object_class: str | None = None,
) -> Image.Image:
    trimmed = crop_kansk_preview_trim(framed, preset)
    # Длинные ножи: bbox цепляется за текстуру — только геометрия.
    if object_class == "ножи":
        return center_crop_on_content(
            trimmed,
            preset.width_ratio,
            preset.height_ratio,
            preset.x_bias,
            preset.y_bias,
        )

    pad, max_cov, min_cov = _preview_bbox_settings(False, object_class)
    tight = crop_to_foreground_bbox(
        trimmed, pad_ratio=pad, max_coverage=max_cov, min_coverage=min_cov
    )
    if tight.size != trimmed.size:
        return tight
    return center_crop_on_content(
        trimmed,
        preset.width_ratio,
        preset.height_ratio,
        preset.x_bias,
        preset.y_bias,
    )


def _installation_preview_ratios(
    pil: Image.Image,
    object_class: str | None,
) -> tuple[float, float, float]:
    """Полевая/установка: умеренный zoom — в кадре и так мало лишнего."""
    w, h = pil.size
    aspect = w / max(h, 1)
    mean_b = float(np.mean(np.array(pil.convert("L"))))

    if object_class == "ножи":
        return 0.72, 0.78, 0.04
    if mean_b < 55:
        return 0.68, 0.72, 0.05
    if aspect >= FIELD_ASPECT_MIN:
        return 0.70, 0.68, 0.04
    if object_class == "накладки":
        return 0.66, 0.64, 0.05
    return 0.68, 0.66, 0.04


def _build_installation_preview(
    pil: Image.Image,
    object_class: str | None,
) -> Image.Image:
    source = pil
    if _has_led_reflection(pil):
        w, h = pil.size
        source = pil.crop((0, int(h * 0.28), w, h))

    wr, hr, y_shift = _installation_preview_ratios(source, object_class)
    return center_crop_on_content(source, wr, hr, y_shift=y_shift)


def _validate_ui_preview(
    original: Image.Image,
    cropped: Image.Image,
    *,
    installation: bool,
    object_class: str | None = None,
) -> tuple[bool, str]:
    """Эвристики: бирка, блик, линейка, слишком сильный/слабый crop."""
    ow, oh = cropped.size
    pw, ph = original.size
    if ow < 32 or oh < 32:
        return False, "too_small"

    area_ratio = (ow * oh) / max(pw * ph, 1)
    fg_frac = float(_foreground_mask(cropped).mean())
    min_area = 0.04 if installation else 0.05
    if area_ratio < min_area and fg_frac < 0.08:
        return False, "crop_too_tight"
    # С более широким pad допустим чуть больший кадр
    if area_ratio > 0.94:
        return False, "crop_too_weak"

    aspect = ow / max(oh, 1)
    max_aspect = 3.6 if object_class == "наконечники стрел" else 2.8
    if object_class == "ножи":
        min_aspect = 0.15
    elif object_class == "наконечники стрел":
        min_aspect = 0.28
    else:
        min_aspect = 0.38
    if aspect > max_aspect or aspect < min_aspect:
        return False, "bad_aspect"

    gray = np.asarray(cropped.convert("L"), dtype=np.float32)
    mean_b = float(gray.mean())
    std_b = float(gray.std())
    white_frac = float(np.mean(gray > 242))

    if mean_b > 200 and white_frac > 0.30:
        return False, "mostly_white"
    if mean_b > 215 and std_b < 22:
        return False, "flat_white"

    w = gray.shape[1]
    h = gray.shape[0]
    if w >= 40:
        core = float(gray[:, w // 5 : 4 * w // 5].mean())
        right_edge = float(gray[:, int(w * 0.82) :].mean())
        left_edge = float(gray[:, : max(1, w // 8)].mean())
        if right_edge - core > 45 and white_frac > 0.22:
            return False, "tag_on_right"
        if left_edge > 210 and white_frac > 0.14:
            return False, "tag_on_left"

    if _led_dominates_crop(cropped):
        return False, "led_reflection"

    if fg_frac < 0.03:
        return False, "no_content"

    cx, cy = _content_center_fraction(cropped)
    if fg_frac < 0.12:
        limit_x = 0.36 if installation else 0.32
        limit_y = 0.38 if installation else 0.34
        if abs(cx - 0.5) > limit_x or abs(cy - 0.5) > limit_y:
            return False, "content_off_center"

    if aspect > 2.3 and std_b < 28:
        return False, "reflection_strip"

    if not installation and h >= 30:
        bottom_band = float(gray[int(h * 0.88) :, :].mean())
        upper = float(gray[: int(h * 0.72), :].mean())
        if bottom_band < 35 and upper - bottom_band > 48:
            return False, "ruler_bottom"

    return True, "ok"


def load_ui_preview_rgb(
    image_bytes: bytes,
    max_load_side: int = MAX_LOAD_SIDE,
    object_class: str | None = None,
    source_path: str | Path | None = None,
) -> tuple[Image.Image, dict]:
    """Превью для UI: пресеты по aspect/классу + fallback на оригинал."""
    pil = _load_rgb_base(image_bytes=image_bytes, max_load_side=max_load_side)
    installation = _is_installation_shot(source_path, pil)
    meta: dict = {
        "installation": installation,
        "fallback": False,
        "preview_mode": "cropped",
        "preset": None,
        "reason": None,
        "object_class": object_class,
    }

    if not installation:
        framed = crop_kansk_frame(pil)
        preset = _resolve_kansk_preset(framed, object_class)
        out = _build_kansk_preview(framed, preset, object_class=object_class)
        meta["preset"] = preset.name
        area_frac = (out.size[0] * out.size[1]) / max(framed.size[0] * framed.size[1], 1)
        if area_frac < 0.32 and object_class != "ножи":
            meta["preset"] = f"{preset.name}+tight"
        ok, reason = _validate_ui_preview(
            pil, out, installation=False, object_class=object_class
        )
        if not ok:
            soft_out = _build_kansk_preview(
                framed, SOFT_KANSK_PRESET, object_class=object_class
            )
            ok_soft, _ = _validate_ui_preview(
                pil, soft_out, installation=False, object_class=object_class
            )
            if ok_soft:
                out = soft_out
                meta["preset"] = SOFT_KANSK_PRESET.name
            else:
                trimmed = crop_kansk_preview_trim(framed, preset)
                pad, max_cov, min_cov = _preview_bbox_settings(False, object_class)
                bbox_out = crop_to_foreground_bbox(
                    trimmed, pad_ratio=pad, max_coverage=max_cov, min_coverage=min_cov
                )
                ok_bbox, _ = _validate_ui_preview(
                    pil, bbox_out, installation=False, object_class=object_class
                )
                if ok_bbox and bbox_out.size != trimmed.size:
                    out = bbox_out
                    meta["preset"] = "tight-bbox"
                else:
                    bbox_full = crop_to_foreground_bbox(
                        framed, pad_ratio=pad, max_coverage=max_cov, min_coverage=min_cov
                    )
                    ok_full, _ = _validate_ui_preview(
                        pil, bbox_full, installation=False, object_class=object_class
                    )
                    if ok_full and bbox_full.size != framed.size:
                        out = bbox_full
                        meta["preset"] = "tight-bbox"
                    else:
                        out = pil
                        meta.update(
                            fallback=True,
                            preview_mode="original",
                            reason=reason,
                        )
    else:
        out = _build_installation_preview(pil, object_class)
        meta["preset"] = "installation+content" if _has_led_reflection(pil) else "installation"
        area_frac = (out.size[0] * out.size[1]) / max(pil.size[0] * pil.size[1], 1)
        if area_frac < 0.38:
            meta["preset"] = str(meta["preset"]) + "+tight"
        ok, reason = _validate_ui_preview(
            pil, out, installation=True, object_class=object_class
        )
        if not ok:
            soft_src = pil
            if _has_led_reflection(pil):
                w, h = pil.size
                soft_src = pil.crop((0, int(h * 0.30), w, h))
            soft_out = center_crop_on_content(soft_src, 0.76, 0.72)
            ok_soft, _ = _validate_ui_preview(
                pil, soft_out, installation=True, object_class=object_class
            )
            if ok_soft:
                out = soft_out
                meta["preset"] = "installation-soft"
            else:
                out = pil
                meta.update(
                    fallback=True,
                    preview_mode="original",
                    reason=reason,
                )

    return _fit_preview_cover(out), meta


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


def classifier_preprocess(
    image_bytes: bytes,
    source_path: str | Path | None = None,
) -> Tuple[torch.Tensor, Image.Image, dict, torch.Tensor | None]:
    """Фото → тензор для сети 1 + текстура, если включена."""
    pil, prep_flags = _prepare_classifier_pil(
        image_bytes=image_bytes,
        path=source_path,
    )
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
