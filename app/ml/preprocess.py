"""
CV-предобработка входа для MobileNet.
Без OpenCV: PIL + NumPy (подходит для Raspberry / маленького экрана).
При установке opencv-python-headless — подключается CLAHE.
"""

from __future__ import annotations

import io
from typing import Tuple

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms

from app.ml.config import IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE


def _try_clahe(gray: np.ndarray) -> np.ndarray:
    try:
        import cv2

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    except ImportError:
        return gray


def cv_preprocess(image_bytes: bytes) -> Tuple[torch.Tensor, Image.Image, dict]:
    """
    Пайплайн:
    EXIF → RGB → авто-контраст → resize/center crop → тензор.
    Возвращает (tensor [1,C,H,W], PIL preview, meta).
    """
    meta: dict = {}
    pil = Image.open(io.BytesIO(image_bytes))
    pil = ImageOps.exif_transpose(pil)
    if pil.mode != "RGB":
        pil = pil.convert("RGB")

    gray = np.array(pil.convert("L"))
    meta["mean_brightness"] = float(np.mean(gray))

    pil = ImageOps.autocontrast(pil, cutoff=1)

    arr = np.array(pil)
    meta["clahe_applied"] = False
    try:
        import cv2  # noqa: F401

        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = _try_clahe(l)
        arr = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
        meta["clahe_applied"] = True
        pil = Image.fromarray(arr)
    except ImportError:
        pass

    tf = transforms.Compose(
        [
            transforms.Resize(INPUT_SIZE),
            transforms.CenterCrop(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    tensor = tf(pil).unsqueeze(0)
    return tensor, pil, meta


def tensor_to_preview_bytes(pil: Image.Image, max_side: int = 320) -> bytes:
    pil = pil.copy()
    pil.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=85)
    return buf.getvalue()
