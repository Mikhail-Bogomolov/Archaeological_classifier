"""Разбор фото и выдача результата классификации."""

from __future__ import annotations

from typing import Any

from app.ml.pipeline import get_pipeline
from app.ml.preprocess import load_ui_preview_rgb, pil_to_jpeg_bytes


def run_inference(contents: bytes, object_name: str | None = None) -> dict[str, Any]:
    pipeline = get_pipeline()
    result = pipeline.predict(contents, object_name)
    d = pipeline.to_api_dict(result)
    try:
        d["preview_image_bytes"] = pil_to_jpeg_bytes(load_ui_preview_rgb(contents))
    except Exception:
        pass
    if result.preprocess_meta.get("clahe_applied"):
        d.setdefault("features", []).append("Предобработка: CLAHE")
    return d
