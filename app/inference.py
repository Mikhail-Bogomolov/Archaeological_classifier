"""Разбор фото и выдача результата классификации."""

from __future__ import annotations

from typing import Any

from app.ml.pipeline import get_pipeline
from app.ml.preprocess import load_ui_preview_rgb, pil_to_jpeg_bytes


def _build_preview(
    contents: bytes,
    object_class: str | None = None,
    source_path: str | Path | None = None,
) -> tuple[bytes | None, dict]:
    try:
        preview, meta = load_ui_preview_rgb(
            contents,
            object_class=object_class,
            source_path=source_path,
        )
        return pil_to_jpeg_bytes(preview), meta
    except Exception:
        return None, {"fallback": True, "preview_mode": "original", "reason": "error"}


def run_inference(
    contents: bytes,
    object_name: str | None = None,
    source_path: str | Path | None = None,
) -> dict[str, Any]:
    pipeline = get_pipeline()
    result = pipeline.predict(contents, object_name, source_path=source_path)
    d = pipeline.to_api_dict(result)
    object_class = str(d.get("object_class") or d.get("category") or "") or None
    preview_bytes, preview_meta = _build_preview(
        contents, object_class=object_class, source_path=source_path
    )
    if preview_bytes is not None:
        d["preview_image_bytes"] = preview_bytes
    d["preview_meta"] = preview_meta
    if result.preprocess_meta.get("clahe_applied"):
        d.setdefault("features", []).append("Предобработка: CLAHE")
    return d
