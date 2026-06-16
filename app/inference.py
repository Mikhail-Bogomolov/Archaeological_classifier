"""Разбор фото и выдача результата классификации."""

from __future__ import annotations

from typing import Any

from app.ml.pipeline import get_pipeline


def run_inference(contents: bytes, object_name: str | None = None) -> dict[str, Any]:
    pipeline = get_pipeline()
    result = pipeline.predict(contents, object_name)
    d = pipeline.to_api_dict(result)
    if result.preprocess_meta.get("clahe_applied"):
        d.setdefault("features", []).append("Предобработка: CLAHE")
    return d
