"""
Двухэтапный инференс:
  1) ObjectClassifierNet → класс объекта
  2) FeatureClassifierNet(image, one_hot(class)) → признаки
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from app.ml.config import (
    FEATURE_INDICES_BY_CLASS,
    FEATURE_LABELS,
    FEATURE_MODEL_FILE,
    MODELS_DIR,
    OBJECT_CLASSES,
    OBJECT_MODEL_FILE,
)
from app.ml.encoders import (
    decode_features,
    decode_object_class,
    one_hot_from_name,
)
from app.ml.models import FeatureClassifierNet, ObjectClassifierNet
from app.ml.preprocess import cv_preprocess


@dataclass
class PredictionResult:
    name: str
    description: str
    category: str
    confidence: int
    features: list[str]
    object_class: str
    object_confidence: float
    is_demo: bool
    preprocess_meta: dict


class ArchaeologyClassifierPipeline:
    def __init__(self, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.object_net = ObjectClassifierNet().to(self.device)
        self.feature_net = FeatureClassifierNet().to(self.device)
        self._weights_loaded = False
        self._load_weights_if_present()
        self.object_net.eval()
        self.feature_net.eval()

    def _load_weights_if_present(self) -> None:
        obj_path = Path(MODELS_DIR) / OBJECT_MODEL_FILE
        feat_path = Path(MODELS_DIR) / FEATURE_MODEL_FILE
        if obj_path.is_file():
            self.object_net.load_state_dict(
                torch.load(obj_path, map_location=self.device, weights_only=True)
            )
            self._weights_loaded = True
        if feat_path.is_file():
            self.feature_net.load_state_dict(
                torch.load(feat_path, map_location=self.device, weights_only=True)
            )

    @property
    def ready(self) -> bool:
        return self._weights_loaded

    @torch.no_grad()
    def predict(self, image_bytes: bytes, object_name: str | None = None) -> PredictionResult:
        tensor, _preview, meta = cv_preprocess(image_bytes)
        tensor = tensor.to(self.device)

        obj_logits = self.object_net(tensor)
        object_class, object_conf = decode_object_class(obj_logits, OBJECT_CLASSES)

        obj_oh = one_hot_from_name(object_class, OBJECT_CLASSES).to(self.device)
        feat_logits = self.feature_net(tensor, obj_oh)
        decoded = decode_features(feat_logits, FEATURE_LABELS, threshold=0.5)

        # Оставляем только признаки, относящиеся к предсказанному классу
        allowed = set(FEATURE_INDICES_BY_CLASS.get(object_class, []))
        feature_lines: list[str] = []
        for label, prob in decoded:
            global_idx = FEATURE_LABELS.index(label)
            if global_idx in allowed:
                short = label.split(":", 1)[-1]
                feature_lines.append(f"{short}: {prob:.0%}")

        is_demo = not self.ready
        if is_demo:
            feature_lines.insert(
                0,
                "Модель не обучена — результат демонстрационный (случайные веса)",
            )

        display_name = object_name.strip() if object_name and object_name.strip() else "Новый объект"
        confidence_pct = int(object_conf * 100)

        description = (
            f"Тип: {object_class}. "
            + ("Ожидается обучение на вашем датасете." if is_demo else "Классификатор обучен.")
        )

        return PredictionResult(
            name=display_name,
            description=description,
            category=object_class,
            confidence=confidence_pct,
            features=feature_lines,
            object_class=object_class,
            object_confidence=object_conf,
            is_demo=is_demo,
            preprocess_meta=meta,
        )

    def to_api_dict(self, result: PredictionResult) -> dict[str, Any]:
        return {
            "name": result.name,
            "description": result.description,
            "category": result.category,
            "confidence": result.confidence,
            "features": result.features,
            "object_class": result.object_class,
            "is_demo": result.is_demo,
        }


_pipeline: ArchaeologyClassifierPipeline | None = None


def get_pipeline() -> ArchaeologyClassifierPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = ArchaeologyClassifierPipeline()
    return _pipeline
