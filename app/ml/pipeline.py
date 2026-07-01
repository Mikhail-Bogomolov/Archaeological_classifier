"""Сначала тип объекта, потом признаки (когда сеть 2 будет готова)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from app.ml.config import (
    FEATURE_SCHEMA,
    FEATURE_MODEL_FILE,
    MODELS_DIR,
    OBJECT_CLASSES,
    OBJECT_MODEL_FILE,
    USE_TEXTURE_FEATURES,
)
from app.ml.encoders import (
    decode_feature_attributes,
    decode_object_class,
    one_hot_from_name,
)
from app.ml.models import FeatureClassifierNet, ObjectClassifierNet
from app.ml.preprocess import classifier_preprocess, cv_preprocess


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
        self.object_net = ObjectClassifierNet(use_texture=USE_TEXTURE_FEATURES).to(self.device)
        self.feature_net: FeatureClassifierNet | None = None
        self._feature_vocab: dict[str, list[str]] | None = None
        self._object_weights_loaded = False
        self._feature_weights_loaded = False
        self._load_weights_if_present()
        self.object_net.eval()
        if self.feature_net is not None:
            self.feature_net.eval()

    def _load_weights_if_present(self) -> None:
        obj_path = Path(MODELS_DIR) / OBJECT_MODEL_FILE
        feat_path = Path(MODELS_DIR) / FEATURE_MODEL_FILE
        if obj_path.is_file():
            state = torch.load(obj_path, map_location=self.device, weights_only=True)
            has_texture = any(k.startswith("texture_mlp") for k in state)
            if USE_TEXTURE_FEATURES and not has_texture:
                self.object_net = ObjectClassifierNet(use_texture=False).to(self.device)
            self.object_net.load_state_dict(state, strict=False)
            self._object_weights_loaded = True
        if feat_path.is_file():
            ckpt = torch.load(feat_path, map_location=self.device, weights_only=False)
            if isinstance(ckpt, dict) and "state_dict" in ckpt and "vocab" in ckpt:
                vocab = ckpt["vocab"]
                state = ckpt["state_dict"]
                use_feat_texture = ckpt.get(
                    "use_texture",
                    any(k.startswith("texture_mlp") for k in state),
                )
                self._feature_vocab = vocab
                self.feature_net = FeatureClassifierNet.from_vocab(
                    vocab, pretrained=False, use_texture=use_feat_texture
                ).to(self.device)
                self.feature_net.load_state_dict(state, strict=False)
                self._feature_weights_loaded = True

    def _predict_feature_logits(
        self,
        tensor: torch.Tensor,
        object_one_hot: torch.Tensor,
        texture: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        assert self.feature_net is not None
        if (
            self.feature_net.use_texture
            and self.feature_net.texture_mlp is not None
            and texture is not None
        ):
            return self.feature_net(tensor, object_one_hot, texture.to(self.device))
        return self.feature_net(tensor, object_one_hot)

    @property
    def ready(self) -> bool:
        return self._object_weights_loaded

    @torch.no_grad()
    def predict(self, image_bytes: bytes, object_name: str | None = None) -> PredictionResult:
        is_demo = not self._object_weights_loaded
        texture = None
        if self._object_weights_loaded:
            tensor, _preview, meta, texture = classifier_preprocess(image_bytes)
            tensor = tensor.to(self.device)
            if (
                self.object_net.use_texture
                and self.object_net.texture_mlp is not None
                and texture is not None
            ):
                texture = texture.to(self.device)
                obj_logits = self.object_net(tensor, texture)
            else:
                obj_logits = self.object_net(tensor)
        else:
            tensor, _preview, meta = cv_preprocess(image_bytes)
            tensor = tensor.to(self.device)
            obj_logits = self.object_net(tensor)
        object_class, object_conf = decode_object_class(obj_logits, OBJECT_CLASSES)

        feature_lines: list[str] = []
        if self._feature_weights_loaded and self.feature_net is not None and self._feature_vocab:
            obj_oh = one_hot_from_name(object_class, OBJECT_CLASSES).to(self.device)
            feat_logits = self._predict_feature_logits(tensor, obj_oh, texture if self._object_weights_loaded else None)
            feature_lines = decode_feature_attributes(
                feat_logits,
                self._feature_vocab,
                object_class,
                FEATURE_SCHEMA.get(object_class, []),
            )
        elif is_demo:
            feature_lines.append(
                "Модель не обучена — результат демонстрационный (случайные веса)",
            )
        else:
            feature_lines.append(
                "Признаки: модуль признаков ещё не обучен",
            )

        display_name = object_name.strip() if object_name and object_name.strip() else "Новый объект"
        confidence_pct = int(object_conf * 100)

        description = (
            f"Тип: {object_class}. "
            + ("Ожидается обучение на вашем датасете." if is_demo else "Классификатор обучен.")
        )
        if not is_demo and object_conf < 0.55:
            description += " Низкая уверенность — результат может быть неточным."

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

    @torch.no_grad()
    def predict_features(self, image_bytes: bytes, object_class: str) -> list[str]:
        """Признаки для уже выбранного пользователем класса."""
        if object_class not in OBJECT_CLASSES:
            return []

        texture = None
        if self._object_weights_loaded:
            tensor, _preview, meta, texture = classifier_preprocess(image_bytes)
            tensor = tensor.to(self.device)
        else:
            tensor, _preview, meta = cv_preprocess(image_bytes)
            tensor = tensor.to(self.device)

        if self._feature_weights_loaded and self.feature_net is not None and self._feature_vocab:
            obj_oh = one_hot_from_name(object_class, OBJECT_CLASSES).to(self.device)
            feat_logits = self._predict_feature_logits(tensor, obj_oh, texture if self._object_weights_loaded else None)
            return decode_feature_attributes(
                feat_logits,
                self._feature_vocab,
                object_class,
                FEATURE_SCHEMA.get(object_class, []),
            )

        if not self._object_weights_loaded:
            return ["Модель не обучена — результат демонстрационный (случайные веса)"]
        return ["Признаки: модуль признаков ещё не обучен"]

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
