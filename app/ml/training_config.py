"""Гиперпараметры обучения и инференса — единый источник правды."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from app.ml.splits import DEFAULT_SPLIT_SEED, DEFAULT_TEST_RATIO, DEFAULT_VAL_RATIO


@dataclass
class ObjectTrainingConfig:
    epochs: int = 50
    batch_size: int = 16
    lr: float = 5e-5
    head_lr: float = 1e-3
    patience: int = 12
    label_smoothing: float = 0.05
    mixup_alpha: float = 0.15
    focal_gamma: float = 0.0
    focus_class: str = "ножи"
    focus_boost: float = 1.25
    selection_metric: str = "balanced"
    freeze_epochs_pretrained: int = 3
    freeze_epochs_scratch: int = 5
    val_ratio: float = DEFAULT_VAL_RATIO
    test_ratio: float = DEFAULT_TEST_RATIO
    split_seed: int = DEFAULT_SPLIT_SEED
    use_texture: bool = True
    pretrained: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelArchitectureConfig:
    tex_hidden: int = 64
    feature_hidden_dim: int = 384
    texture_size: int = 224

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InferenceConfig:
    object_low_conf_threshold: float = 0.55
    feature_min_conf: float = 0.2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_OBJECT_TRAINING = ObjectTrainingConfig()
DEFAULT_MODEL_ARCH = ModelArchitectureConfig()
DEFAULT_INFERENCE = InferenceConfig()


def checkpoint_payload(
    state_dict: dict[str, Any],
    *,
    model_kind: str,
    training: ObjectTrainingConfig | dict[str, Any] | None = None,
    architecture: ModelArchitectureConfig | dict[str, Any] | None = None,
    best_epoch: int | None = None,
    best_score: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model_kind": model_kind,
        "state_dict": state_dict,
    }
    if training is not None:
        payload["training"] = (
            training if isinstance(training, dict) else training.to_dict()
        )
    if architecture is not None:
        payload["architecture"] = (
            architecture if isinstance(architecture, dict) else architecture.to_dict()
        )
    if best_epoch is not None:
        payload["best_epoch"] = best_epoch
    if best_score is not None:
        payload["best_score"] = best_score
    if extra:
        payload["extra"] = extra
    return payload


def load_state_dict(checkpoint: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Поддержка старых .pt (только state_dict) и новых (dict с meta)."""
    meta: dict[str, Any] = {}
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        meta = {k: v for k, v in checkpoint.items() if k != "state_dict"}
        return checkpoint["state_dict"], meta
    if isinstance(checkpoint, dict):
        return checkpoint, meta
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")
