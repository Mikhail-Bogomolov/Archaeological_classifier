"""Кодирование меток для обучения."""

from __future__ import annotations

import torch


def one_hot_class(index: int, num_classes: int) -> torch.Tensor:
    v = torch.zeros(num_classes, dtype=torch.float32)
    if 0 <= index < num_classes:
        v[index] = 1.0
    return v


def one_hot_from_name(name: str, classes: list[str]) -> torch.Tensor:
    try:
        idx = classes.index(name)
    except ValueError:
        idx = classes.index("неопределено") if "неопределено" in classes else 0
    return one_hot_class(idx, len(classes))


def multihot_features(active_indices: list[int], num_features: int) -> torch.Tensor:
    v = torch.zeros(num_features, dtype=torch.float32)
    for i in active_indices:
        if 0 <= i < num_features:
            v[i] = 1.0
    return v


def decode_object_class(logits: torch.Tensor, classes: list[str]) -> tuple[str, float]:
    probs = torch.softmax(logits, dim=-1)
    conf, idx = torch.max(probs, dim=-1)
    i = int(idx.item())
    return classes[i], float(conf.item())


def decode_features(
    logits: torch.Tensor,
    feature_labels: list[str],
    threshold: float = 0.5,
) -> list[tuple[str, float]]:
    """Устаревший multi-label декодер (сеть 2 v1)."""
    probs = torch.sigmoid(logits)
    out: list[tuple[str, float]] = []
    for i, p in enumerate(probs.squeeze().tolist()):
        if p >= threshold and i < len(feature_labels):
            out.append((feature_labels[i], float(p)))
    return out


def decode_feature_attributes(
    logits: dict[str, torch.Tensor],
    vocab: dict[str, list[str]],
    object_class: str,
    feature_names: list[str],
    min_conf: float = 0.2,
) -> list[str]:
    """Декодирование голов сети 2 в строки для UI."""
    lines: list[str] = []
    for feat_name in feature_names:
        key = f"{object_class}:{feat_name}"
        if key not in logits or key not in vocab:
            continue
        head_logits = logits[key]
        if head_logits.dim() == 1:
            head_logits = head_logits.unsqueeze(0)
        probs = torch.softmax(head_logits, dim=-1).squeeze(0)
        conf, idx = torch.max(probs, dim=-1)
        if float(conf) < min_conf:
            continue
        label = vocab[key][int(idx.item())]
        lines.append(f"{feat_name}: {label} ({int(float(conf) * 100)}%)")
    return lines
