"""One-hot и multi-hot кодирование меток."""

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
    probs = torch.sigmoid(logits)
    out: list[tuple[str, float]] = []
    for i, p in enumerate(probs.squeeze().tolist()):
        if p >= threshold and i < len(feature_labels):
            out.append((feature_labels[i], float(p)))
    return out
