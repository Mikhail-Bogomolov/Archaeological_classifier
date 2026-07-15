"""Focal loss и веса классов для мультиклассовой головы."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    weight: torch.Tensor | None = None,
    gamma: float = 2.0,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, targets, weight=weight, reduction="none")
    pt = torch.exp(-ce)
    loss = ((1.0 - pt) ** gamma) * ce
    return loss.mean()


def class_weights_from_counts(counts: list[int], device: torch.device) -> torch.Tensor:
    """Обратно частоте; нулевые классы получают вес 0."""
    total = sum(counts)
    n = len(counts)
    weights = []
    for c in counts:
        weights.append(total / (n * c) if c > 0 else 0.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_object_loss_weights(
    counts: list[int],
    device: torch.device,
    *,
    focus_class_idx: int | None = None,
    focus_boost: float = 1.0,
) -> torch.Tensor:
    """Веса CE; для focus-класса (ножи) — дополнительный множитель."""
    weights = class_weights_from_counts(counts, device)
    if focus_class_idx is not None and focus_boost > 1.0:
        weights[focus_class_idx] *= focus_boost
    return weights
