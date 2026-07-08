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
