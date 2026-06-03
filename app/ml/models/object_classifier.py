"""Сеть 1: классификация типа объекта (softmax / one-hot при обучении)."""

from __future__ import annotations

import torch
import torch.nn as nn

from app.ml.config import NUM_OBJECT_CLASSES
from app.ml.models.backbone import create_mobilenet_backbone


class ObjectClassifierNet(nn.Module):
    def __init__(
        self,
        num_classes: int = NUM_OBJECT_CLASSES,
        backbone: str = "v3_small",
        pretrained: bool = False,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone, feat_dim = create_mobilenet_backbone(backbone, pretrained=pretrained)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)
