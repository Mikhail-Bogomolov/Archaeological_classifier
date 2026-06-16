"""Признаки объекта по типу (сеть 2)."""

from __future__ import annotations

import torch
import torch.nn as nn

from app.ml.config import NUM_FEATURES, NUM_OBJECT_CLASSES
from app.ml.models.backbone import create_mobilenet_backbone


class FeatureClassifierNet(nn.Module):
    def __init__(
        self,
        num_features: int = NUM_FEATURES,
        num_object_classes: int = NUM_OBJECT_CLASSES,
        backbone: str = "v2",
        pretrained: bool = False,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_object_classes = num_object_classes
        self.backbone, feat_dim = create_mobilenet_backbone(backbone, pretrained=pretrained)
        self.head = nn.Sequential(
            nn.Linear(feat_dim + num_object_classes, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_features),
        )

    def forward(self, x: torch.Tensor, object_one_hot: torch.Tensor) -> torch.Tensor:
        visual = self.backbone(x)
        if object_one_hot.dim() == 1:
            object_one_hot = object_one_hot.unsqueeze(0)
        fused = torch.cat([visual, object_one_hot], dim=1)
        return self.head(fused)
