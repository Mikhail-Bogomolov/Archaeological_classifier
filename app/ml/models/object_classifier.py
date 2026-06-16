"""Классификатор типа объекта + текстура поверхности."""

from __future__ import annotations

import torch
import torch.nn as nn

from app.ml.config import NUM_OBJECT_CLASSES
from app.ml.models.backbone import create_mobilenet_backbone
from app.ml.texture_features import texture_dim


class ObjectClassifierNet(nn.Module):
    def __init__(
        self,
        num_classes: int = NUM_OBJECT_CLASSES,
        backbone: str = "v3_small",
        pretrained: bool = True,
        dropout: float = 0.25,
        texture_dim_in: int | None = None,
        use_texture: bool = True,
    ):
        super().__init__()
        self.use_texture = use_texture
        self.backbone, feat_dim = create_mobilenet_backbone(backbone, pretrained=pretrained)

        tex_in = texture_dim_in if texture_dim_in is not None else texture_dim()
        tex_hidden = 64
        if use_texture:
            self.texture_mlp = nn.Sequential(
                nn.Linear(tex_in, tex_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            head_in = feat_dim + tex_hidden
        else:
            self.texture_mlp = None
            head_in = feat_dim

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(head_in, num_classes),
        )

    def forward(self, x: torch.Tensor, texture: torch.Tensor | None = None) -> torch.Tensor:
        visual = self.backbone(x)
        if self.use_texture and self.texture_mlp is not None:
            if texture is None:
                raise ValueError("Нужен вектор текстуры")
            tex = self.texture_mlp(texture)
            visual = torch.cat([visual, tex], dim=1)
        return self.head(visual)
