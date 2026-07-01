"""Признаки объекта по типу (сеть 2) — голова на признак + опционально текстура."""

from __future__ import annotations

import torch
import torch.nn as nn

from app.ml.config import NUM_OBJECT_CLASSES
from app.ml.feature_vocab import head_key
from app.ml.models.backbone import create_mobilenet_backbone
from app.ml.texture_features import texture_dim


class FeatureClassifierNet(nn.Module):
    def __init__(
        self,
        attribute_specs: dict[str, int],
        num_object_classes: int = NUM_OBJECT_CLASSES,
        backbone: str = "v2",
        pretrained: bool = True,
        dropout: float = 0.3,
        hidden_dim: int = 256,
        use_texture: bool = True,
        texture_dim_in: int | None = None,
    ):
        super().__init__()
        self.num_object_classes = num_object_classes
        self.attribute_specs = dict(attribute_specs)
        self.attr_keys = sorted(self.attribute_specs.keys())
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
            fused_dim = feat_dim + num_object_classes + tex_hidden
        else:
            self.texture_mlp = None
            fused_dim = feat_dim + num_object_classes

        self.shared = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleDict({
            head_key(key): nn.Linear(hidden_dim, n_classes)
            for key, n_classes in self.attribute_specs.items()
        })

    def forward(
        self,
        x: torch.Tensor,
        object_one_hot: torch.Tensor,
        texture: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        visual = self.backbone(x)
        if object_one_hot.dim() == 1:
            object_one_hot = object_one_hot.unsqueeze(0)
        parts = [visual, object_one_hot]
        if self.use_texture and self.texture_mlp is not None:
            if texture is None:
                raise ValueError("Нужен вектор текстуры")
            parts.append(self.texture_mlp(texture))
        fused = torch.cat(parts, dim=1)
        hidden = self.shared(fused)
        return {key: self.heads[head_key(key)](hidden) for key in self.attr_keys}

    @classmethod
    def from_vocab(
        cls,
        vocab: dict[str, list[str]],
        **kwargs,
    ) -> FeatureClassifierNet:
        specs = {key: len(labels) for key, labels in vocab.items()}
        return cls(attribute_specs=specs, **kwargs)
