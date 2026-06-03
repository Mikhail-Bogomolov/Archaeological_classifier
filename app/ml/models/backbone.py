"""MobileNet V2 / V3 backbone (torchvision)."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


def create_mobilenet_backbone(
    version: str = "v3_small",
    pretrained: bool = False,
) -> tuple[nn.Module, int]:
    """
    version: v2 | v3_small | v3_large
    Возвращает (feature_extractor, out_channels).
    """
    version = version.lower()
    if version == "v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        net = models.mobilenet_v2(weights=weights)
        feat_dim = net.classifier[1].in_features
        net.classifier = nn.Identity()
        return net, feat_dim

    if version == "v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        net = models.mobilenet_v3_large(weights=weights)
    else:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        net = models.mobilenet_v3_small(weights=weights)

    feat_dim = net.classifier[0].in_features
    net.classifier = nn.Identity()
    return net, feat_dim
