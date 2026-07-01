"""
Оценка сети 2 на val: точность по признакам.

    python -m app.ml.evaluate_feature_classifier
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from app.ml.config import FEATURE_MODEL_FILE, MODELS_DIR
from app.ml.feature_dataset import KanskFeatureDataset, collate_features
from app.ml.models import FeatureClassifierNet
from app.ml.train_classifier import build_val_transforms


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Оценка сети 2")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = Path(MODELS_DIR) / FEATURE_MODEL_FILE
    if not weights.is_file():
        raise SystemExit(f"Нет весов: {weights}. Сначала: python -m app.ml.train_feature_classifier")

    ckpt = torch.load(weights, map_location=device, weights_only=False)
    vocab = ckpt["vocab"]
    state = ckpt["state_dict"]
    use_texture = ckpt.get("use_texture", any(k.startswith("texture_mlp") for k in state))
    model = FeatureClassifierNet.from_vocab(
        vocab, pretrained=False, use_texture=use_texture
    ).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()

    ds = KanskFeatureDataset(
        vocab=vocab,
        transform=build_val_transforms(),
        split=args.split,
        use_texture=use_texture,
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_features)

    attr_keys = ds.attr_keys
    correct: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)

    for x, one_hot, _class_idx, targets, texture in loader:
        x = x.to(device)
        one_hot = one_hot.to(device)
        targets = targets.to(device)
        if model.use_texture and texture is not None:
            logits = model(x, one_hot, texture.to(device))
        else:
            logits = model(x, one_hot)

        for i, key in enumerate(attr_keys):
            y = targets[:, i]
            mask = y >= 0
            if mask.sum() == 0:
                continue
            preds = logits[key][mask].argmax(dim=1)
            correct[key] += (preds == y[mask]).sum().item()
            total[key] += mask.sum().item()

    print(f"\nТочность по признакам ({args.split}), texture={use_texture}:")
    all_c = sum(correct.values())
    all_t = sum(total.values())
    for key in sorted(attr_keys):
        c, t = correct[key], total[key]
        if t == 0:
            continue
        print(f"  {key:<35} {c:4d}/{t:4d}  {c/t:.1%}")
    print(f"\n  ИТОГО: {all_c}/{all_t}  {all_c/max(all_t,1):.1%}")


if __name__ == "__main__":
    main()
