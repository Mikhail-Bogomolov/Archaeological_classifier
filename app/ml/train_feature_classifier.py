"""
Обучение сети 2 — признаки по типу объекта.

Запуск:
    python -m app.ml.train_feature_classifier --verify-only
    python -m app.ml.train_feature_classifier --epochs 30 --batch-size 16
    python -m app.ml.train_feature_classifier --no-texture
    python -m app.ml.train_feature_classifier --no-boost --epochs 40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from app.ml.augmentations import build_train_transforms_mild, build_val_transforms
from app.ml.config import (
    FEATURE_HEAD_MINORITY_BOOST,
    FEATURE_MINORITY_BOOST_FACTOR,
    FEATURE_MODEL_FILE,
    FEATURE_SCHEMA,
    MODELS_DIR,
    OBJECT_CLASSES,
    USE_TEXTURE_FEATURES,
)
from app.ml.feature_dataset import KanskFeatureDataset, collate_features
from collections import Counter

from app.ml.feature_vocab import scan_tables
from app.ml.losses import class_weights_from_counts
from app.ml.models import FeatureClassifierNet


def build_head_class_weights(
    vocab: dict[str, list[str]],
    attr_keys: list[str],
    device: torch.device,
    *,
    use_minority_boost: bool = True,
) -> dict[str, torch.Tensor]:
    """Веса CE по головам: inverse-freq + опциональный буст редких классов."""
    table_counts = scan_tables()
    weights: dict[str, torch.Tensor] = {}
    for key in attr_keys:
        labels = vocab[key]
        counts = [table_counts.get(key, Counter()).get(lab, 0) for lab in labels]
        w = class_weights_from_counts(counts, device)
        if use_minority_boost and key in FEATURE_HEAD_MINORITY_BOOST:
            positive = [c for c in counts if c > 0]
            if positive:
                mean = sum(positive) / len(positive)
                for i, c in enumerate(counts):
                    if 0 < c < mean * 0.75:
                        w[i] *= FEATURE_MINORITY_BOOST_FACTOR
        weights[key] = w
    return weights


def compute_loss(
    logits: dict[str, torch.Tensor],
    targets: torch.Tensor,
    attr_keys: list[str],
    head_weights: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, int]:
    total = torch.tensor(0.0, device=targets.device)
    n_heads = 0
    for i, key in enumerate(attr_keys):
        head_logits = logits[key]
        y = targets[:, i]
        mask = y >= 0
        if mask.sum() == 0:
            continue
        weight = head_weights.get(key) if head_weights else None
        total = total + F.cross_entropy(head_logits[mask], y[mask], weight=weight)
        n_heads += 1
    if n_heads == 0:
        return total, 0
    return total / n_heads, n_heads


def _model_forward(
    model: FeatureClassifierNet,
    batch: tuple,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    x, one_hot, _class_idx, targets, texture = batch
    x = x.to(device)
    one_hot = one_hot.to(device)
    targets = targets.to(device)
    if model.use_texture and texture is not None:
        logits = model(x, one_hot, texture.to(device))
    else:
        logits = model(x, one_hot)
    return logits, targets


def train_one_epoch(
    model: FeatureClassifierNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    attr_keys: list[str],
    head_weights: dict[str, torch.Tensor] | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    steps = 0
    for batch in loader:
        optimizer.zero_grad()
        logits, targets = _model_forward(model, batch, device)
        loss, n_heads = compute_loss(
            logits, targets, attr_keys, head_weights=head_weights
        )
        if n_heads == 0:
            continue
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        steps += 1
    return total_loss / max(steps, 1)


@torch.no_grad()
def evaluate(
    model: FeatureClassifierNet,
    loader: DataLoader,
    device: torch.device,
    attr_keys: list[str],
    head_weights: dict[str, torch.Tensor] | None = None,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    steps = 0

    for batch in loader:
        logits, targets = _model_forward(model, batch, device)
        loss, n_heads = compute_loss(
            logits, targets, attr_keys, head_weights=head_weights
        )
        if n_heads == 0:
            continue
        total_loss += loss.item()
        steps += 1

        for i, key in enumerate(attr_keys):
            y = targets[:, i]
            mask = y >= 0
            if mask.sum() == 0:
                continue
            preds = logits[key][mask].argmax(dim=1)
            correct += (preds == y[mask]).sum().item()
            total += mask.sum().item()

    avg_loss = total_loss / max(steps, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def save_checkpoint(
    path: Path,
    model: FeatureClassifierNet,
    vocab: dict[str, list[str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab": vocab,
            "attribute_specs": model.attribute_specs,
            "attr_keys": model.attr_keys,
            "use_texture": model.use_texture,
        },
        path,
    )
    json_path = path.with_suffix(".vocab.json")
    json_path.write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def print_vocab_summary(
    vocab: dict[str, list[str]], *, use_minority_boost: bool = True
) -> None:
    print(f"\nПризнаков для обучения: {len(vocab)}")
    for key in sorted(vocab):
        labels = vocab[key]
        preview = ", ".join(labels[:6])
        if len(labels) > 6:
            preview += ", …"
        boost = (
            " [boost]"
            if use_minority_boost and key in FEATURE_HEAD_MINORITY_BOOST
            else ""
        )
        print(f"  {key}: {len(labels)} знач. — {preview}{boost}")

    expected = sum(len(FEATURE_SCHEMA.get(c, [])) for c in OBJECT_CLASSES)
    missing = [
        f"{c}:{f}"
        for c in OBJECT_CLASSES
        for f in FEATURE_SCHEMA.get(c, [])
        if f"{c}:{f}" not in vocab
    ]
    if missing:
        print(f"\nПредупреждение: не в vocab ({len(missing)}): {', '.join(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение сети 2 (признаки)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-texture", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--no-boost",
        action="store_true",
        help="отключить буст редких классов (FEATURE_HEAD_MINORITY_BOOST)",
    )
    args = parser.parse_args()

    pretrained = not args.no_pretrained
    use_texture = USE_TEXTURE_FEATURES and not args.no_texture
    use_minority_boost = not args.no_boost
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, texture={use_texture}")

    vocab = KanskFeatureDataset.build_vocab()
    if not vocab:
        raise SystemExit(
            "Не удалось построить словарь признаков. "
            "Проверьте data/dataset/tables/*.xlsx"
        )
    print_vocab_summary(vocab, use_minority_boost=use_minority_boost)

    train_ds = KanskFeatureDataset(
        vocab=vocab,
        transform=build_train_transforms_mild(),
        split="train",
        use_texture=use_texture,
    )
    val_ds = KanskFeatureDataset(
        vocab=vocab,
        transform=build_val_transforms(),
        split="val",
        use_texture=use_texture,
    )
    attr_keys = train_ds.attr_keys
    head_weights = build_head_class_weights(
        vocab, attr_keys, device, use_minority_boost=use_minority_boost
    )
    if use_minority_boost:
        boost_count = sum(1 for k in attr_keys if k in FEATURE_HEAD_MINORITY_BOOST)
        print(
            f"\nБустинг редких классов: {boost_count} голов, "
            f"множитель={FEATURE_MINORITY_BOOST_FACTOR}"
        )
    else:
        print("\nБустинг редких классов: выключен (--no-boost)")

    model = FeatureClassifierNet.from_vocab(
        vocab, pretrained=pretrained, use_texture=use_texture
    ).to(device)

    if args.verify_only:
        sample = train_ds[0]
        x, one_hot = sample[0], sample[1]
        tex = sample[4] if len(sample) == 5 else None
        if model.use_texture and tex is not None:
            out = model(
                x.unsqueeze(0).to(device),
                one_hot.unsqueeze(0).to(device),
                tex.unsqueeze(0).to(device),
            )
        else:
            out = model(x.unsqueeze(0).to(device), one_hot.unsqueeze(0).to(device))
        print(f"Sample class={sample[2]}, heads={len(out)}")
        for key in attr_keys[:3]:
            print(f"  {key}: {tuple(out[key].shape)}")
        print("OK")
        return

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_features,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_features,
    )

    FREEZE_EPOCHS = 3 if pretrained else 2
    for param in model.backbone.parameters():
        param.requires_grad = False
    head_params = list(model.shared.parameters()) + list(model.heads.parameters())
    if model.texture_mlp is not None:
        head_params += list(model.texture_mlp.parameters())
    optimizer = torch.optim.AdamW(head_params, lr=args.head_lr)

    best_val_acc = 0.0
    best_epoch = 0
    no_improve = 0
    scheduler = None
    weights_path = Path(MODELS_DIR) / FEATURE_MODEL_FILE

    print("\nОбучение сети 2 началось.\n")

    for epoch in range(1, args.epochs + 1):
        if epoch == FREEZE_EPOCHS + 1:
            for param in model.backbone.parameters():
                param.requires_grad = True
            param_groups = [
                {"params": model.backbone.parameters(), "lr": args.lr},
                {"params": model.shared.parameters(), "lr": args.head_lr},
                {"params": model.heads.parameters(), "lr": args.head_lr},
            ]
            if model.texture_mlp is not None:
                param_groups.append(
                    {"params": model.texture_mlp.parameters(), "lr": args.head_lr}
                )
            optimizer = torch.optim.AdamW(param_groups)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(args.epochs - FREEZE_EPOCHS, 1)
            )
            print(f"  [epoch {epoch}] Разморозка backbone, lr={args.lr}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, attr_keys, head_weights
        )
        val_loss, val_acc = evaluate(
            model, val_loader, device, attr_keys, head_weights
        )

        if scheduler is not None:
            scheduler.step()

        improved = val_acc > best_val_acc
        marker = " <- best" if improved else ""
        print(
            f"epoch {epoch:3d}/{args.epochs}"
            f"  train_loss={train_loss:.4f}"
            f"  val_loss={val_loss:.4f}"
            f"  val_attr_acc={val_acc:.3f}{marker}"
        )

        if improved:
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve = 0
            if not args.dry_run:
                save_checkpoint(weights_path, model, vocab)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping: нет улучшения {args.patience} эпох")
                break

    print(f"\nЛучшая точность по признакам: {best_val_acc:.3f} (эпоха {best_epoch})")
    if not args.dry_run:
        print(f"Веса: {weights_path}")
        print(f"Словарь: {weights_path.with_suffix('.vocab.json')}")


if __name__ == "__main__":
    main()
