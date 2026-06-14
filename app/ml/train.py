"""
Пайплайн обучения двух сетей (без датасета — на тестовых фото + заглушках меток).

Запуск:
  python -m app.ml.train --epochs 3
  python -m app.ml.train --verify-only
  python -m app.ml.train --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from app.ml.config import MODELS_DIR, NUM_FEATURES, NUM_OBJECT_CLASSES, OBJECT_CLASSES
from app.ml.image_processing import discover_test_images, process_file
from app.ml.models import FeatureClassifierNet, ObjectClassifierNet
from app.ml.preprocess import five_channel_to_mobilenet_tensor


def verify_architecture(device: torch.device) -> None:
    """Проверка размерностей MobileNet V3 + V2 и one-hot стыковки."""
    batch = 2
    x = torch.randn(batch, 3, 224, 224, device=device)
    obj_net = ObjectClassifierNet().to(device)
    feat_net = FeatureClassifierNet().to(device)

    obj_logits = obj_net(x)
    assert obj_logits.shape == (batch, NUM_OBJECT_CLASSES), obj_logits.shape

    obj_idx = torch.zeros(batch, dtype=torch.long, device=device)
    obj_oh = torch.nn.functional.one_hot(obj_idx, num_classes=NUM_OBJECT_CLASSES).float()
    feat_logits = feat_net(x, obj_oh)
    assert feat_logits.shape == (batch, NUM_FEATURES), feat_logits.shape

    print("Architecture OK:")
    print(f"  object classes: {NUM_OBJECT_CLASSES} {OBJECT_CLASSES}")
    print(f"  feature outputs: {NUM_FEATURES}")
    print(f"  object logits: {tuple(obj_logits.shape)}")
    print(f"  feature logits: {tuple(feat_logits.shape)}")


class TestImageDataset(Dataset):
    """
    Заглушка датасета: test_img_02.jpg, test_img_10.jpg + синтетические метки.
    Вход: (5, 50, 50) после CV; метки: класс объекта + multi-hot признаков.
    """

    def __init__(self, augment_copies: int = 8):
        self.samples: list[Path] = discover_test_images()
        if not self.samples:
            raise FileNotFoundError(
                "Тестовые изображения не найдены (test_img_02.jpg, test_img_10.jpg)."
            )
        self.augment_copies = augment_copies

    def __len__(self) -> int:
        return len(self.samples) * self.augment_copies

    def __getitem__(self, idx: int):
        path = self.samples[idx % len(self.samples)]
        five_ch = process_file(path)
        x = five_channel_to_mobilenet_tensor(five_ch).squeeze(0)

        # Заглушки меток: всегда «объект 1», признаки по индексу
        y_obj = 0
        y_feat = torch.zeros(NUM_FEATURES)
        if idx % 2 == 0 and NUM_FEATURES > 0:
            y_feat[idx % NUM_FEATURES] = 1.0

        return x, torch.tensor(y_obj, dtype=torch.long), y_feat


class SyntheticDataset(Dataset):
    """Резерв, если нет OpenCV/тестовых фото."""

    def __init__(self, length: int = 16, size: int = 224):
        self.length = length
        self.size = size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        x = torch.randn(3, self.size, self.size)
        y_obj = 0
        y_feat = torch.zeros(NUM_FEATURES)
        if idx % 2 == 0 and NUM_FEATURES > 0:
            y_feat[idx % NUM_FEATURES] = 1.0
        return x, torch.tensor(y_obj, dtype=torch.long), y_feat


def train_one_epoch_object(
    model: ObjectClassifierNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total = 0.0
    for x, y_obj, _ in loader:
        x, y_obj = x.to(device), y_obj.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y_obj)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)


def train_one_epoch_feature(
    model: FeatureClassifierNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total = 0.0
    for x, y_obj, y_feat in loader:
        x = x.to(device)
        y_obj_oh = torch.nn.functional.one_hot(
            y_obj, num_classes=NUM_OBJECT_CLASSES
        ).float().to(device)
        y_feat = y_feat.to(device)
        optimizer.zero_grad()
        logits = model(x, y_obj_oh)
        loss = criterion(logits, y_feat)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)


def build_dataloader(batch_size: int) -> tuple[DataLoader, str]:
    try:
        ds = TestImageDataset()
        return DataLoader(ds, batch_size=batch_size, shuffle=True), "test_images"
    except Exception as exc:
        print(f"TestImageDataset недоступен ({exc}), используем SyntheticDataset.")
        return DataLoader(SyntheticDataset(), batch_size=batch_size, shuffle=True), "synthetic"


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение археологического классификатора")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dry-run", action="store_true", help="Обучить, но не сохранять веса")
    parser.add_argument("--verify-only", action="store_true", help="Только проверка архитектуры")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    verify_architecture(device)
    if args.verify_only:
        return

    loader, source = build_dataloader(args.batch_size)
    print(f"Dataset source: {source}, batches: {len(loader)}")

    object_net = ObjectClassifierNet().to(device)
    feature_net = FeatureClassifierNet().to(device)
    opt_obj = torch.optim.AdamW(object_net.parameters(), lr=args.lr)
    opt_feat = torch.optim.AdamW(feature_net.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        lo = train_one_epoch_object(object_net, loader, opt_obj, device)
        lf = train_one_epoch_feature(feature_net, loader, opt_feat, device)
        print(f"epoch {epoch + 1}/{args.epochs}: loss_object={lo:.4f} loss_feature={lf:.4f}")

    if args.dry_run:
        print("dry-run: веса не сохранены")
        return

    out = Path(MODELS_DIR)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(object_net.state_dict(), out / "object_classifier.pt")
    torch.save(feature_net.state_dict(), out / "feature_classifier.pt")
    print(f"Веса сохранены в {out} (заглушки — замените после разметки датасета)")


if __name__ == "__main__":
    main()
