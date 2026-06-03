"""
Скелет обучения (запускать после подготовки датасета).

Ожидаемая структура:
  data/dataset/
    images/<id>.jpg
    labels.csv   # columns: image_id, object_class, feature_керамика:орнамент_есть, ...

Пока датасета нет — скрипт только проверяет конфигурацию и создаёт пустые веса-заглушки.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from app.ml.config import MODELS_DIR, NUM_FEATURES, NUM_OBJECT_CLASSES, OBJECT_CLASSES
from app.ml.models import FeatureClassifierNet, ObjectClassifierNet


class PlaceholderDataset(Dataset):
    """Заглушка: 4 синтетических батча для проверки пайплайна обучения."""

    def __init__(self, length: int = 32, size: int = 224):
        self.length = length
        self.size = size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        x = torch.randn(3, self.size, self.size)
        y_obj = idx % NUM_OBJECT_CLASSES
        y_feat = torch.zeros(NUM_FEATURES)
        if idx % 3 == 0:
            y_feat[idx % NUM_FEATURES] = 1.0
        return x, y_obj, y_feat


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
        y_obj_oh = torch.nn.functional.one_hot(y_obj, num_classes=NUM_OBJECT_CLASSES).float().to(device)
        y_feat = y_feat.to(device)
        optimizer.zero_grad()
        logits = model(x, y_obj_oh)
        loss = criterion(logits, y_feat)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение археологического классификатора")
    parser.add_argument("--epochs", type=int, default=1, help="Эпох (на заглушке)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="Только проверка, без сохранения")
    args = parser.parse_args()

    data_root = Path("data/dataset")
    if not data_root.exists():
        print(
            "Датасет не найден: data/dataset/\n"
            "Создайте папку и labels.csv, либо используйте --dry-run для проверки кода."
        )
        if not args.dry_run:
            return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(PlaceholderDataset(), batch_size=args.batch_size, shuffle=True)

    object_net = ObjectClassifierNet().to(device)
    feature_net = FeatureClassifierNet().to(device)

    opt_obj = torch.optim.AdamW(object_net.parameters(), lr=1e-4)
    opt_feat = torch.optim.AdamW(feature_net.parameters(), lr=1e-4)

    for epoch in range(args.epochs):
        lo = train_one_epoch_object(object_net, loader, opt_obj, device)
        lf = train_one_epoch_feature(feature_net, loader, opt_feat, device)
        print(f"epoch {epoch + 1}: loss_object={lo:.4f} loss_feature={lf:.4f}")

    if args.dry_run:
        print("dry-run: веса не сохранены")
        return

    out = Path(MODELS_DIR)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(object_net.state_dict(), out / "object_classifier.pt")
    torch.save(feature_net.state_dict(), out / "feature_classifier.pt")
    print(f"Сохранено в {out} (обучено на заглушке — замените реальным датасетом)")


if __name__ == "__main__":
    main()
