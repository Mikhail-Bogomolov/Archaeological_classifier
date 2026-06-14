"""
Обучение Сети 1 (классификатор объекта) на датасете Канск 2023.

Запуск:
    python -m app.ml.train_classifier
    python -m app.ml.train_classifier --epochs 20 --batch-size 16
    python -m app.ml.train_classifier --verify-only
    python -m app.ml.train_classifier --dry-run

Предобработка для Сети 1 намеренно простая:
  - EXIF-поворот
  - RandomResizedCrop(224) + горизонтальный флип + ColorJitter при обучении
  - CenterCrop(224) при валидации
  - Нормализация по ImageNet mean/std
Сложный CV-пайплайн (маска + Canny) не нужен для классификации класса объекта
и слишком медленен для больших фото.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

try:
    from app.ml.preprocess import load_classifier_rgb, item_key_from_filename
except ImportError as e:
    raise SystemExit(
        "Нет Pillow. Установите: pip install Pillow"
    ) from e

try:
    import openpyxl
except ImportError as e:
    raise SystemExit(
        "Нет openpyxl. Установите: pip install openpyxl"
    ) from e

from app.ml.config import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    KANSK_PHOTOS_DIR,
    KANSK_TABLES_DIR,
    MODELS_DIR,
    OBJECT_CLASSES,
    OBJECT_MODEL_FILE,
)
from app.ml.models import ObjectClassifierNet


# --------------------------------------------------------------------------- #
# Трансформации                                                                #
# --------------------------------------------------------------------------- #

def build_train_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_val_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# --------------------------------------------------------------------------- #
# Датасет                                                                      #
# --------------------------------------------------------------------------- #

class KanskDataset(Dataset):
    """
    Читает all_classes.xlsx из data/perdataset/kansk_2023/tables/ и загружает
    соответствующие фото из photos/.

    Колонки в Excel: sample_id, image_path, класс, ...признаки...
    Поле image_path содержит 'photos/уд85_X-Y_ракурс.jpg'.
    """

    CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(OBJECT_CLASSES)}

    def __init__(
        self,
        photos_dir: str | Path = KANSK_PHOTOS_DIR,
        tables_dir: str | Path = KANSK_TABLES_DIR,
        transform: Optional[transforms.Compose] = None,
        split: str = "train",
        val_ratio: float = 0.15,
        seed: int = 42,
    ):
        self.photos_dir = Path(photos_dir)
        self.transform = transform
        self.split = split

        rows = self._load_rows(Path(tables_dir) / "all_classes.xlsx")
        rows = self._filter_existing(rows)
        if not rows:
            raise FileNotFoundError(
                f"Не найдено ни одного фото из таблицы. "
                f"Проверьте {tables_dir}/all_classes.xlsx и {photos_dir}/"
            )

        # Стратифицированное разбиение по классу
        train_rows, val_rows = self._stratified_split(rows, val_ratio, seed)
        self.samples = train_rows if split == "train" else val_rows

        cls_counts = Counter(r["class_idx"] for r in self.samples)
        labels = sorted(cls_counts)
        print(f"[KanskDataset] {split}: {len(self.samples)} фото | "
              + " | ".join(f"{OBJECT_CLASSES[i]}={cls_counts[i]}" for i in labels))

    # ------------------------------------------------------------------ #

    def _load_rows(self, xlsx_path: Path) -> list[dict]:
        if not xlsx_path.is_file():
            raise FileNotFoundError(f"Не найдена таблица: {xlsx_path}")
        wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)

        result: list[dict] = []
        seen_paths: set[str] = set()
        for ws in wb.worksheets:
            rows_iter = ws.iter_rows(values_only=True)
            header_row = next(rows_iter, None)
            if not header_row:
                continue
            headers = [str(h).strip().lower() if h is not None else "" for h in header_row]

            def col(name: str) -> int | None:
                for i, h in enumerate(headers):
                    if name in h:
                        return i
                return None

            img_col = col("image_path")
            cls_col = col("класс")
            if img_col is None or cls_col is None:
                continue

            for row in rows_iter:
                img_rel = str(row[img_col]).strip() if row[img_col] else ""
                cls_name = str(row[cls_col]).strip().lower() if row[cls_col] else ""
                if not img_rel or cls_name not in self.CLASS_TO_IDX:
                    continue
                img_path = self.photos_dir / Path(img_rel).name
                key = str(img_path).lower()
                if key in seen_paths:
                    continue
                seen_paths.add(key)
                result.append({
                    "path": img_path,
                    "class_idx": self.CLASS_TO_IDX[cls_name],
                    "item_key": item_key_from_filename(img_path),
                })
        return result

    def _filter_existing(self, rows: list[dict]) -> list[dict]:
        ok = [r for r in rows if r["path"].is_file()]
        missing = len(rows) - len(ok)
        if missing:
            print(f"[KanskDataset] Предупреждение: {missing} фото не найдено на диске, пропускаем.")
        return ok

    @staticmethod
    def _stratified_split(
        rows: list[dict], val_ratio: float, seed: int
    ) -> tuple[list[dict], list[dict]]:
        """Разбиение по предметам (1-1, 1-2…), а не по отдельным ракурсам."""
        rng = random.Random(seed)
        by_class_items: dict[int, dict[str, list[dict]]] = {}
        for r in rows:
            by_class_items.setdefault(r["class_idx"], {}).setdefault(r["item_key"], []).append(r)

        train, val = [], []
        for items in by_class_items.values():
            keys = list(items.keys())
            rng.shuffle(keys)
            n_val = max(1, int(len(keys) * val_ratio))
            for key in keys[:n_val]:
                val.extend(items[key])
            for key in keys[n_val:]:
                train.extend(items[key])
        return train, val

    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        img = load_classifier_rgb(path=item["path"])
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(item["class_idx"], dtype=torch.long)

    def class_weights(self) -> torch.Tensor:
        """Веса для WeightedRandomSampler (обратно пропорционально частоте класса)."""
        counts = Counter(r["class_idx"] for r in self.samples)
        n_total = len(self.samples)
        weights = []
        for item in self.samples:
            c = item["class_idx"]
            weights.append(n_total / (len(counts) * counts[c]))
        return torch.tensor(weights, dtype=torch.float)


# --------------------------------------------------------------------------- #
# Обучение                                                                     #
# --------------------------------------------------------------------------- #

def train_one_epoch(
    model: ObjectClassifierNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(
    model: ObjectClassifierNet,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Возвращает (loss, accuracy)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item()
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    acc = correct / max(total, 1)
    return total_loss / max(len(loader), 1), acc


# --------------------------------------------------------------------------- #
# Точка входа                                                                  #
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение Сети 1 (классификатор объекта)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dry-run", action="store_true", help="Не сохранять веса")
    parser.add_argument("--verify-only", action="store_true", help="Только проверить датасет и архитектуру")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Classes ({len(OBJECT_CLASSES)}): {OBJECT_CLASSES}")

    # --- Датасет ---
    train_ds = KanskDataset(transform=build_train_transforms(), split="train")
    val_ds = KanskDataset(transform=build_val_transforms(), split="val")

    if args.verify_only:
        x, y = train_ds[0]
        print(f"Sample tensor: {tuple(x.shape)}, label: {y.item()} ({OBJECT_CLASSES[y.item()]})")
        model = ObjectClassifierNet().to(device)
        out = model(x.unsqueeze(0).to(device))
        print(f"Model output shape: {tuple(out.shape)}  OK")
        return

    # Взвешенный сэмплер для балансировки классов
    sampler = WeightedRandomSampler(
        weights=train_ds.class_weights(),
        num_samples=len(train_ds),
        replacement=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # --- Модель ---
    model = ObjectClassifierNet().to(device)

    # Fine-tune: размораживаем backbone постепенно
    # Первые N эпох — только голова (classifier), потом — всё
    FREEZE_EPOCHS = 5
    for param in model.backbone.parameters():
        param.requires_grad = False
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr * 5,
    )

    # Веса в loss пропорциональны обратной частоте класса (дополнительно к сэмплеру)
    counts = Counter(r["class_idx"] for r in train_ds.samples)
    n_total = len(train_ds)
    loss_weights = torch.tensor(
        [n_total / (len(OBJECT_CLASSES) * counts.get(i, 1)) for i in range(len(OBJECT_CLASSES))],
        dtype=torch.float,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=loss_weights)

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        # Разморозка backbone после FREEZE_EPOCHS
        if epoch == FREEZE_EPOCHS + 1:
            for param in model.backbone.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - FREEZE_EPOCHS
            )
            print(f"  [epoch {epoch}] Backbone разморожен, lr={args.lr}")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, device)

        if epoch > FREEZE_EPOCHS:
            scheduler.step()

        marker = " ← лучший" if val_acc > best_val_acc else ""
        print(
            f"epoch {epoch:3d}/{args.epochs}"
            f"  train_loss={train_loss:.4f}"
            f"  val_loss={val_loss:.4f}"
            f"  val_acc={val_acc:.3f}{marker}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            if not args.dry_run:
                out_dir = Path(MODELS_DIR)
                out_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), out_dir / OBJECT_MODEL_FILE)

    print(f"\nЛучшая точность: {best_val_acc:.3f} на эпохе {best_epoch}")
    if args.dry_run:
        print("dry-run: веса не сохранены")
    else:
        print(f"Веса сохранены: {MODELS_DIR}/{OBJECT_MODEL_FILE}")

    # Per-class accuracy на валидации
    _print_per_class_accuracy(model, val_ds, device)


@torch.no_grad()
def _print_per_class_accuracy(
    model: ObjectClassifierNet,
    dataset: KanskDataset,
    device: torch.device,
) -> None:
    model.eval()
    correct: dict[int, int] = {i: 0 for i in range(len(OBJECT_CLASSES))}
    total: dict[int, int] = {i: 0 for i in range(len(OBJECT_CLASSES))}
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    for x, y in loader:
        x = x.to(device)
        preds = model(x).argmax(dim=1).cpu()
        for pred, label in zip(preds.tolist(), y.tolist()):
            total[label] += 1
            if pred == label:
                correct[label] += 1
    print("\nТочность по классам:")
    for i, cls in enumerate(OBJECT_CLASSES):
        n = total[i]
        c = correct[i]
        bar = "█" * int((c / max(n, 1)) * 20)
        print(f"  {cls:<20} {c:3d}/{n:3d}  {c/max(n,1):.2%}  {bar}")


if __name__ == "__main__":
    main()
