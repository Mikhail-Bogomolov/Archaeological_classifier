"""
Обучение классификатора типа объекта (Канск 2023).

Запуск:
    python -m app.ml.train_classifier
    python -m app.ml.train_classifier --epochs 50 --batch-size 16
    python -m app.ml.train_classifier --verify-only
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

try:
    from app.ml.preprocess import load_classifier_rgb, item_key_from_filename
except ImportError as e:
    raise SystemExit("Нет Pillow. Установите: pip install Pillow") from e

try:
    import openpyxl
except ImportError as e:
    raise SystemExit("Нет openpyxl. Установите: pip install openpyxl") from e

from app.ml.config import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    KANSK_PHOTOS_DIR,
    KANSK_TABLES_DIR,
    MODELS_DIR,
    OBJECT_CLASSES,
    OBJECT_MODEL_FILE,
    USE_TEXTURE_FEATURES,
)
from app.ml.models import ObjectClassifierNet
from app.ml.texture_features import extract_texture_vector



def build_train_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.65, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25),
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



class KanskDataset(Dataset):
    CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(OBJECT_CLASSES)}

    def __init__(
        self,
        photos_dir: str | Path = KANSK_PHOTOS_DIR,
        tables_dir: str | Path = KANSK_TABLES_DIR,
        transform: Optional[transforms.Compose] = None,
        split: str = "train",
        val_ratio: float = 0.15,
        seed: int = 42,
        use_texture: bool = USE_TEXTURE_FEATURES,
        cache_texture: bool = True,
    ):
        self.photos_dir = Path(photos_dir)
        self.transform = transform
        self.split = split
        self.use_texture = use_texture
        self.cache_texture = cache_texture
        self._tex_cache: dict[str, np.ndarray] = {}

        rows = self._load_rows(Path(tables_dir) / "all_classes.xlsx")
        rows = self._filter_existing(rows)
        if not rows:
            raise FileNotFoundError(
                f"Не найдено ни одного фото. Проверьте {tables_dir}/all_classes.xlsx"
            )

        train_rows, val_rows = self._stratified_split(rows, val_ratio, seed)
        self.samples = train_rows if split == "train" else val_rows

        cls_counts = Counter(r["class_idx"] for r in self.samples)
        labels = sorted(cls_counts)
        print(
            f"[KanskDataset] {split}: {len(self.samples)} фото | "
            + " | ".join(f"{OBJECT_CLASSES[i]}={cls_counts[i]}" for i in labels)
        )

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
            print(f"[KanskDataset] Предупреждение: {missing} фото не найдено, пропускаем.")
        return ok

    @staticmethod
    def _stratified_split(
        rows: list[dict], val_ratio: float, seed: int
    ) -> tuple[list[dict], list[dict]]:
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

    def _load_sample(self, path: Path) -> tuple:
        key = str(path)
        tex = None
        if self.use_texture and self.cache_texture and key in self._tex_cache:
            tex = self._tex_cache[key]

        pil = load_classifier_rgb(path=path)
        if self.use_texture:
            if tex is None:
                tex = extract_texture_vector(pil)
                if self.cache_texture:
                    self._tex_cache[key] = tex
        else:
            tex = None
        return pil, tex

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        pil, tex_vec = self._load_sample(item["path"])
        try:
            if self.transform:
                x = self.transform(pil)
            else:
                x = transforms.ToTensor()(pil)
        finally:
            pil.close()
        y = torch.tensor(item["class_idx"], dtype=torch.long)
        if self.use_texture and tex_vec is not None:
            tex = torch.tensor(tex_vec, dtype=torch.float32)
            return x, tex, y
        return x, y

    def class_weights(self) -> torch.Tensor:
        counts = Counter(r["class_idx"] for r in self.samples)
        n_total = len(self.samples)
        weights = []
        for sample in self.samples:
            c = sample["class_idx"]
            weights.append(n_total / (len(counts) * counts[c]))
        return torch.tensor(weights, dtype=torch.float)


def _collate_batch(batch):
    if len(batch[0]) == 3:
        xs, texs, ys = zip(*batch)
        return torch.stack(xs), torch.stack(texs), torch.stack(ys)
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys)



def train_one_epoch(
    model: ObjectClassifierNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_texture: bool,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        if use_texture:
            x, tex, y = batch
            x, tex, y = x.to(device), tex.to(device), y.to(device)
            logits = model(x, tex)
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
        optimizer.zero_grad()
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(
    model: ObjectClassifierNet,
    loader: DataLoader,
    device: torch.device,
    use_texture: bool,
) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in loader:
        if use_texture:
            x, tex, y = batch
            x, tex, y = x.to(device), tex.to(device), y.to(device)
            logits = model(x, tex)
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
        total_loss += criterion(logits, y).item()
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return total_loss / max(len(loader), 1), correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение Сети 1 (классификатор объекта)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=12, help="Early stopping")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-texture", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    use_texture = USE_TEXTURE_FEATURES and not args.no_texture
    pretrained = not args.no_pretrained

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Classes: {OBJECT_CLASSES}")
    print(f"pretrained={pretrained}, texture={use_texture}")

    train_ds = KanskDataset(
        transform=build_train_transforms(), split="train", use_texture=use_texture
    )
    val_ds = KanskDataset(
        transform=build_val_transforms(), split="val", use_texture=use_texture
    )

    if args.verify_only:
        sample = train_ds[0]
        if use_texture:
            x, tex, y = sample
            print(f"Sample: {tuple(x.shape)}, texture {tuple(tex.shape)}, label {y.item()}")
        else:
            x, y = sample
            print(f"Sample: {tuple(x.shape)}, label {y.item()}")
        model = ObjectClassifierNet(pretrained=pretrained, use_texture=use_texture).to(device)
        if use_texture:
            out = model(x.unsqueeze(0).to(device), tex.unsqueeze(0).to(device))
        else:
            out = model(x.unsqueeze(0).to(device))
        print(f"Model output: {tuple(out.shape)}  OK")
        return

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
        collate_fn=_collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_batch,
    )

    model = ObjectClassifierNet(pretrained=pretrained, use_texture=use_texture).to(device)

    FREEZE_EPOCHS = 3 if pretrained else 5
    for param in model.backbone.parameters():
        param.requires_grad = False

    head_params = list(model.head.parameters())
    if model.texture_mlp is not None:
        head_params += list(model.texture_mlp.parameters())
    optimizer = torch.optim.AdamW(head_params, lr=args.head_lr)

    counts = Counter(r["class_idx"] for r in train_ds.samples)
    n_total = len(train_ds)
    loss_weights = torch.tensor(
        [n_total / (len(OBJECT_CLASSES) * counts.get(i, 1)) for i in range(len(OBJECT_CLASSES))],
        dtype=torch.float,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=loss_weights, label_smoothing=0.05)

    best_val_acc = 0.0
    best_epoch = 0
    no_improve = 0
    scheduler = None

    print("\nОбучение началось. Первая эпоха может идти несколько минут.\n")

    for epoch in range(1, args.epochs + 1):
        if epoch == FREEZE_EPOCHS + 1:
            for param in model.backbone.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW([
                {"params": model.backbone.parameters(), "lr": args.lr},
                {"params": model.head.parameters(), "lr": args.head_lr},
            ] + (
                [{"params": model.texture_mlp.parameters(), "lr": args.head_lr}]
                if model.texture_mlp is not None else []
            ))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - FREEZE_EPOCHS
            )
            print(f"  [epoch {epoch}] Донастройка всей модели, lr={args.lr}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, use_texture
        )
        val_loss, val_acc = evaluate(model, val_loader, device, use_texture)

        if scheduler is not None:
            scheduler.step()

        improved = val_acc > best_val_acc
        marker = " <- best" if improved else ""
        print(
            f"epoch {epoch:3d}/{args.epochs}"
            f"  train_loss={train_loss:.4f}"
            f"  val_loss={val_loss:.4f}"
            f"  val_acc={val_acc:.3f}{marker}"
        )

        if improved:
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve = 0
            if not args.dry_run:
                Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), Path(MODELS_DIR) / OBJECT_MODEL_FILE)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping: нет улучшения {args.patience} эпох")
                break

    print(f"\nЛучшая точность: {best_val_acc:.3f} на эпохе {best_epoch}")
    if not args.dry_run:
        print(f"Веса: {MODELS_DIR}/{OBJECT_MODEL_FILE}")
    _print_per_class_accuracy(model, val_ds, device, use_texture)


@torch.no_grad()
def _print_per_class_accuracy(
    model: ObjectClassifierNet,
    dataset: KanskDataset,
    device: torch.device,
    use_texture: bool,
) -> None:
    model.eval()
    correct: dict[int, int] = {i: 0 for i in range(len(OBJECT_CLASSES))}
    total: dict[int, int] = {i: 0 for i in range(len(OBJECT_CLASSES))}
    loader = DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=_collate_batch
    )
    for batch in loader:
        if use_texture:
            x, tex, y = batch
            x, tex = x.to(device), tex.to(device)
            preds = model(x, tex).argmax(dim=1).cpu()
        else:
            x, y = batch
            preds = model(x.to(device)).argmax(dim=1).cpu()
            y = y
        for pred, label in zip(preds.tolist(), y.tolist()):
            total[label] += 1
            if pred == label:
                correct[label] += 1
    print("\nТочность по классам:")
    for i, cls in enumerate(OBJECT_CLASSES):
        n, c = total[i], correct[i]
        bar = "█" * int((c / max(n, 1)) * 20)
        print(f"  {cls:<20} {c:3d}/{n:3d}  {c/max(n,1):.2%}  {bar}")


if __name__ == "__main__":
    main()
