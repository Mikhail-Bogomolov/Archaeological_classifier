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
    KANSK_PHOTOS_DIR,
    KANSK_TABLES_DIR,
    MODELS_DIR,
    OBJECT_CLASSES,
    OBJECT_MODEL_FILE,
    TRAINING_LOG_DIR,
    USE_TEXTURE_FEATURES,
)
from app.ml.augmentations import build_train_transforms, build_val_transforms
from app.ml.dataset_stats import RowLoadStats
from app.ml.losses import build_object_loss_weights, focal_cross_entropy
from app.ml.models import ObjectClassifierNet
from app.ml.splits import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_TEST_RATIO,
    DEFAULT_VAL_RATIO,
    rows_for_split,
)
from app.ml.texture_features import extract_texture_vector
from app.ml.training_config import (
    DEFAULT_OBJECT_TRAINING,
    ObjectTrainingConfig,
    checkpoint_payload,
)
from app.ml.experiment_log import EpochLogger


DEFAULT_FOCUS_CLASS = DEFAULT_OBJECT_TRAINING.focus_class

class KanskDataset(Dataset):
    CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(OBJECT_CLASSES)}

    def __init__(
        self,
        photos_dir: str | Path = KANSK_PHOTOS_DIR,
        tables_dir: str | Path = KANSK_TABLES_DIR,
        transform: Optional[transforms.Compose] = None,
        split: str = "train",
        val_ratio: float = DEFAULT_VAL_RATIO,
        test_ratio: float = DEFAULT_TEST_RATIO,
        seed: int = DEFAULT_SPLIT_SEED,
        use_texture: bool = USE_TEXTURE_FEATURES,
        cache_texture: bool = True,
    ):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train|val|test, got {split!r}")
        self.photos_dir = Path(photos_dir)
        self.transform = transform
        self.split = split
        self.use_texture = use_texture
        self.cache_texture = cache_texture
        self._tex_cache: dict[str, np.ndarray] = {}

        rows, load_stats = self._load_rows(Path(tables_dir) / "all_classes.xlsx")
        rows = self._filter_existing(rows)
        if not rows:
            raise FileNotFoundError(
                f"Не найдено ни одного фото. Проверьте {tables_dir}/all_classes.xlsx"
            )

        self.samples = rows_for_split(
            rows,
            split,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        self.load_stats = load_stats

        cls_counts = Counter(r["class_idx"] for r in self.samples)
        labels = sorted(cls_counts)
        print(
            f"[KanskDataset] {split}: {len(self.samples)} фото | "
            + " | ".join(f"{OBJECT_CLASSES[i]}={cls_counts[i]}" for i in labels)
        )
        for line in load_stats.summary_lines("[KanskDataset] "):
            if "пропущ" in line.lower() or "  - " in line:
                print(line)

    def _load_rows(self, xlsx_path: Path) -> tuple[list[dict], RowLoadStats]:
        if not xlsx_path.is_file():
            raise FileNotFoundError(f"Не найдена таблица: {xlsx_path}")
        wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
        stats = RowLoadStats()
        result: list[dict] = []
        seen_paths: set[str] = set()
        for ws in wb.worksheets:
            stats.worksheets_seen += 1
            rows_iter = ws.iter_rows(values_only=True)
            header_row = next(rows_iter, None)
            if not header_row:
                stats.worksheets_skipped += 1
                stats.skip("пустой лист")
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
                stats.worksheets_skipped += 1
                stats.skip("нет image_path или класс")
                continue

            for row in rows_iter:
                stats.rows_total += 1
                img_rel = str(row[img_col]).strip() if row[img_col] else ""
                cls_name = str(row[cls_col]).strip().lower() if row[cls_col] else ""
                if not img_rel:
                    stats.skip("пустой image_path")
                    continue
                if cls_name not in self.CLASS_TO_IDX:
                    stats.skip("неизвестный класс")
                    continue
                img_path = self.photos_dir / Path(img_rel).name
                key = str(img_path).lower()
                if key in seen_paths:
                    stats.skip("дубликат пути")
                    continue
                seen_paths.add(key)
                stats.rows_kept += 1
                result.append({
                    "path": img_path,
                    "class_idx": self.CLASS_TO_IDX[cls_name],
                    "item_key": item_key_from_filename(img_path),
                })
        return result, stats

    def _filter_existing(self, rows: list[dict]) -> list[dict]:
        ok = [r for r in rows if r["path"].is_file()]
        missing = len(rows) - len(ok)
        if missing:
            print(f"[KanskDataset] Предупреждение: {missing} фото не найдено, пропускаем.")
        return ok

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

    def class_weights(
        self,
        focus_class_idx: int | None = None,
        focus_boost: float = 1.0,
        minority_boost: float = 1.25,
    ) -> torch.Tensor:
        counts = Counter(r["class_idx"] for r in self.samples)
        n_total = len(self.samples)
        mean = n_total / max(len(counts), 1)
        weights = []
        for sample in self.samples:
            c = sample["class_idx"]
            w = n_total / (len(counts) * counts[c])
            if counts[c] < mean * 0.75 and minority_boost > 1.0:
                w *= minority_boost
            if focus_class_idx is not None and c == focus_class_idx and focus_boost > 1.0:
                w *= focus_boost
            weights.append(w)
        return torch.tensor(weights, dtype=torch.float)


def _mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[perm]
    return mixed_x, y, y[perm], lam


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
    *,
    mixup_alpha: float = 0.0,
    focal_gamma: float = 0.0,
    loss_weights: torch.Tensor | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        if use_texture:
            x, tex, y = batch
            x, tex, y = x.to(device), tex.to(device), y.to(device)
            if mixup_alpha > 0 and x.size(0) > 1:
                x, ya, yb, lam = _mixup_batch(x, y, mixup_alpha)
                logits = model(x, tex)
                if focal_gamma > 0:
                    loss = lam * focal_cross_entropy(
                        logits, ya, weight=loss_weights, gamma=focal_gamma
                    ) + (1.0 - lam) * focal_cross_entropy(
                        logits, yb, weight=loss_weights, gamma=focal_gamma
                    )
                else:
                    loss = lam * criterion(logits, ya) + (1.0 - lam) * criterion(logits, yb)
            else:
                logits = model(x, tex)
                if focal_gamma > 0:
                    loss = focal_cross_entropy(logits, y, weight=loss_weights, gamma=focal_gamma)
                else:
                    loss = criterion(logits, y)
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            if mixup_alpha > 0 and x.size(0) > 1:
                x, ya, yb, lam = _mixup_batch(x, y, mixup_alpha)
                logits = model(x)
                if focal_gamma > 0:
                    loss = lam * focal_cross_entropy(
                        logits, ya, weight=loss_weights, gamma=focal_gamma
                    ) + (1.0 - lam) * focal_cross_entropy(
                        logits, yb, weight=loss_weights, gamma=focal_gamma
                    )
                else:
                    loss = lam * criterion(logits, ya) + (1.0 - lam) * criterion(logits, yb)
            else:
                logits = model(x)
                if focal_gamma > 0:
                    loss = focal_cross_entropy(logits, y, weight=loss_weights, gamma=focal_gamma)
                else:
                    loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate_per_class_recall(
    model: ObjectClassifierNet,
    loader: DataLoader,
    device: torch.device,
    use_texture: bool,
) -> dict[int, float]:
    model.eval()
    correct: dict[int, int] = {i: 0 for i in range(len(OBJECT_CLASSES))}
    total: dict[int, int] = {i: 0 for i in range(len(OBJECT_CLASSES))}
    for batch in loader:
        if use_texture:
            x, tex, y = batch
            x, tex, y = x.to(device), tex.to(device), y.to(device)
            preds = model(x, tex).argmax(dim=1)
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
        for pred, label in zip(preds.tolist(), y.tolist()):
            total[label] += 1
            if pred == label:
                correct[label] += 1
    return {i: correct[i] / max(total[i], 1) for i in range(len(OBJECT_CLASSES))}


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
    parser.add_argument(
        "--focus-class",
        type=str,
        default=DEFAULT_FOCUS_CLASS,
        help="Усилить обучение для класса (по умолчанию: ножи)",
    )
    parser.add_argument(
        "--no-focus-class",
        action="store_true",
        help="Отключить усиление focus-класса",
    )
    parser.add_argument(
        "--focus-boost",
        type=float,
        default=DEFAULT_OBJECT_TRAINING.focus_boost,
        help="Множитель веса и oversampling для focus-класса",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("accuracy", "focus_recall", "balanced"),
        default=DEFAULT_OBJECT_TRAINING.selection_metric,
        help="Критерий сохранения лучших весов",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=0.0,
        help="Focal loss gamma (0 = обычный CE)",
    )
    parser.add_argument("--mixup", action="store_true", help="Включить Mixup (для ножей обычно выкл.)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED, help="Seed сплита train/val/test")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO)
    parser.add_argument(
        "--log-csv",
        type=str,
        default="",
        help=f"CSV-лог эпох (по умолчанию {TRAINING_LOG_DIR}/object_classifier.csv)",
    )
    args = parser.parse_args()

    focus_class_idx: int | None = None
    if not args.no_focus_class:
        focus_name = args.focus_class.strip().lower()
        if focus_name in OBJECT_CLASSES:
            focus_class_idx = OBJECT_CLASSES.index(focus_name)
        else:
            print(f"Предупреждение: неизвестный focus-class '{args.focus_class}', отключаем.")

    use_texture = USE_TEXTURE_FEATURES and not args.no_texture
    pretrained = not args.no_pretrained

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Classes: {OBJECT_CLASSES}")
    print(f"pretrained={pretrained}, texture={use_texture}")
    if focus_class_idx is not None:
        print(
            f"focus_class={OBJECT_CLASSES[focus_class_idx]}, "
            f"boost={args.focus_boost}, selection={args.selection_metric}"
        )

    train_ds = KanskDataset(
        transform=build_train_transforms(),
        split="train",
        use_texture=use_texture,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    val_ds = KanskDataset(
        transform=build_val_transforms(),
        split="val",
        use_texture=use_texture,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
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
        weights=train_ds.class_weights(
            focus_class_idx=focus_class_idx,
            focus_boost=args.focus_boost,
        ),
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

    FREEZE_EPOCHS = (
        DEFAULT_OBJECT_TRAINING.freeze_epochs_pretrained
        if pretrained
        else DEFAULT_OBJECT_TRAINING.freeze_epochs_scratch
    )
    for param in model.backbone.parameters():
        param.requires_grad = False

    head_params = list(model.head.parameters())
    if model.texture_mlp is not None:
        head_params += list(model.texture_mlp.parameters())
    optimizer = torch.optim.AdamW(head_params, lr=args.head_lr)

    counts = [Counter(r["class_idx"] for r in train_ds.samples).get(i, 0) for i in range(len(OBJECT_CLASSES))]
    loss_weights = build_object_loss_weights(
        counts,
        device,
        focus_class_idx=focus_class_idx,
        focus_boost=args.focus_boost,
    )
    criterion = nn.CrossEntropyLoss(weight=loss_weights, label_smoothing=DEFAULT_OBJECT_TRAINING.label_smoothing)

    training_cfg = ObjectTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        head_lr=args.head_lr,
        patience=args.patience,
        mixup_alpha=DEFAULT_OBJECT_TRAINING.mixup_alpha,
        focal_gamma=args.focal_gamma,
        focus_class=args.focus_class if focus_class_idx is not None else "",
        focus_boost=args.focus_boost,
        selection_metric=args.selection_metric,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_seed=args.seed,
        use_texture=use_texture,
        pretrained=pretrained,
    )

    log_path = args.log_csv or str(Path(TRAINING_LOG_DIR) / "object_classifier.csv")
    epoch_logger = EpochLogger(log_path)

    best_score = -1.0
    best_val_acc = 0.0
    best_focus_recall = 0.0
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
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            use_texture,
            mixup_alpha=DEFAULT_OBJECT_TRAINING.mixup_alpha if args.mixup else 0.0,
            focal_gamma=args.focal_gamma,
            loss_weights=loss_weights,
        )
        val_loss, val_acc = evaluate(model, val_loader, device, use_texture)
        per_class_recall = evaluate_per_class_recall(model, val_loader, device, use_texture)
        focus_recall = (
            per_class_recall.get(focus_class_idx, val_acc)
            if focus_class_idx is not None
            else val_acc
        )

        if args.selection_metric == "accuracy":
            score = val_acc
        elif args.selection_metric == "focus_recall":
            score = focus_recall
        else:
            score = 0.5 * val_acc + 0.5 * focus_recall

        if scheduler is not None:
            scheduler.step()

        improved = score > best_score
        marker = " <- best" if improved else ""
        focus_label = (
            f"  {OBJECT_CLASSES[focus_class_idx]}_recall={focus_recall:.3f}"
            if focus_class_idx is not None
            else ""
        )
        print(
            f"epoch {epoch:3d}/{args.epochs}"
            f"  train_loss={train_loss:.4f}"
            f"  val_loss={val_loss:.4f}"
            f"  val_acc={val_acc:.3f}{focus_label}{marker}"
        )

        epoch_logger.log({
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "val_loss": f"{val_loss:.6f}",
            "val_acc": f"{val_acc:.6f}",
            "focus_recall": f"{focus_recall:.6f}",
            "selection_score": f"{score:.6f}",
            "selection_metric": args.selection_metric,
            "seed": args.seed,
        })

        if improved:
            best_score = score
            best_val_acc = val_acc
            best_focus_recall = focus_recall
            best_epoch = epoch
            no_improve = 0
            if not args.dry_run:
                Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
                ckpt = checkpoint_payload(
                    model.state_dict(),
                    model_kind="object_classifier",
                    training=training_cfg,
                    best_epoch=epoch,
                    best_score=score,
                    extra={
                        "val_acc": val_acc,
                        "focus_recall": focus_recall,
                        "focus_class": OBJECT_CLASSES[focus_class_idx] if focus_class_idx is not None else None,
                    },
                )
                torch.save(ckpt, Path(MODELS_DIR) / OBJECT_MODEL_FILE)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping: нет улучшения {args.patience} эпох")
                break

    print(
        f"\nЛучший результат ({args.selection_metric}={best_score:.3f}): "
        f"val_acc={best_val_acc:.3f}, focus_recall={best_focus_recall:.3f} "
        f"на эпохе {best_epoch}"
    )
    if not args.dry_run:
        print(f"Веса: {MODELS_DIR}/{OBJECT_MODEL_FILE}")
        print(f"Лог эпох: {log_path}")
        print(
            "Итоговые метрики смотрите на test-сете (не использовался при обучении):\n"
            "  py -m app.ml.evaluate_classifier --split test --json-out reports/object_test.json"
        )
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
