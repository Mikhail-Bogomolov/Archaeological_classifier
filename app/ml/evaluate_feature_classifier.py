"""
Бенчмарк признаков: accuracy / P / R / F1 по признакам и по типам объектов.

    python -m app.ml.evaluate_feature_classifier
    python -m app.ml.evaluate_feature_classifier --split test --json-out reports/feature_test.json
    python -m app.ml.evaluate_feature_classifier --class ножи
"""

from __future__ import annotations

import argparse
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from app.ml.augmentations import build_val_transforms
from app.ml.config import FEATURE_MODEL_FILE, MODELS_DIR, OBJECT_CLASSES
from app.ml.evaluation_metrics import (
    BenchmarkReport,
    compute_benchmark_report,
    save_confusion_heatmap,
)
from app.ml.feature_dataset import KanskFeatureDataset, collate_features
from app.ml.models import FeatureClassifierNet
from app.ml.splits import DEFAULT_SPLIT_SEED, DEFAULT_TEST_RATIO, DEFAULT_VAL_RATIO


def _summarize_by_class(
    per_head_reports: dict[str, BenchmarkReport],
    photos_by_class: dict[str, int],
) -> dict[str, dict[str, Any]]:
    """Метрики признаков по каждому типу объекта."""
    by_class: dict[str, dict[str, Any]] = {}

    for class_name in OBJECT_CLASSES:
        heads_for_class = {
            key.split(":", 1)[1]: rep
            for key, rep in per_head_reports.items()
            if key.startswith(f"{class_name}:")
        }
        if not heads_for_class:
            continue

        class_correct = 0
        class_labels = 0
        macro_f1_values: list[float] = []
        head_payload: dict[str, Any] = {}

        for feat in sorted(heads_for_class):
            rep = heads_for_class[feat]
            head_payload[feat] = rep.to_dict()
            macro_f1_values.append(rep.macro_f1)
            class_correct += sum(rep.confusion[i][i] for i in range(len(rep.labels)))
            class_labels += rep.n_samples

        by_class[class_name] = {
            "n_photos": photos_by_class.get(class_name, 0),
            "n_labels": class_labels,
            "n_heads": len(heads_for_class),
            "overall_accuracy": class_correct / max(class_labels, 1),
            "mean_macro_f1": sum(macro_f1_values) / max(len(macro_f1_values), 1),
            "heads": head_payload,
        }

    return by_class


def _print_by_class_summary(by_class: dict[str, dict[str, Any]], split: str) -> None:
    print(f"\n=== По типам объектов ({split}) ===")
    print(
        f"  {'класс':<22} {'фото':>5} {'меток':>6} "
        f"{'acc':>7} {'macro-F1':>9}"
    )
    for class_name in OBJECT_CLASSES:
        row = by_class.get(class_name)
        if not row:
            continue
        print(
            f"  {class_name:<22} {row['n_photos']:5d} {row['n_labels']:6d} "
            f"{row['overall_accuracy']:6.1%} {row['mean_macro_f1']:8.1%}"
        )

    for class_name in OBJECT_CLASSES:
        row = by_class.get(class_name)
        if not row:
            continue
        print(
            f"\n--- {class_name} "
            f"(фото={row['n_photos']}, меток={row['n_labels']}, "
            f"acc={row['overall_accuracy']:.1%}, macro-F1={row['mean_macro_f1']:.1%}) ---"
        )
        print(f"  {'признак':<22} {'меток':>6} {'acc':>7} {'macro-F1':>9}")
        for feat, head in sorted(row["heads"].items()):
            print(
                f"  {feat:<22} {head['n_samples']:6d} "
                f"{head['accuracy']:6.1%} {head['macro_f1']:8.1%}"
            )


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Бенчмарк сети признаков")
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="test — holdout для честной оценки",
    )
    parser.add_argument(
        "--class",
        dest="object_class",
        default="",
        help="Только один тип объекта (например: ножи)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO)
    parser.add_argument(
        "--json-out",
        type=str,
        default="",
        help="Сохранить сводный отчёт в JSON",
    )
    parser.add_argument(
        "--heatmap-dir",
        type=str,
        default="",
        help="Папка для PNG heatmap по головам (по умолчанию рядом с json или reports/feature_heatmaps)",
    )
    parser.add_argument(
        "--no-heatmap",
        action="store_true",
        help="Не сохранять PNG визуализацию",
    )
    parser.add_argument(
        "--no-head-detail",
        action="store_true",
        help="Не печатать детальный отчёт по каждому значению признака",
    )
    args = parser.parse_args()

    if args.split != "test":
        print(
            f"Внимание: метрики на split={args.split!r}. "
            "Для честной оценки используйте --split test."
        )

    if args.object_class and args.object_class not in OBJECT_CLASSES:
        raise SystemExit(
            f"Неизвестный класс {args.object_class!r}. "
            f"Допустимо: {', '.join(OBJECT_CLASSES)}"
        )

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
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_features)

    photos_by_class = Counter(
        OBJECT_CLASSES[s["class_idx"]] for s in ds.samples
    )

    attr_keys = ds.attr_keys
    if args.object_class:
        attr_keys = [k for k in attr_keys if k.startswith(f"{args.object_class}:")]
        if not attr_keys:
            raise SystemExit(f"В vocab нет голов для класса {args.object_class!r}")

    y_true_by_attr: dict[str, list[int]] = defaultdict(list)
    y_pred_by_attr: dict[str, list[int]] = defaultdict(list)

    for x, one_hot, _class_idx, targets, texture in loader:
        x = x.to(device)
        one_hot = one_hot.to(device)
        targets = targets.to(device)
        if model.use_texture and texture is not None:
            logits = model(x, one_hot, texture.to(device))
        else:
            logits = model(x, one_hot)

        for i, key in enumerate(ds.attr_keys):
            if key not in attr_keys:
                continue
            y = targets[:, i]
            mask = y >= 0
            if mask.sum() == 0:
                continue
            preds = logits[key][mask].argmax(dim=1)
            y_true_by_attr[key].extend(y[mask].tolist())
            y_pred_by_attr[key].extend(preds.tolist())

    print(f"\n=== Бенчмарк: feature_classification ({args.split}) ===")
    print(f"texture={use_texture}, heads={len(attr_keys)}")
    if args.object_class:
        print(f"фильтр класса: {args.object_class}")

    per_head_reports: dict[str, BenchmarkReport] = {}
    macro_f1_values: list[float] = []
    total_correct = 0
    total_labels = 0

    for key in sorted(attr_keys):
        y_true = y_true_by_attr.get(key, [])
        y_pred = y_pred_by_attr.get(key, [])
        if not y_true:
            continue

        labels = vocab.get(key, [])
        if not labels:
            continue

        head_report = compute_benchmark_report(
            y_true,
            y_pred,
            labels,
            task=f"feature:{key}",
            split=args.split,
        )
        per_head_reports[key] = head_report
        macro_f1_values.append(head_report.macro_f1)
        total_correct += sum(head_report.confusion[i][i] for i in range(len(labels)))
        total_labels += len(y_true)

        if not args.no_head_detail:
            print(f"\n--- {key} (n={head_report.n_samples}) ---")
            print(
                f"Acc={head_report.accuracy:.1%}  "
                f"macro P/R/F1={head_report.macro_precision:.1%}/"
                f"{head_report.macro_recall:.1%}/{head_report.macro_f1:.1%}  "
                f"weighted F1={head_report.weighted_f1:.1%}"
            )
            print(f"  {'значение':<28} {'sup':>5} {'prec':>7} {'rec':>7} {'f1':>7}")
            for row in head_report.per_class:
                if row.support == 0 and row.predicted == 0:
                    continue
                label = row.label[:28]
                print(
                    f"  {label:<28} {row.support:5d} "
                    f"{row.precision:6.1%} {row.recall:6.1%} {row.f1:6.1%}"
                )

    by_class = _summarize_by_class(per_head_reports, dict(photos_by_class))
    if args.object_class:
        by_class = {k: v for k, v in by_class.items() if k == args.object_class}
    _print_by_class_summary(by_class, args.split)

    overall_acc = total_correct / max(total_labels, 1)
    mean_macro_f1 = sum(macro_f1_values) / max(len(macro_f1_values), 1)
    print(
        f"\n=== ИТОГО по всем признакам ===\n"
        f"Меток: {total_labels}  |  "
        f"Accuracy: {overall_acc:.1%}  |  "
        f"Средний macro-F1 по головам: {mean_macro_f1:.1%}"
    )

    if args.json_out:
        payload = {
            "task": "feature_classification",
            "split": args.split,
            "use_texture": use_texture,
            "overall_accuracy": overall_acc,
            "mean_macro_f1": mean_macro_f1,
            "total_labels": total_labels,
            "by_class": by_class,
            "heads": {key: rep.to_dict() for key, rep in per_head_reports.items()},
        }
        if args.object_class:
            payload["filter_class"] = args.object_class
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            __import__("json").dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nОтчёт сохранён: {out}")

    if not args.no_heatmap and per_head_reports:
        if args.heatmap_dir:
            heat_dir = Path(args.heatmap_dir)
        elif args.json_out:
            heat_dir = Path(args.json_out).with_suffix("").parent / (
                Path(args.json_out).stem + "_heatmaps"
            )
        else:
            heat_dir = Path("reports") / f"feature_heatmaps_{args.split}"
        heat_dir.mkdir(parents=True, exist_ok=True)
        index: dict[str, str] = {}
        for key, head_report in per_head_reports.items():
            safe = re.sub(r"[^\w\-]+", "_", key, flags=re.UNICODE).strip("_")
            digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
            fname = f"{safe}_{digest}.png" if safe else f"{digest}.png"
            path = heat_dir / fname
            save_confusion_heatmap(head_report, path, title=key)
            index[fname] = key
        index_path = heat_dir / "index.json"
        index_path.write_text(
            __import__("json").dumps(index, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Heatmaps: {heat_dir} ({len(index)} файлов)")


if __name__ == "__main__":
    main()
