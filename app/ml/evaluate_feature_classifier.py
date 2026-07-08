"""
Бенчмарк сети 2: accuracy, precision, recall, F1 по каждому признаку.

    python -m app.ml.evaluate_feature_classifier
    python -m app.ml.evaluate_feature_classifier --json-out reports/features_val.json
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from app.ml.config import FEATURE_MODEL_FILE, MODELS_DIR
from app.ml.evaluation_metrics import (
    BenchmarkReport,
    compute_benchmark_report,
    save_benchmark_report,
)
from app.ml.feature_dataset import KanskFeatureDataset, collate_features
from app.ml.models import FeatureClassifierNet
from app.ml.augmentations import build_val_transforms


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Бенчмарк сети признаков")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument(
        "--json-out",
        type=str,
        default="",
        help="Сохранить сводный отчёт в JSON",
    )
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

        for i, key in enumerate(attr_keys):
            y = targets[:, i]
            mask = y >= 0
            if mask.sum() == 0:
                continue
            preds = logits[key][mask].argmax(dim=1)
            y_true_by_attr[key].extend(y[mask].tolist())
            y_pred_by_attr[key].extend(preds.tolist())

    print(f"\n=== Бенчмарк: feature_classification ({args.split}) ===")
    print(f"texture={use_texture}, heads={len(attr_keys)}")

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
            "heads": {key: rep.to_dict() for key, rep in per_head_reports.items()},
        }
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            __import__("json").dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nОтчёт сохранён: {out}")


if __name__ == "__main__":
    main()
