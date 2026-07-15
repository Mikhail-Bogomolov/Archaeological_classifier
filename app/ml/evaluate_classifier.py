"""
Бенчмарк сети 1: accuracy, precision, recall, F1, confusion matrix.

Запуск:
    python -m app.ml.evaluate_classifier
    python -m app.ml.evaluate_classifier --split test --json-out reports/object_test.json
    python -m app.ml.evaluate_classifier --split val --calibrate
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from app.ml.augmentations import build_val_transforms
from app.ml.calibration import expected_calibration_error
from app.ml.config import (
    MODELS_DIR,
    OBJECT_CLASSES,
    OBJECT_MODEL_FILE,
    USE_TEXTURE_FEATURES,
)
from app.ml.evaluation_metrics import (
    compute_benchmark_report,
    print_benchmark_report,
    resolve_report_image_paths,
    save_benchmark_report,
    save_confusion_heatmap,
    save_per_class_bars,
)
from app.ml.models import ObjectClassifierNet
from app.ml.splits import DEFAULT_SPLIT_SEED, DEFAULT_TEST_RATIO, DEFAULT_VAL_RATIO
from app.ml.train_classifier import KanskDataset, _collate_batch
from app.ml.training_config import DEFAULT_INFERENCE, load_state_dict


@torch.no_grad()
def _run_inference(
    model: ObjectClassifierNet,
    loader: DataLoader,
    ds: KanskDataset,
    device: torch.device,
    use_texture: bool,
) -> tuple[list[int], list[int], list[float], list[tuple[str, str, str, float]]]:
    y_true: list[int] = []
    y_pred: list[int] = []
    confidences: list[float] = []
    wrong_examples: list[tuple[str, str, str, float]] = []
    idx_offset = 0

    for batch in loader:
        if use_texture:
            batch_x, batch_tex, batch_y = batch
            batch_x = batch_x.to(device)
            batch_tex = batch_tex.to(device)
            logits = model(batch_x, batch_tex)
        else:
            batch_x, batch_y = batch
            batch_x = batch_x.to(device)
            logits = model(batch_x)

        probs = torch.softmax(logits, dim=-1)
        conf, preds = probs.max(dim=-1)

        for i in range(batch_y.size(0)):
            true_i = int(batch_y[i].item())
            pred_i = int(preds[i].item())
            conf_f = float(conf[i].item())
            sample = ds.samples[idx_offset + i]
            path = Path(sample["path"]).name

            y_true.append(true_i)
            y_pred.append(pred_i)
            confidences.append(conf_f)
            if pred_i != true_i:
                wrong_examples.append(
                    (path, OBJECT_CLASSES[true_i], OBJECT_CLASSES[pred_i], conf_f)
                )
        idx_offset += batch_y.size(0)

    return y_true, y_pred, confidences, wrong_examples


@torch.no_grad()
def _item_level_accuracy(
    model: ObjectClassifierNet,
    loader: DataLoader,
    ds: KanskDataset,
    device: torch.device,
    use_texture: bool,
) -> tuple[int, int]:
    by_item: dict[str, list[bool]] = defaultdict(list)
    idx_offset = 0
    for batch in loader:
        if use_texture:
            batch_x, batch_tex, batch_y = batch
            preds = model(batch_x.to(device), batch_tex.to(device)).argmax(dim=-1).cpu()
        else:
            batch_x, batch_y = batch
            preds = model(batch_x.to(device)).argmax(dim=-1).cpu()
        for i in range(batch_y.size(0)):
            sample = ds.samples[idx_offset + i]
            ok = int(preds[i].item()) == int(batch_y[i].item())
            by_item[sample["item_key"]].append(ok)
        idx_offset += batch_y.size(0)

    item_correct = sum(all(results) for results in by_item.values())
    return item_correct, len(by_item)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Бенчмарк классификатора объектов")
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="test — holdout, не используется при early stopping (рекомендуется для отчёта)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Ограничить число фото (0 = все)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO)
    parser.add_argument(
        "--json-out",
        type=str,
        default="",
        help="Сохранить отчёт в JSON (например reports/object_test.json)",
    )
    parser.add_argument(
        "--heatmap-out",
        type=str,
        default="",
        help="PNG confusion matrix (по умолчанию рядом с json или reports/object_<split>.png)",
    )
    parser.add_argument(
        "--no-heatmap",
        action="store_true",
        help="Не сохранять PNG визуализацию",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Посчитать ECE и предложить порог low-confidence (на указанном split)",
    )
    args = parser.parse_args()

    if args.split != "test":
        print(
            f"Внимание: метрики на split={args.split!r}. "
            "Для честной оценки используйте --split test."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = Path(MODELS_DIR) / OBJECT_MODEL_FILE
    if not weights.is_file():
        raise SystemExit(f"Нет весов: {weights}. Сначала обучите модель.")

    ds = KanskDataset(
        transform=build_val_transforms(),
        split=args.split,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    if args.limit > 0:
        ds.samples = ds.samples[: args.limit]

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=_collate_batch
    )
    model = ObjectClassifierNet(use_texture=USE_TEXTURE_FEATURES).to(device)
    raw_ckpt = torch.load(weights, map_location=device, weights_only=False)
    state, meta = load_state_dict(raw_ckpt)
    has_texture = any(k.startswith("texture_mlp") for k in state)
    if USE_TEXTURE_FEATURES and not has_texture:
        print("Внимание: в файле весов нет текстуры, считаем только по фото")
        model = ObjectClassifierNet(use_texture=False).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    use_texture = model.use_texture and model.texture_mlp is not None

    if meta.get("training"):
        tr = meta["training"]
        print(
            f"Чекпоинт: best_epoch={meta.get('best_epoch')}, "
            f"selection={tr.get('selection_metric')}, seed={tr.get('split_seed')}"
        )

    y_true, y_pred, confidences, wrong_examples = _run_inference(
        model, loader, ds, device, use_texture
    )
    item_correct, item_total = _item_level_accuracy(model, loader, ds, device, use_texture)

    correct_conf = [c for t, p, c in zip(y_true, y_pred, confidences) if t == p]
    wrong_conf = [c for t, p, c in zip(y_true, y_pred, confidences) if t != p]

    extra: dict = {
        "item_accuracy": item_correct / max(item_total, 1),
        "item_correct": item_correct,
        "item_total": item_total,
        "mean_confidence": sum(confidences) / max(len(confidences), 1),
        "mean_confidence_correct": (
            sum(correct_conf) / len(correct_conf) if correct_conf else None
        ),
        "mean_confidence_wrong": (
            sum(wrong_conf) / len(wrong_conf) if wrong_conf else None
        ),
        "split_seed": args.seed,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
    }

    if args.calibrate:
        cal = expected_calibration_error(y_true, y_pred, confidences)
        extra["calibration"] = cal.as_dict()
        print(
            f"\nКалибровка ({args.split}): ECE={cal.ece:.3f}, "
            f"предложенный порог предупреждения={cal.suggested_threshold:.2f} "
            f"(сейчас в коде: {DEFAULT_INFERENCE.object_low_conf_threshold})"
        )

    report = compute_benchmark_report(
        y_true,
        y_pred,
        OBJECT_CLASSES,
        task="object_classification",
        split=args.split,
        extra=extra,
    )
    print_benchmark_report(report)

    if report.extra:
        print(
            f"\nПредметы (все ракурсы верны): "
            f"{item_correct}/{item_total} = {report.extra['item_accuracy']:.1%}"
        )
        if confidences:
            print(f"Средняя уверенность: {report.extra['mean_confidence']:.1%}")
        if correct_conf:
            print(f"  при правильном ответе: {report.extra['mean_confidence_correct']:.1%}")
        if wrong_conf:
            print(f"  при ошибке:          {report.extra['mean_confidence_wrong']:.1%}")

    if wrong_examples:
        print("\nПримеры ошибок (файл | истина -> предсказание | уверенность):")
        for path, true_c, pred_c, conf_f in wrong_examples[:15]:
            print(f"  {path}: {true_c} -> {pred_c} ({conf_f:.0%})")

    if args.json_out:
        out_path = save_benchmark_report(report, args.json_out)
        print(f"\nОтчёт сохранён: {out_path}")

    if not args.no_heatmap:
        heat_path, bars_path = resolve_report_image_paths(
            args.json_out or None,
            args.heatmap_out or None,
            default_stem=f"object_{args.split}",
        )
        saved_heat = save_confusion_heatmap(report, heat_path)
        print(f"Heatmap: {saved_heat}")
        saved_bars = save_per_class_bars(report, bars_path)
        if saved_bars:
            print(f"Bars:    {saved_bars}")
        elif bars_path:
            print("Bars: пропущены (установите matplotlib: pip install matplotlib)")


if __name__ == "__main__":
    main()
