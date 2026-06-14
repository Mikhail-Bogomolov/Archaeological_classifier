"""
Оценка Сети 1 на val-выборке: точность, матрица ошибок, уверенность.

Запуск:
    python -m app.ml.evaluate_classifier
    python -m app.ml.evaluate_classifier --split val
    python -m app.ml.evaluate_classifier --split train --limit 50
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from app.ml.config import MODELS_DIR, OBJECT_CLASSES, OBJECT_MODEL_FILE
from app.ml.models import ObjectClassifierNet
from app.ml.train_classifier import KanskDataset, build_val_transforms


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Оценка классификатора объектов")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--limit", type=int, default=0, help="Ограничить число фото (0 = все)")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = Path(MODELS_DIR) / OBJECT_MODEL_FILE
    if not weights.is_file():
        raise SystemExit(f"Нет весов: {weights}. Сначала обучите модель.")

    ds = KanskDataset(transform=build_val_transforms(), split=args.split)
    if args.limit > 0:
        ds.samples = ds.samples[: args.limit]

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = ObjectClassifierNet().to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.eval()

    confusion: dict[tuple[int, int], int] = Counter()
    confidences: list[float] = []
    correct_conf: list[float] = []
    wrong_conf: list[float] = []
    wrong_examples: list[tuple[str, str, str, float]] = []

    correct = 0
    total = 0
    idx_offset = 0
    for batch_x, batch_y in loader:
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

            confusion[(true_i, pred_i)] += 1
            confidences.append(conf_f)
            total += 1
            if pred_i == true_i:
                correct += 1
                correct_conf.append(conf_f)
            else:
                wrong_conf.append(conf_f)
                wrong_examples.append(
                    (path, OBJECT_CLASSES[true_i], OBJECT_CLASSES[pred_i], conf_f)
                )
        idx_offset += batch_y.size(0)

    acc = correct / max(total, 1)
    print(f"\n=== Оценка на {args.split} ({total} фото) ===")
    print(f"Точность: {correct}/{total} = {acc:.1%}")
    if confidences:
        print(f"Средняя уверенность: {sum(confidences)/len(confidences):.1%}")
    if correct_conf:
        print(f"  при правильном ответе: {sum(correct_conf)/len(correct_conf):.1%}")
    if wrong_conf:
        print(f"  при ошибке:          {sum(wrong_conf)/len(wrong_conf):.1%}")

    print("\nМатрица (строка = истина, столбец = предсказание):")
    header = " " * 18 + " | " + " | ".join(f"{c[:8]:>8}" for c in OBJECT_CLASSES)
    print(header)
    for ti, true_cls in enumerate(OBJECT_CLASSES):
        row = [f"{true_cls[:16]:>16}"]
        for pi in range(len(OBJECT_CLASSES)):
            row.append(f"{confusion.get((ti, pi), 0):8d}")
        print(" | ".join(row))

    # По предметам
    by_item: dict[str, list[bool]] = defaultdict(list)
    idx_offset = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        preds = model(batch_x).argmax(dim=-1).cpu()
        for i in range(batch_y.size(0)):
            sample = ds.samples[idx_offset + i]
            ok = int(preds[i].item()) == int(batch_y[i].item())
            by_item[sample["item_key"]].append(ok)
        idx_offset += batch_y.size(0)

    item_correct = sum(all(r for r in v) for v in by_item.values())
    item_total = len(by_item)
    print(f"\nПредметы (все ракурсы верны): {item_correct}/{item_total} = {item_correct/max(item_total,1):.1%}")

    if wrong_examples:
        print("\nПримеры ошибок (файл | истина -> предсказание | уверенность):")
        for path, true_c, pred_c, conf_f in wrong_examples[:15]:
            print(f"  {path}: {true_c} -> {pred_c} ({conf_f:.0%})")

    print("\nПодсказка: 3/6 вручную ≈ 50%. При val accuracy ~59% это в пределах нормы.")
    print("Запустите на val целиком — это честнее, чем 6 случайных фото.")


if __name__ == "__main__":
    main()
