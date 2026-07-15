"""Аудит датасета: сплиты, утечки, manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.config import KANSK_TABLES_DIR, OBJECT_CLASSES, SPLIT_MANIFEST_PATH
from app.ml.splits import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_TEST_RATIO,
    DEFAULT_VAL_RATIO,
    audit_leakage,
    export_split_manifest,
)
from app.ml.train_classifier import KanskDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Аудит train/val/test и утечек item_key")
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO)
    parser.add_argument(
        "--manifest-out",
        type=str,
        default=SPLIT_MANIFEST_PATH,
        help="Куда сохранить JSON со списком item_key по сплитам",
    )
    args = parser.parse_args()

    probe = KanskDataset(
        split="train", seed=args.seed, val_ratio=args.val_ratio, test_ratio=args.test_ratio
    )
    rows = list(probe.samples)
    for split_name in ("val", "test"):
        part = KanskDataset(
            split=split_name,
            seed=args.seed,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )
        rows.extend(part.samples)

    by_path = {str(r["path"]).lower(): r for r in rows}
    all_rows = list(by_path.values())

    print(f"Всего фото после merge сплитов: {len(all_rows)}")
    report = audit_leakage(
        all_rows,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    if report.ok:
        print("Утечки не обнаружены.")
    else:
        print("Предупреждения:")
        for w in report.warnings:
            print(f"  - {w}")
        if report.item_key_collisions[:5]:
            print("\nПримеры подозрительных item_key:")
            for item in report.item_key_collisions[:5]:
                print(
                    f"  {item['item_key']}: {item.get('note', '')} [{item.get('stems', '')}]"
                )

    stats = export_split_manifest(
        all_rows,
        args.manifest_out,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(f"\nManifest: {args.manifest_out}")
    print(
        f"items train/val/test: {stats.train_items}/{stats.val_items}/{stats.test_items} | "
        f"photos: {stats.train_photos}/{stats.val_photos}/{stats.test_photos}"
    )
    print(f"Классы: {', '.join(OBJECT_CLASSES)}")
    print(f"Таблицы: {KANSK_TABLES_DIR}")


if __name__ == "__main__":
    main()
