"""Train/val/test split по item_key и аудит утечек."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

SplitName = Literal["train", "val", "test"]

DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15
DEFAULT_SPLIT_SEED = 42


@dataclass
class SplitStats:
    train_items: int = 0
    val_items: int = 0
    test_items: int = 0
    train_photos: int = 0
    val_photos: int = 0
    test_photos: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "train_items": self.train_items,
            "val_items": self.val_items,
            "test_items": self.test_items,
            "train_photos": self.train_photos,
            "val_photos": self.val_photos,
            "test_photos": self.test_photos,
        }


@dataclass
class LeakageReport:
    duplicate_paths: list[str] = field(default_factory=list)
    item_key_collisions: list[dict[str, str]] = field(default_factory=list)
    cross_split_item_keys: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.duplicate_paths and not self.cross_split_item_keys


def stratified_item_split(
    rows: list[dict],
    *,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = DEFAULT_SPLIT_SEED,
) -> tuple[list[dict], list[dict], list[dict], SplitStats]:
    """Разбивка по item_key внутри каждого класса: train / val / test."""
    rng = random.Random(seed)
    by_class_items: dict[int, dict[str, list[dict]]] = {}
    for row in rows:
        by_class_items.setdefault(row["class_idx"], {}).setdefault(row["item_key"], []).append(row)

    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []
    stats = SplitStats()

    for items in by_class_items.values():
        keys = list(items.keys())
        rng.shuffle(keys)
        n = len(keys)

        if n == 1:
            train.extend(items[keys[0]])
            stats.train_items += 1
            stats.train_photos += len(items[keys[0]])
            continue

        if n == 2:
            val.extend(items[keys[0]])
            train.extend(items[keys[1]])
            stats.val_items += 1
            stats.train_items += 1
            stats.val_photos += len(items[keys[0]])
            stats.train_photos += len(items[keys[1]])
            continue

        n_test = max(1, int(round(n * test_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        while n_test + n_val >= n:
            if n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            else:
                break

        test_keys = keys[:n_test]
        val_keys = keys[n_test : n_test + n_val]
        train_keys = keys[n_test + n_val :]

        for key in test_keys:
            test.extend(items[key])
            stats.test_items += 1
            stats.test_photos += len(items[key])
        for key in val_keys:
            val.extend(items[key])
            stats.val_items += 1
            stats.val_photos += len(items[key])
        for key in train_keys:
            train.extend(items[key])
            stats.train_items += 1
            stats.train_photos += len(items[key])

    return train, val, test, stats


def rows_for_split(
    rows: list[dict],
    split: SplitName,
    *,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = DEFAULT_SPLIT_SEED,
) -> list[dict]:
    train, val, test, _ = stratified_item_split(
        rows,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    if split == "train":
        return train
    if split == "val":
        return val
    return test


def export_split_manifest(
    rows: list[dict],
    out_path: str | Path,
    *,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = DEFAULT_SPLIT_SEED,
) -> SplitStats:
    train, val, test, stats = stratified_item_split(
        rows,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    def _pack(split_rows: list[dict]) -> list[dict]:
        by_item: dict[str, list[str]] = defaultdict(list)
        for row in split_rows:
            by_item[row["item_key"]].append(Path(row["path"]).name)
        return [
            {"item_key": key, "photos": sorted(names)}
            for key, names in sorted(by_item.items())
        ]

    payload = {
        "seed": seed,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "stats": stats.as_dict(),
        "splits": {
            "train": _pack(train),
            "val": _pack(val),
            "test": _pack(test),
        },
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def audit_leakage(
    rows: list[dict],
    *,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = DEFAULT_SPLIT_SEED,
) -> LeakageReport:
    report = LeakageReport()

    path_counts = Counter(str(r["path"]).lower() for r in rows)
    report.duplicate_paths = [p for p, c in path_counts.items() if c > 1]

    by_key: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_key[row["item_key"]].add(Path(row["path"]).stem.lower())
    for item_key, stems in sorted(by_key.items()):
        prefixes = {s.rsplit("_", 1)[0] if "_" in s else s for s in stems}
        if len(prefixes) > 1 and len(stems) > 1:
            report.item_key_collisions.append(
                {
                    "item_key": item_key,
                    "stems": ", ".join(sorted(stems)[:5]),
                    "note": "разные stem-префиксы под одним item_key",
                }
            )

    train, val, test, _ = stratified_item_split(
        rows,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    train_keys = {r["item_key"] for r in train}
    val_keys = {r["item_key"] for r in val}
    test_keys = {r["item_key"] for r in test}
    report.cross_split_item_keys = sorted(
        (train_keys & val_keys) | (train_keys & test_keys) | (val_keys & test_keys)
    )

    if report.duplicate_paths:
        report.warnings.append(f"Дубликаты путей: {len(report.duplicate_paths)}")
    if report.item_key_collisions:
        report.warnings.append(
            f"Подозрительные item_key: {len(report.item_key_collisions)}"
        )
    if report.cross_split_item_keys:
        report.warnings.append(
            f"item_key в нескольких сплитах: {len(report.cross_split_item_keys)}"
        )

    return report
