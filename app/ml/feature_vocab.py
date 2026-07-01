"""Словари значений признаков из Excel Канск 2023."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import openpyxl

from app.ml.config import FEATURE_SCHEMA, KANSK_TABLES_DIR, OBJECT_CLASSES

NOT_SPECIFIED = frozenset({"не указано", "", "nan", "none", "—", "-"})


def attribute_key(class_name: str, feature_name: str) -> str:
    return f"{class_name}:{feature_name}"


def head_key(attr_key: str) -> str:
    return attr_key.replace(":", "__")


def from_head_key(head: str) -> str:
    return head.replace("__", ":")


def normalize_feature_value(raw: object) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip().lower()
    text = re.sub(r"\s+", " ", text)
    if text in NOT_SPECIFIED:
        return None
    if text in {"бронза", "bronze"}:
        return "бронза"
    if text in {"железо", "iron"}:
        return "железо"
    return text


def scan_tables(tables_dir: str | Path = KANSK_TABLES_DIR) -> dict[str, Counter]:
    tables_dir = Path(tables_dir)
    counts: dict[str, Counter] = {}

    for class_name in OBJECT_CLASSES:
        xlsx = tables_dir / f"{class_name}.xlsx"
        if not xlsx.is_file():
            continue
        wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
        ws = wb.active
        header_row = next(ws.iter_rows(min_row=1, max_row=1, values_only=True), None)
        if not header_row:
            continue
        headers = [str(h).strip() if h is not None else "" for h in header_row]
        col_index = {name: i for i, name in enumerate(headers)}

        for feature_name in FEATURE_SCHEMA.get(class_name, []):
            if feature_name not in col_index:
                continue
            key = attribute_key(class_name, feature_name)
            counts.setdefault(key, Counter())
            idx = col_index[feature_name]
            for row in ws.iter_rows(min_row=2, values_only=True):
                if idx >= len(row):
                    continue
                norm = normalize_feature_value(row[idx])
                if norm:
                    counts[key][norm] += 1
    return counts


def build_vocab(
    tables_dir: str | Path = KANSK_TABLES_DIR,
    min_count: int = 1,
    min_classes: int = 2,
) -> dict[str, list[str]]:
    """attr_key → отсортированный список классов (минимум min_classes значения)."""
    counts = scan_tables(tables_dir)
    vocab: dict[str, list[str]] = {}

    for key, counter in counts.items():
        labels = [label for label, c in counter.items() if c >= min_count]
        labels.sort(key=lambda x: (-counter[x], x))
        if len(labels) >= min_classes:
            vocab[key] = labels
    return vocab


def value_to_index(vocab: dict[str, list[str]], attr_key: str, raw: object) -> int:
    norm = normalize_feature_value(raw)
    if norm is None:
        return -1
    labels = vocab.get(attr_key)
    if not labels:
        return -1
    if norm in labels:
        return labels.index(norm)
    if "другое" in labels:
        return labels.index("другое")
    return -1


def ensure_other_bucket(vocab: dict[str, list[str]], tables_dir: str | Path) -> dict[str, list[str]]:
    """Редкие значения сводим в «другое», если встречаются в таблице."""
    counts = scan_tables(tables_dir)
    out: dict[str, list[str]] = {}

    for key, labels in vocab.items():
        counter = counts.get(key, Counter())
        frequent = set(labels)
        merged = list(labels)
        rare_seen = any(
            normalize_feature_value(v) not in frequent and normalize_feature_value(v)
            for v, _ in counter.items()
        )
        if rare_seen and "другое" not in merged:
            merged.append("другое")
        out[key] = merged
    return out
