"""
Шаблон разметки и экспорт отсканированных объектов в Excel/CSV.
"""

from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.ml.config import (
    FEATURE_SCHEMA,
    MARKUP_TEMPLATE_PATH,
    OBJECT_CLASSES,
    markup_columns_for_class,
    markup_feature_names,
)

_CONFIDENCE_SUFFIX = re.compile(r"\s*\(\d+%\)\s*$")
MARKUP_TEMPLATE_CSV_ZIP_PATH = "data/templates/markup_template_csv.zip"


def _strip_confidence(value: str) -> str:
    return _CONFIDENCE_SUFFIX.sub("", value).strip()


def _normalize_features(features: list[str] | str | None) -> list[str]:
    if features is None:
        return []
    if isinstance(features, str):
        try:
            parsed = json.loads(features)
            return parsed if isinstance(parsed, list) else [features]
        except json.JSONDecodeError:
            return [features]
    return list(features)


def _feature_values_for_export(features: list[str] | None, object_class: str) -> list[str]:
    """Разбирает сохранённые признаки в 5 колонок по именам из FEATURE_SCHEMA."""
    feat_names = markup_feature_names(object_class)
    values = {name: "" for name in feat_names}
    if not features:
        return [values[name] for name in feat_names]

    for line in features:
        text = str(line).strip()
        if not text:
            continue
        lower = text.lower()
        for feat_name in sorted(feat_names, key=len, reverse=True):
            prefix = f"{feat_name.lower()}:"
            if lower.startswith(prefix):
                raw = text.split(":", 1)[-1].strip()
                values[feat_name] = _strip_confidence(raw)
                break

    return [values[name] for name in feat_names]


def _row_from_object(obj: dict[str, Any]) -> list:
    obj_class = obj.get("category") or OBJECT_CLASSES[0]
    feats = _normalize_features(obj.get("features"))
    feat_cols = _feature_values_for_export(feats, obj_class)
    return [obj.get("id", ""), obj.get("name", ""), *feat_cols]


def _group_objects_by_class(objects: list[dict[str, Any]] | None) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    if objects:
        for obj in objects:
            sheet_name = obj.get("category") or OBJECT_CLASSES[0]
            grouped[sheet_name].append(obj)
    if not grouped:
        for cls in OBJECT_CLASSES:
            grouped[cls] = []
    return grouped


def _safe_sheet_filename(sheet_name: str) -> str:
    return sheet_name.replace("/", "_").replace("\\", "_")


def build_markup_workbook_bytes(objects: list[dict[str, Any]] | None = None) -> bytes:
    from openpyxl import Workbook

    wb = Workbook()
    wb.remove(wb.active)

    for sheet_name, rows in _group_objects_by_class(objects).items():
        title = sheet_name[:31]
        ws = wb.create_sheet(title=title)
        ws.append(markup_columns_for_class(sheet_name))
        for obj in rows:
            ws.append(_row_from_object(obj))

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def build_markup_template_csv_zip_bytes() -> bytes:
    """Пустой шаблон: ZIP с CSV по одному на каждый тип объекта."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for cls in OBJECT_CLASSES:
            csv_buf = io.StringIO()
            writer = csv.writer(csv_buf, lineterminator="\n")
            writer.writerow(markup_columns_for_class(cls))
            safe_name = _safe_sheet_filename(cls)
            zf.writestr(f"{safe_name}.csv", csv_buf.getvalue().encode("utf-8-sig"))
    return buf.getvalue()


def ensure_markup_template_on_disk() -> Path:
    path = Path(MARKUP_TEMPLATE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(build_markup_workbook_bytes(objects=None))
    return path


def ensure_markup_csv_template_on_disk() -> Path:
    path = Path(MARKUP_TEMPLATE_CSV_ZIP_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(build_markup_template_csv_zip_bytes())
    return path


def export_objects_xlsx(objects: list[dict[str, Any]]) -> bytes:
    """Экспорт БД: лист на каждый тип объекта."""
    return build_markup_workbook_bytes(objects)


def export_objects_csv_zip(objects: list[dict[str, Any]]) -> bytes:
    """ZIP с CSV-файлами (по одному на тип объекта)."""
    grouped = _group_objects_by_class(objects)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for sheet_name, rows in grouped.items():
            csv_buf = io.StringIO()
            writer = csv.writer(csv_buf, lineterminator="\n")
            writer.writerow(markup_columns_for_class(sheet_name))
            for obj in rows:
                writer.writerow(_row_from_object(obj))
            safe_name = _safe_sheet_filename(sheet_name)
            zf.writestr(f"{safe_name}.csv", csv_buf.getvalue().encode("utf-8-sig"))
    return buf.getvalue()
