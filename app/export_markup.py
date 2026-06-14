"""
Шаблон разметки и экспорт отсканированных объектов.

Формат: Excel с листами по типу объекта (сеть 1).
На листе колонки: номер, название, признак 1 … признак 5.
"""

from __future__ import annotations

import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.ml.config import FEATURE_SCHEMA, MARKUP_COLUMNS, MARKUP_TEMPLATE_PATH, OBJECT_CLASSES


def _feature_values_for_export(features: list[str] | None, object_class: str) -> list[str]:
    """Разбирает сохранённые признаки в 5 колонок шаблона."""
    schema = FEATURE_SCHEMA.get(object_class, FEATURE_SCHEMA.get(OBJECT_CLASSES[0], []))
    values = [""] * 5
    if not features:
        return values

    for line in features:
        for i, feat_name in enumerate(schema[:5]):
            if feat_name in line:
                if ":" in line:
                    values[i] = line.split(":", 1)[-1].strip()
                else:
                    values[i] = "да"
                break
    return values


def _row_from_object(obj: dict[str, Any]) -> list:
    obj_class = obj.get("category") or OBJECT_CLASSES[0]
    feats = obj.get("features")
    if isinstance(feats, str):
        try:
            feats = json.loads(feats)
        except json.JSONDecodeError:
            feats = [feats]
    feat_cols = _feature_values_for_export(feats if isinstance(feats, list) else None, obj_class)
    return [obj.get("id", ""), obj.get("name", ""), *feat_cols]


def build_markup_workbook_bytes(objects: list[dict[str, Any]] | None = None) -> bytes:
    from openpyxl import Workbook

    wb = Workbook()
    wb.remove(wb.active)

    grouped: dict[str, list[dict]] = defaultdict(list)
    if objects:
        for obj in objects:
            sheet_name = obj.get("category") or OBJECT_CLASSES[0]
            grouped[sheet_name].append(obj)
    if not grouped:
        for cls in OBJECT_CLASSES:
            grouped[cls] = []

    for sheet_name, rows in grouped.items():
        title = sheet_name[:31]
        ws = wb.create_sheet(title=title)
        ws.append(MARKUP_COLUMNS)
        for obj in rows:
            ws.append(_row_from_object(obj))

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def ensure_markup_template_on_disk() -> Path:
    path = Path(MARKUP_TEMPLATE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        path.write_bytes(build_markup_workbook_bytes(objects=None))
    return path


def export_objects_xlsx(objects: list[dict[str, Any]]) -> bytes:
    """Экспорт БД: лист на каждый тип объекта."""
    return build_markup_workbook_bytes(objects)


def export_objects_csv_zip(objects: list[dict[str, Any]]) -> bytes:
    """ZIP с CSV-файлами (по одному на тип объекта) — «листы» в терминах CSV."""
    import csv

    grouped: dict[str, list[dict]] = defaultdict(list)
    for obj in objects:
        sheet_name = obj.get("category") or OBJECT_CLASSES[0]
        grouped[sheet_name].append(obj)
    if not grouped:
        for cls in OBJECT_CLASSES:
            grouped[cls] = []

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for sheet_name, rows in grouped.items():
            csv_buf = io.StringIO()
            writer = csv.writer(csv_buf, lineterminator="\n")
            writer.writerow(MARKUP_COLUMNS)
            for obj in rows:
                writer.writerow(_row_from_object(obj))
            safe_name = sheet_name.replace("/", "_").replace("\\", "_")
            zf.writestr(f"{safe_name}.csv", csv_buf.getvalue().encode("utf-8-sig"))
    return buf.getvalue()
