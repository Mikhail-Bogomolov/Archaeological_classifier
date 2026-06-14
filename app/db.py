import json
import math
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any

PAGE_SIZE = 10


DB_PATH = Path("app") / "data.sqlite3"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS objects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                category TEXT NOT NULL,
                confidence INTEGER NOT NULL,
                date TEXT NOT NULL,
                features_json TEXT,
                image_bytes BLOB NOT NULL,
                image_mime TEXT NOT NULL
            )
            """
        )
        conn.commit()


def add_object(
    *,
    name: str,
    description: str,
    category: str,
    confidence: int,
    date: str,
    features: list[str] | None,
    image_bytes: bytes,
    image_mime: str,
) -> int:
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO objects (name, description, category, confidence, date, features_json, image_bytes, image_mime)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                description,
                category,
                confidence,
                date,
                json.dumps(features or [], ensure_ascii=False),
                sqlite3.Binary(image_bytes),
                image_mime,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def count_objects() -> int:
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS cnt FROM objects").fetchone()
        return int(row["cnt"]) if row else 0


def list_objects_paginated(page: int = 1, per_page: int = PAGE_SIZE) -> list[dict[str, Any]]:
    page = max(1, page)
    offset = (page - 1) * per_page
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, name, description, category, confidence, date
            FROM objects
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            (per_page, offset),
        ).fetchall()
        return [dict(r) for r in rows]


def list_objects() -> list[dict[str, Any]]:
    """Все объекты (краткий список)."""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, name, description, category, confidence, date
            FROM objects
            ORDER BY id DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]


def parse_object_date(date_str: str) -> datetime | None:
    for fmt in ("%d.%m.%Y %H:%M", "%d.%m.%Y"):
        try:
            return datetime.strptime(str(date_str).strip(), fmt)
        except ValueError:
            continue
    return None


def _object_in_date_range(obj: dict[str, Any], date_from: date | None, date_to: date | None) -> bool:
    if date_from is None and date_to is None:
        return True
    parsed = parse_object_date(str(obj.get("date") or ""))
    if parsed is None:
        return False
    obj_date = parsed.date()
    if date_from is not None and obj_date < date_from:
        return False
    if date_to is not None and obj_date > date_to:
        return False
    return True


def list_objects_for_export(
    date_from: date | None = None,
    date_to: date | None = None,
) -> list[dict[str, Any]]:
    """Объекты с признаками для экспорта; опционально — фильтр по дате сканирования."""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, name, description, category, confidence, date, features_json
            FROM objects
            ORDER BY id DESC
            """
        ).fetchall()
        result = []
        for row in rows:
            obj = dict(row)
            try:
                obj["features"] = json.loads(obj.get("features_json") or "[]")
            except Exception:
                obj["features"] = []
            obj.pop("features_json", None)
            if _object_in_date_range(obj, date_from, date_to):
                result.append(obj)
        return result


def get_export_date_bounds() -> dict[str, str | None]:
    """Минимальная и максимальная дата объектов в формате YYYY-MM-DD для полей ввода."""
    with _connect() as conn:
        rows = conn.execute("SELECT date FROM objects").fetchall()
    dates: list[date] = []
    for row in rows:
        parsed = parse_object_date(str(row["date"]))
        if parsed is not None:
            dates.append(parsed.date())
    if not dates:
        return {"min_date": None, "max_date": None}
    return {
        "min_date": min(dates).isoformat(),
        "max_date": max(dates).isoformat(),
    }


def pagination_meta(page: int, per_page: int = PAGE_SIZE) -> dict[str, int]:
    total = count_objects()
    total_pages = max(1, math.ceil(total / per_page)) if total else 1
    page = min(max(1, page), total_pages)
    return {
        "page": page,
        "per_page": per_page,
        "total_count": total,
        "total_pages": total_pages,
        "has_prev": page > 1,
        "has_next": page < total_pages,
        "prev_page": page - 1 if page > 1 else 1,
        "next_page": page + 1 if page < total_pages else total_pages,
    }


def get_object(object_id: int) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT id, name, description, category, confidence, date, features_json
            FROM objects
            WHERE id = ?
            """,
            (object_id,),
        ).fetchone()
        if row is None:
            return None
        obj = dict(row)
        try:
            obj["features"] = json.loads(obj.get("features_json") or "[]")
        except Exception:
            obj["features"] = []
        obj.pop("features_json", None)
        return obj


def get_object_image(object_id: int) -> tuple[bytes, str] | None:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT image_bytes, image_mime
            FROM objects
            WHERE id = ?
            """,
            (object_id,),
        ).fetchone()
        if row is None:
            return None
        return (bytes(row["image_bytes"]), str(row["image_mime"]))

