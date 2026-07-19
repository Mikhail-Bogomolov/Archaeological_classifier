"""
Аудит обрезки предмета: подозрительные кадры (слишком туго / слабо / экстремальный aspect).

Запуск из корня:
    py scripts/audit_crops.py
    py scripts/audit_crops.py --limit 200 --json-out reports/crop_audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from app.ml.config import KANSK_PHOTOS_DIR, OBJECT_CLASSES
from app.ml.preprocess import _foreground_mask, _prepare_classifier_pil


def _audit_one(path: Path) -> dict:
    pil, flags = _prepare_classifier_pil(path=path)
    w, h = pil.size
    area = w * h
    elong = max(h / max(w, 1), w / max(h, 1))
    fg = float(_foreground_mask(pil).mean()) if area > 0 else 0.0
    reasons: list[str] = []
    if flags.get("artifact_cropped"):
        if fg > 0.88:
            reasons.append("fg_too_large")
        if fg < 0.04:
            reasons.append("fg_too_small")
        if elong > 4.5:
            reasons.append("extreme_aspect")
        if area < 80 * 80:
            reasons.append("tiny_crop")
    return {
        "file": path.name,
        "size": [w, h],
        "fg_frac": round(fg, 4),
        "elongation": round(elong, 3),
        "installation_shot": bool(flags.get("installation_shot")),
        "frame_cropped": bool(flags.get("frame_cropped")),
        "artifact_cropped": bool(flags.get("artifact_cropped")),
        "suspicious": reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Аудит crop для сети 1")
    parser.add_argument("--photos-dir", type=str, default=KANSK_PHOTOS_DIR)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--json-out", type=str, default="reports/crop_audit.json")
    args = parser.parse_args()

    photos = Path(args.photos_dir)
    if not photos.is_dir():
        raise SystemExit(f"Нет папки фото: {photos}")

    files = sorted(
        p for p in photos.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if args.limit > 0:
        files = files[: args.limit]

    rows: list[dict] = []
    suspicious: list[dict] = []
    for path in files:
        try:
            row = _audit_one(path)
        except Exception as exc:  # noqa: BLE001 — аудит не должен падать на одном файле
            row = {"file": path.name, "error": str(exc), "suspicious": ["load_error"]}
        rows.append(row)
        if row.get("suspicious"):
            suspicious.append(row)

    summary = {
        "n_files": len(rows),
        "n_suspicious": len(suspicious),
        "by_reason": {},
        "classes_hint": OBJECT_CLASSES,
    }
    for row in suspicious:
        for reason in row.get("suspicious", []):
            summary["by_reason"][reason] = summary["by_reason"].get(reason, 0) + 1

    out = {"summary": summary, "suspicious": suspicious, "all": rows}
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Файлов: {summary['n_files']}, подозрительных: {summary['n_suspicious']}")
    for reason, n in sorted(summary["by_reason"].items(), key=lambda x: -x[1]):
        print(f"  {reason}: {n}")
    print(f"Отчёт: {out_path}")
    if suspicious:
        print("Примеры:")
        for row in suspicious[:12]:
            print(
                f"  {row['file']}: {row.get('suspicious')} "
                f"fg={row.get('fg_frac')} elong={row.get('elongation')}"
            )


if __name__ == "__main__":
    main()
