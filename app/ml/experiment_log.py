"""CSV-лог метрик по эпохам (без внешних сервисов)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class EpochLogger:
    def __init__(self, csv_path: str | Path, fieldnames: list[str] | None = None):
        self.path = Path(csv_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        self._initialized = self.path.is_file() and self.path.stat().st_size > 0

    def log(self, row: dict[str, Any]) -> None:
        if self.fieldnames is None:
            self.fieldnames = list(row.keys())
        write_header = not self._initialized
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
                self._initialized = True
            writer.writerow({k: row.get(k, "") for k in self.fieldnames})
