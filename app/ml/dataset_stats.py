"""Счётчики пропусков при загрузке Excel."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RowLoadStats:
    worksheets_seen: int = 0
    worksheets_skipped: int = 0
    rows_total: int = 0
    rows_kept: int = 0
    skip_reasons: dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str, count: int = 1) -> None:
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + count

    def summary_lines(self, prefix: str = "") -> list[str]:
        lines = [
            f"{prefix}строк в таблице: {self.rows_total}, принято: {self.rows_kept}",
        ]
        if self.worksheets_skipped:
            lines.append(
                f"{prefix}листов пропущено (нет колонок): {self.worksheets_skipped}"
            )
        for reason, count in sorted(self.skip_reasons.items(), key=lambda x: -x[1]):
            if count:
                lines.append(f"{prefix}  - {reason}: {count}")
        return lines
