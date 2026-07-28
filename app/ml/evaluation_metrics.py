"""Precision, recall, F1 и confusion matrix для бенчмарка."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

LOW_SUPPORT_THRESHOLD = 5


@dataclass(frozen=True)
class PerClassMetrics:
    label: str
    index: int
    support: int
    predicted: int
    tp: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class BenchmarkReport:
    task: str
    split: str
    n_samples: int
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    micro_precision: float
    micro_recall: float
    micro_f1: float
    per_class: list[PerClassMetrics]
    confusion: list[list[int]]
    labels: list[str]
    extra: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.extra:
            d["extra"] = self.extra
        return d


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _f1(precision: float, recall: float) -> float:
    return _safe_div(2.0 * precision * recall, precision + recall)


def compute_benchmark_report(
    y_true: list[int],
    y_pred: list[int],
    labels: list[str],
    *,
    task: str,
    split: str,
    extra: dict[str, Any] | None = None,
) -> BenchmarkReport:
    """Мультиклассовые метрики из списков индексов."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true и y_pred должны быть одной длины")
    n_classes = len(labels)
    confusion = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
    for true_i, pred_i in zip(y_true, y_pred):
        if 0 <= true_i < n_classes and 0 <= pred_i < n_classes:
            confusion[true_i][pred_i] += 1

    n_samples = len(y_true)
    total_tp = sum(confusion[i][i] for i in range(n_classes))
    accuracy = _safe_div(total_tp, n_samples)

    per_class: list[PerClassMetrics] = []
    macro_p: list[float] = []
    macro_r: list[float] = []
    macro_f: list[float] = []
    w_prec_num = 0.0
    w_rec_num = 0.0
    w_f1_num = 0.0
    w_den = 0.0

    for i, label in enumerate(labels):
        tp = confusion[i][i]
        support = sum(confusion[i][j] for j in range(n_classes))
        predicted = sum(confusion[j][i] for j in range(n_classes))
        prec = _safe_div(tp, predicted)
        rec = _safe_div(tp, support)
        f1 = _f1(prec, rec)
        per_class.append(
            PerClassMetrics(
                label=label,
                index=i,
                support=support,
                predicted=predicted,
                tp=tp,
                precision=prec,
                recall=rec,
                f1=f1,
            )
        )
        if support > 0:
            macro_p.append(prec)
            macro_r.append(rec)
            macro_f.append(f1)
            w_prec_num += prec * support
            w_rec_num += rec * support
            w_f1_num += f1 * support
            w_den += support

    macro_precision = _safe_div(sum(macro_p), len(macro_p))
    macro_recall = _safe_div(sum(macro_r), len(macro_r))
    macro_f1 = _safe_div(sum(macro_f), len(macro_f))
    weighted_precision = _safe_div(w_prec_num, w_den)
    weighted_recall = _safe_div(w_rec_num, w_den)
    weighted_f1 = _safe_div(w_f1_num, w_den)

    micro_precision = accuracy
    micro_recall = accuracy
    micro_f1 = accuracy

    return BenchmarkReport(
        task=task,
        split=split,
        n_samples=n_samples,
        accuracy=accuracy,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        weighted_precision=weighted_precision,
        weighted_recall=weighted_recall,
        weighted_f1=weighted_f1,
        micro_precision=micro_precision,
        micro_recall=micro_recall,
        micro_f1=micro_f1,
        per_class=per_class,
        confusion=confusion,
        labels=list(labels),
        extra=extra,
    )


def print_benchmark_report(report: BenchmarkReport) -> None:
    """Печать отчёта в консоль."""
    print(f"\n=== Бенчмарк: {report.task} ({report.split}, n={report.n_samples}) ===")
    print(f"Accuracy:  {report.accuracy:.1%}")
    print(
        "Macro:     "
        f"P={report.macro_precision:.1%}  "
        f"R={report.macro_recall:.1%}  "
        f"F1={report.macro_f1:.1%}"
    )
    print(
        "Weighted:  "
        f"P={report.weighted_precision:.1%}  "
        f"R={report.weighted_recall:.1%}  "
        f"F1={report.weighted_f1:.1%}"
    )
    print(
        "Micro:     "
        f"P={report.micro_precision:.1%}  "
        f"R={report.micro_recall:.1%}  "
        f"F1={report.micro_f1:.1%}"
    )

    print("\nПо классам (support = число истинных меток класса):")
    print(f"  {'класс':<22} {'sup':>5} {'pred':>5} {'prec':>7} {'rec':>7} {'f1':>7}")
    for row in report.per_class:
        if row.support == 0 and row.predicted == 0:
            continue
        print(
            f"  {row.label[:22]:<22} {row.support:5d} {row.predicted:5d} "
            f"{row.precision:6.1%} {row.recall:6.1%} {row.f1:6.1%}"
        )

    print("\nConfusion matrix (строка = истина, столбец = предсказание):")
    short = [lbl[:8] for lbl in report.labels]
    print(" " * 20 + " | " + " | ".join(f"{s:>8}" for s in short))
    for i, label in enumerate(report.labels):
        counts = " | ".join(f"{report.confusion[i][j]:8d}" for j in range(len(report.labels)))
        print(f"{label[:18]:>18} | {counts}")


def save_benchmark_report(report: BenchmarkReport, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _short_label(label: str, max_len: int = 14) -> str:
    return label if len(label) <= max_len else label[: max_len - 1] + "…"


def save_confusion_heatmap(
    report: BenchmarkReport,
    path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Сохранить confusion matrix как PNG (heatmap)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    labels = [_short_label(x) for x in report.labels]
    matrix = report.confusion
    chart_title = title or f"{report.task} · {report.split} · acc={report.accuracy:.1%}"

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        data = np.array(matrix, dtype=float)
        n = len(labels)
        fig, ax = plt.subplots(figsize=(max(6.5, n * 1.1), max(5.5, n * 1.0)))
        im = ax.imshow(data, cmap="Blues")
        ax.set_xticks(range(n), labels, rotation=35, ha="right")
        ax.set_yticks(range(n), labels)
        ax.set_xlabel("Предсказание")
        ax.set_ylabel("Истина")
        ax.set_title(chart_title)
        vmax = float(data.max()) if data.size else 1.0
        for i in range(n):
            for j in range(n):
                val = int(data[i, j])
                color = "white" if vmax and data[i, j] > vmax * 0.55 else "black"
                ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out
    except ImportError:
        return _save_confusion_heatmap_pil(report, out, chart_title)


def _save_confusion_heatmap_pil(
    report: BenchmarkReport,
    out: Path,
    title: str,
) -> Path:
    """Запасной heatmap без matplotlib (Pillow)."""
    from PIL import Image, ImageDraw, ImageFont

    labels = [_short_label(x, 12) for x in report.labels]
    matrix = report.confusion
    n = len(labels)
    cell = 56
    left = 110
    top = 70
    bottom = 40
    right = 30
    w = left + n * cell + right
    h = top + n * cell + bottom
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
        font_sm = ImageFont.truetype("arial.ttf", 11)
    except OSError:
        font = ImageFont.load_default()
        font_sm = font

    vmax = max((max(row) for row in matrix), default=1) or 1
    draw.text((12, 12), title[:70], fill=(20, 20, 20), font=font)

    for i in range(n):
        draw.text((8, top + i * cell + cell // 3), labels[i], fill=(30, 30, 30), font=font_sm)
        for j in range(n):
            val = matrix[i][j]
            t = val / vmax
            color = (
                int(235 - 140 * t),
                int(245 - 80 * t),
                int(255 - 20 * t),
            )
            x0 = left + j * cell
            y0 = top + i * cell
            draw.rectangle([x0, y0, x0 + cell - 2, y0 + cell - 2], fill=color, outline=(180, 180, 180))
            text_color = (255, 255, 255) if t > 0.55 else (20, 20, 20)
            draw.text((x0 + cell // 2 - 6, y0 + cell // 2 - 6), str(val), fill=text_color, font=font)
    for j in range(n):
        draw.text((left + j * cell + 4, top + n * cell + 8), labels[j], fill=(30, 30, 30), font=font_sm)

    img.save(out)
    return out


def save_per_class_bars(
    report: BenchmarkReport,
    path: str | Path,
    *,
    title: str | None = None,
) -> Path | None:
    """Столбцы precision / recall / F1 по классам. Нужен matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [r for r in report.per_class if r.support > 0 or r.predicted > 0]
    if not rows:
        return None

    labels = [_short_label(r.label) for r in rows]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.2), 4.5))
    ax.bar(x - width, [r.precision for r in rows], width, label="Precision", color="#4C78A8")
    ax.bar(x, [r.recall for r in rows], width, label="Recall", color="#F58518")
    ax.bar(x + width, [r.f1 for r in rows], width, label="F1", color="#54A24B")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title or f"{report.task} · {report.split} · per-class")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def resolve_report_image_paths(
    json_out: str | Path | None,
    heatmap_out: str | Path | None,
    *,
    default_stem: str,
) -> tuple[Path, Path]:
    """Пути для confusion PNG и bars PNG."""
    if heatmap_out:
        heat = Path(heatmap_out)
    elif json_out:
        heat = Path(json_out).with_suffix(".png")
    else:
        heat = Path("reports") / f"{default_stem}.png"
    bars = heat.with_name(heat.stem + "_bars.png")
    return heat, bars
