"""Калибровка confidence: ECE и подбор порога предупреждения."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CalibrationReport:
    ece: float
    n_bins: int
    bin_accuracy: list[float]
    bin_confidence: list[float]
    bin_counts: list[int]
    suggested_threshold: float

    def as_dict(self) -> dict:
        return {
            "ece": self.ece,
            "n_bins": self.n_bins,
            "bin_accuracy": self.bin_accuracy,
            "bin_confidence": self.bin_confidence,
            "bin_counts": self.bin_counts,
            "suggested_threshold": self.suggested_threshold,
        }


def expected_calibration_error(
    y_true: list[int],
    y_pred: list[int],
    confidences: list[float],
    *,
    n_bins: int = 10,
) -> CalibrationReport:
    if not y_true:
        return CalibrationReport(
            ece=0.0,
            n_bins=n_bins,
            bin_accuracy=[0.0] * n_bins,
            bin_confidence=[0.0] * n_bins,
            bin_counts=[0] * n_bins,
            suggested_threshold=0.55,
        )

    correct = [int(t == p) for t, p in zip(y_true, y_pred)]
    bins = [[] for _ in range(n_bins)]
    for c, conf in zip(correct, confidences):
        conf_clamped = min(max(float(conf), 0.0), 0.999999)
        idx = min(int(conf_clamped * n_bins), n_bins - 1)
        bins[idx].append((c, conf_clamped))

    bin_accuracy: list[float] = []
    bin_confidence: list[float] = []
    bin_counts: list[int] = []
    ece = 0.0
    n = len(y_true)

    for bucket in bins:
        if not bucket:
            bin_accuracy.append(0.0)
            bin_confidence.append(0.0)
            bin_counts.append(0)
            continue
        acc = sum(c for c, _ in bucket) / len(bucket)
        avg_conf = sum(conf for _, conf in bucket) / len(bucket)
        bin_accuracy.append(acc)
        bin_confidence.append(avg_conf)
        bin_counts.append(len(bucket))
        ece += (len(bucket) / n) * abs(acc - avg_conf)

    suggested = _suggest_threshold(correct, confidences)
    return CalibrationReport(
        ece=ece,
        n_bins=n_bins,
        bin_accuracy=bin_accuracy,
        bin_confidence=bin_confidence,
        bin_counts=bin_counts,
        suggested_threshold=suggested,
    )


def _suggest_threshold(correct: list[int], confidences: list[float]) -> float:
    """Порог: максимизируем долю правильных среди conf >= t при t in [0.4..0.9]."""
    if not confidences:
        return 0.55
    best_t = 0.55
    best_score = -1.0
    for step in range(11, 91):
        t = step / 100.0
        matched = [(c, conf) for c, conf in zip(correct, confidences) if conf >= t]
        if len(matched) < max(5, len(confidences) // 20):
            continue
        precision = sum(c for c, _ in matched) / len(matched)
        coverage = len(matched) / len(confidences)
        score = precision * 0.7 + coverage * 0.3
        if score > best_score:
            best_score = score
            best_t = t
    return round(best_t, 2)
