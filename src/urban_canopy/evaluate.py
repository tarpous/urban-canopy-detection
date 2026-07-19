"""Detection metrics: COCO-style mAP plus per-size precision/recall.

A compact, dependency-free implementation of the standard COCO protocol —
greedy IoU matching per image, precision/recall integrated over score
thresholds, mAP@0.5 and mAP@[0.5:0.95] — so the metric that headlines the
README is auditable in this repo rather than hidden inside a framework. It is
cross-checked against a tiny hand-computed fixture in the tests.

The per-size split (small/medium/large by crown pixel area) is where the honest
story lives: aerial crown detectors miss small crowns, and reporting mAP by
size band is what separates a real evaluation from a single vanity number.
"""

from __future__ import annotations

from dataclasses import dataclass

from urban_canopy.labels import Box

# COCO area bands, in pixels² (small < 32², medium < 96², large otherwise).
SMALL_MAX = 32**2
MEDIUM_MAX = 96**2
IOU_SWEEP = tuple(round(0.5 + 0.05 * step, 2) for step in range(10))  # 0.50 … 0.95


@dataclass(frozen=True, slots=True)
class Detection:
    box: Box
    score: float


def iou(a: Box, b: Box) -> float:
    """Intersection-over-union of two pixel boxes."""
    ix0, iy0 = max(a.xmin, b.xmin), max(a.ymin, b.ymin)
    ix1, iy1 = min(a.xmax, b.xmax), min(a.ymax, b.ymax)
    inter = max(ix1 - ix0, 0.0) * max(iy1 - iy0, 0.0)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def size_band(box: Box) -> str:
    if box.area < SMALL_MAX:
        return "small"
    return "medium" if box.area < MEDIUM_MAX else "large"


@dataclass(frozen=True, slots=True)
class _Match:
    """One detection labelled TP/FP after matching, with its score."""

    score: float
    is_true_positive: bool


def _match_image(
    detections: list[Detection], truths: list[Box], iou_threshold: float
) -> tuple[list[_Match], int]:
    """Greedy highest-score-first matching for one image; returns (matches, n_truth)."""
    ordered = sorted(detections, key=lambda d: d.score, reverse=True)
    claimed = [False] * len(truths)
    matches = []
    for detection in ordered:
        best_iou, best_index = 0.0, -1
        for index, truth in enumerate(truths):
            if claimed[index]:
                continue
            value = iou(detection.box, truth)
            if value >= iou_threshold and value > best_iou:
                best_iou, best_index = value, index
        if best_index >= 0:
            claimed[best_index] = True
            matches.append(_Match(detection.score, True))
        else:
            matches.append(_Match(detection.score, False))
    return matches, len(truths)


def _average_precision(matches: list[_Match], n_truth: int) -> float:
    """101-point interpolated AP (COCO convention) from labelled matches."""
    if n_truth == 0:
        return float("nan")
    ordered = sorted(matches, key=lambda m: m.score, reverse=True)
    tp = fp = 0
    precisions, recalls = [], []
    for match in ordered:
        if match.is_true_positive:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / n_truth)

    ap = 0.0
    for target in (index / 100 for index in range(101)):
        candidates = [p for p, r in zip(precisions, recalls, strict=True) if r >= target]
        ap += max(candidates) if candidates else 0.0
    return ap / 101


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    map_50: float
    map_50_95: float
    precision: float
    recall: float
    recall_by_size: dict[str, float]
    n_truth: int
    n_pred: int


def evaluate(
    predictions: list[list[Detection]],
    ground_truth: list[list[Box]],
    *,
    score_threshold: float = 0.3,
) -> EvaluationResult:
    """Evaluate per-image predictions against ground truth over the COCO sweep.

    ``predictions`` and ``ground_truth`` are parallel per-image lists.
    Precision/recall are reported at ``score_threshold`` (the deployment
    operating point); mAP integrates over all scores as COCO does.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("predictions and ground_truth must be per-image parallel lists")

    ap_per_iou = []
    for threshold in IOU_SWEEP:
        matches: list[_Match] = []
        total_truth = 0
        for detections, truths in zip(predictions, ground_truth, strict=True):
            image_matches, n_truth = _match_image(detections, truths, threshold)
            matches.extend(image_matches)
            total_truth += n_truth
        ap_per_iou.append(_average_precision(matches, total_truth))

    map_50 = ap_per_iou[0]
    valid = [ap for ap in ap_per_iou if ap == ap]  # drop NaN (no truth)
    map_50_95 = sum(valid) / len(valid) if valid else float("nan")

    precision, recall = _operating_point(predictions, ground_truth, score_threshold)
    recall_by_size = _recall_by_size(predictions, ground_truth, score_threshold)
    return EvaluationResult(
        map_50=round(map_50, 4),
        map_50_95=round(map_50_95, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        recall_by_size={band: round(value, 4) for band, value in recall_by_size.items()},
        n_truth=sum(len(truths) for truths in ground_truth),
        n_pred=sum(len(detections) for detections in predictions),
    )


def _operating_point(
    predictions: list[list[Detection]], ground_truth: list[list[Box]], score_threshold: float
) -> tuple[float, float]:
    tp = fp = 0
    total_truth = 0
    for detections, truths in zip(predictions, ground_truth, strict=True):
        kept = [d for d in detections if d.score >= score_threshold]
        matches, n_truth = _match_image(kept, truths, 0.5)
        tp += sum(1 for match in matches if match.is_true_positive)
        fp += sum(1 for match in matches if not match.is_true_positive)
        total_truth += n_truth
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / total_truth if total_truth else 1.0
    return precision, recall


def _recall_by_size(
    predictions: list[list[Detection]], ground_truth: list[list[Box]], score_threshold: float
) -> dict[str, float]:
    """Recall@0.5 for truths in each size band, at the deployment score.

    Recall — not AP — is reported per size because attributing a detector's
    false positives to a *truth* size band is ill-defined (an FP has no truth
    size). Recall by band answers the question that actually matters for crown
    detection: what fraction of small vs large crowns do we find?
    """
    result = {}
    for band in ("small", "medium", "large"):
        found = total = 0
        for detections, truths in zip(predictions, ground_truth, strict=True):
            band_truths = [box for box in truths if size_band(box) == band]
            kept = [d for d in detections if d.score >= score_threshold]
            matches, n_truth = _match_image(kept, band_truths, 0.5)
            found += sum(1 for match in matches if match.is_true_positive)
            total += n_truth
        result[band] = found / total if total else float("nan")
    return result
