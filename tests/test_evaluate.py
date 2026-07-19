"""Metric tests: IoU, matching, and mAP against hand-computed values."""

import math

import pytest

from urban_canopy.evaluate import (
    Detection,
    EvaluationResult,
    evaluate,
    iou,
    size_band,
)
from urban_canopy.labels import Box


def det(xmin: float, ymin: float, xmax: float, ymax: float, score: float) -> Detection:
    return Detection(Box(xmin, ymin, xmax, ymax), score)


class TestIou:
    def test_identical_boxes(self) -> None:
        box = Box(0, 0, 10, 10)
        assert iou(box, box) == pytest.approx(1.0)

    def test_disjoint_boxes(self) -> None:
        assert iou(Box(0, 0, 10, 10), Box(20, 20, 30, 30)) == 0.0

    def test_half_overlap(self) -> None:
        # 10x10 boxes sharing a 5x10 strip: inter 50, union 150.
        assert iou(Box(0, 0, 10, 10), Box(5, 0, 15, 10)) == pytest.approx(50 / 150)


class TestSizeBand:
    @pytest.mark.parametrize(
        ("side", "expected"),
        [(20, "small"), (50, "medium"), (200, "large")],
    )
    def test_bands(self, side: float, expected: str) -> None:
        assert size_band(Box(0, 0, side, side)) == expected


class TestEvaluate:
    def test_perfect_predictions_score_one(self) -> None:
        truths = [[Box(0, 0, 40, 40), Box(100, 100, 160, 160)]]
        preds = [[det(0, 0, 40, 40, 0.9), det(100, 100, 160, 160, 0.8)]]
        result = evaluate(preds, truths)
        assert result.map_50 == pytest.approx(1.0)
        assert result.map_50_95 == pytest.approx(1.0)
        assert result.precision == pytest.approx(1.0)
        assert result.recall == pytest.approx(1.0)

    def test_no_predictions_is_zero_recall(self) -> None:
        result = evaluate([[]], [[Box(0, 0, 40, 40)]])
        assert result.map_50 == 0.0
        assert result.recall == 0.0
        assert result.precision == 1.0  # vacuously: no FP

    def test_one_hit_one_miss_one_false_alarm(self) -> None:
        truths = [[Box(0, 0, 40, 40), Box(100, 100, 140, 140)]]
        preds = [[det(0, 0, 40, 40, 0.9), det(500, 500, 540, 540, 0.7)]]  # 1 TP, 1 FP, miss 2nd
        result = evaluate(preds, truths)
        assert result.precision == pytest.approx(0.5)
        assert result.recall == pytest.approx(0.5)
        # AP with recall capped at 0.5: interpolated over 101 points → ~0.5.
        assert result.map_50 == pytest.approx(0.5, abs=0.02)

    def test_low_iou_match_fails_at_high_threshold(self) -> None:
        truths = [[Box(0, 0, 100, 100)]]
        preds = [[det(0, 0, 60, 100, 0.9)]]  # IoU = 0.6
        result = evaluate(preds, truths)
        assert result.map_50 == pytest.approx(1.0)  # counts at 0.50/0.55/0.60
        assert result.map_50_95 == pytest.approx(0.3)  # 3 of 10 thresholds pass

    def test_duplicate_detection_is_a_false_positive(self) -> None:
        truths = [[Box(0, 0, 40, 40)]]
        preds = [[det(0, 0, 40, 40, 0.9), det(0, 0, 40, 40, 0.8)]]  # second is dup
        result = evaluate(preds, truths)
        assert result.precision == pytest.approx(0.5)
        assert result.recall == pytest.approx(1.0)

    def test_recall_by_size_isolates_small_crowns(self) -> None:
        truths = [[Box(0, 0, 20, 20), Box(100, 100, 300, 300)]]  # small + large
        preds = [[det(100, 100, 300, 300, 0.9)]]  # only the large one detected
        result = evaluate(preds, truths)
        assert result.recall_by_size["large"] == pytest.approx(1.0)
        assert result.recall_by_size["small"] == pytest.approx(0.0)
        assert math.isnan(result.recall_by_size["medium"])

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="parallel"):
            evaluate([[]], [[], []])

    def test_result_is_serializable_shape(self) -> None:
        result = evaluate([[det(0, 0, 40, 40, 0.9)]], [[Box(0, 0, 40, 40)]])
        assert isinstance(result, EvaluationResult)
        assert result.n_pred == 1
        assert result.n_truth == 1
