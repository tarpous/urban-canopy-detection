"""Prediction I/O and scoring tests, running on the committed samples."""

from pathlib import Path

import pytest

from urban_canopy.evaluate import Detection
from urban_canopy.labels import Box, parse_voc_xml
from urban_canopy.predictions import (
    detections_from_json,
    detections_to_json,
    load_ground_truth,
    result_to_model_row,
    score_predictions,
)

SAMPLES = Path("data/sample")


class TestJsonRoundTrip:
    def test_round_trip_is_byte_exact(self) -> None:
        per_image = {
            "b.tif": [Detection(Box(1, 2, 3, 4), 0.9)],
            "a.tif": [Detection(Box(10, 20, 30, 40), 0.5), Detection(Box(0, 0, 5, 5), 0.7)],
        }
        text = detections_to_json(per_image)
        assert detections_to_json(detections_from_json(text)) == text

    def test_keys_are_sorted(self) -> None:
        text = detections_to_json({"z.tif": [], "a.tif": []})
        assert text.index('"a.tif"') < text.index('"z.tif"')


class TestGroundTruth:
    def test_loads_sample_annotations(self) -> None:
        truth = load_ground_truth(SAMPLES)
        assert set(truth) == {
            "BLAN_005_2019.tif",
            "2018_SJER_3_252000_4104000_image_628.tif",
        }
        assert len(truth["BLAN_005_2019.tif"]) == 33


class TestScoring:
    def test_perfect_predictions_score_one_on_samples(self) -> None:
        truth = load_ground_truth(SAMPLES)
        predictions = {
            name: [Detection(box, 0.99) for box in boxes] for name, boxes in truth.items()
        }
        result = score_predictions(predictions, truth)
        assert result.map_50 == pytest.approx(1.0)
        assert result.recall == pytest.approx(1.0)
        assert result.n_truth == 41

    def test_missing_prediction_entry_penalizes_recall(self) -> None:
        truth = load_ground_truth(SAMPLES)
        # Predict for only one of the two images.
        one = next(iter(truth))
        predictions = {one: [Detection(box, 0.9) for box in truth[one]]}
        result = score_predictions(predictions, truth)
        assert 0.0 < result.recall < 1.0

    def test_real_voc_boxes_match_themselves(self) -> None:
        annotation = parse_voc_xml(SAMPLES / "BLAN_005_2019.xml")
        truth = {annotation.image_name: list(annotation.boxes)}
        predictions = {annotation.image_name: [Detection(b, 0.8) for b in annotation.boxes]}
        result = score_predictions(predictions, truth)
        assert result.precision == pytest.approx(1.0)


class TestModelRow:
    def test_row_shape_matches_metrics_schema(self) -> None:
        truth = load_ground_truth(SAMPLES)
        predictions = {name: [Detection(b, 0.9) for b in boxes] for name, boxes in truth.items()}
        result = score_predictions(predictions, truth)
        row = result_to_model_row("YOLO26-s", result, inference="SAHI sliced")
        assert row["status"] == "measured"
        assert row["name"] == "YOLO26-s"
        assert set(row) == {
            "name",
            "status",
            "map_50",
            "map_50_95",
            "precision",
            "recall",
            "recall_small",
            "recall_medium",
            "recall_large",
            "inference",
        }
