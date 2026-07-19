"""Serialize per-image detections and score them, independent of any detector.

Every model (DeepForest baseline, YOLO26, RF-DETR) writes predictions in this
one JSON shape; ``score_predictions`` then evaluates any of them against the
benchmark ground truth with the same code, so the comparison is apples-to-apples
by construction. This is the seam between the GPU notebooks and the CPU-side
metrics: notebooks emit ``predictions.json``, this module turns it into the
``results/metrics.json`` row.
"""

from __future__ import annotations

import json
from pathlib import Path

from urban_canopy.evaluate import Detection, EvaluationResult, evaluate
from urban_canopy.labels import Box, ImageAnnotation, parse_voc_xml


def detections_to_json(per_image: dict[str, list[Detection]]) -> str:
    """Serialize {image_name: [Detection, …]} to canonical JSON."""
    payload = {
        name: [
            {
                "bbox": [
                    round(d.box.xmin, 2),
                    round(d.box.ymin, 2),
                    round(d.box.xmax, 2),
                    round(d.box.ymax, 2),
                ],
                "score": round(d.score, 4),
            }
            for d in detections
        ]
        for name, detections in sorted(per_image.items())
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def detections_from_json(text: str) -> dict[str, list[Detection]]:
    """Inverse of :func:`detections_to_json`."""
    raw = json.loads(text)
    return {
        name: [Detection(Box(*entry["bbox"]), float(entry["score"])) for entry in detections]
        for name, detections in raw.items()
    }


def load_ground_truth(annotation_dir: Path) -> dict[str, list[Box]]:
    """Load VOC ground-truth boxes keyed by image file name."""
    truth: dict[str, list[Box]] = {}
    for xml_path in sorted(annotation_dir.glob("*.xml")):
        annotation: ImageAnnotation = parse_voc_xml(xml_path)
        truth[annotation.image_name] = list(annotation.boxes)
    return truth


def score_predictions(
    predictions: dict[str, list[Detection]],
    ground_truth: dict[str, list[Box]],
    *,
    score_threshold: float = 0.3,
) -> EvaluationResult:
    """Evaluate predictions against ground truth over their shared images.

    Only images present in the ground truth are scored (the benchmark's
    evaluation set); a missing prediction entry counts as "no detections",
    which correctly penalizes recall.
    """
    names = sorted(ground_truth)
    pred_lists = [predictions.get(name, []) for name in names]
    truth_lists = [ground_truth[name] for name in names]
    return evaluate(pred_lists, truth_lists, score_threshold=score_threshold)


def result_to_model_row(
    name: str, result: EvaluationResult, *, inference: str
) -> dict[str, object]:
    """Shape an EvaluationResult into a ``results/metrics.json`` model row."""
    return {
        "name": name,
        "status": "measured",
        "map_50": result.map_50,
        "map_50_95": result.map_50_95,
        "precision": result.precision,
        "recall": result.recall,
        "recall_small": result.recall_by_size.get("small"),
        "recall_medium": result.recall_by_size.get("medium"),
        "recall_large": result.recall_by_size.get("large"),
        "inference": inference,
    }
