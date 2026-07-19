"""Evaluate the published DeepForest RetinaNet baseline (CPU, inference-only).

DeepForest ships a RetinaNet pretrained on the NEON crowns; running it on the
same evaluation tiles gives the honest external reference the whole study is
measured against — no training, just inference through the tested scorer.

Run: uv run python scripts/run_baseline.py --images data/raw/evaluation/RGB \
         --annotations data/raw/annotations
Smoke: uv run python scripts/run_baseline.py --smoke   (stub predictions, no model)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from urban_canopy.evaluate import Detection
from urban_canopy.labels import Box, parse_voc_xml
from urban_canopy.predictions import load_ground_truth, result_to_model_row, score_predictions

RESULTS = Path("results")
NAME = "DeepForest RetinaNet (published baseline)"


def deepforest_predictions(images: Path) -> dict[str, list[Detection]]:
    """Run DeepForest's pretrained RetinaNet over every tile in ``images``."""
    from deepforest import main as deepforest_main

    model = deepforest_main.deepforest()
    model.use_release()  # download the published NEON-trained weights
    predictions: dict[str, list[Detection]] = {}
    for image_path in sorted(images.glob("*.tif")):
        boxes = model.predict_image(path=str(image_path))
        detections = []
        if boxes is not None:
            for row in boxes.itertuples():
                detections.append(
                    Detection(Box(row.xmin, row.ymin, row.xmax, row.ymax), float(row.score))
                )
        predictions[image_path.name] = detections
    return predictions


def stub_predictions(annotations: Path) -> dict[str, list[Detection]]:
    """Smoke-mode stand-in: echo the ground-truth boxes at a fixed score."""
    predictions: dict[str, list[Detection]] = {}
    for xml_path in sorted(annotations.glob("*.xml")):
        annotation = parse_voc_xml(xml_path)
        predictions[annotation.image_name] = [Detection(b, 0.7) for b in annotation.boxes]
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", default="data/raw/evaluation/RGB")
    parser.add_argument("--annotations", default="data/raw/annotations")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    images = Path("data/sample") if args.smoke else Path(args.images)
    annotations = Path("data/sample") if args.smoke else Path(args.annotations)

    predictions = stub_predictions(annotations) if args.smoke else deepforest_predictions(images)
    truth = load_ground_truth(annotations)
    result = score_predictions(predictions, truth, score_threshold=0.3)
    print(
        f"{NAME}: mAP50={result.map_50} mAP50-95={result.map_50_95} "
        f"P={result.precision} R={result.recall} by_size={result.recall_by_size}"
    )
    if args.smoke:
        print("smoke OK: baseline scoring wiring intact")
        return

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "metrics.json"
    metrics = json.loads(path.read_text(encoding="utf-8"))
    row = result_to_model_row(NAME, result, inference="whole-image, CPU")
    metrics["models"] = [m for m in metrics["models"] if m["name"] != NAME] + [row]
    path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print("wrote results/metrics.json")


if __name__ == "__main__":
    main()
