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

from urban_canopy.dataset import discover_pairs
from urban_canopy.evaluate import Detection
from urban_canopy.labels import Box, parse_voc_xml
from urban_canopy.predictions import load_ground_truth, result_to_model_row, score_predictions
from urban_canopy.splits import split_by_site

RESULTS = Path("results")
NAME = "DeepForest RetinaNet (published baseline)"


def val_tile_stems(images: Path, annotations: Path, *, val_fraction: float, seed: int) -> set[str]:
    """The held-out val tiles, using the same site split the YOLO scripts use.

    Guarantees the baseline is scored on the identical test set as the trained
    detectors — the only honest way to compare a pretrained baseline against
    fine-tuned models.
    """
    pairs = discover_pairs(images, annotations)
    names = sorted(image_path.name for image_path, _ in pairs)
    split = split_by_site(names, val_fraction=val_fraction, seed=seed)
    return {Path(name).stem for name in split.val}


def deepforest_predictions(
    images: Path, *, only_stems: set[str] | None = None
) -> dict[str, list[Detection]]:
    """Run DeepForest's pretrained RetinaNet over the (optionally val-only) tiles."""
    from deepforest import main as deepforest_main

    model = deepforest_main.deepforest()
    # DeepForest 2.x: the published NEON-trained RetinaNet lives on the HF Hub.
    model.load_model(model_name="weecology/deepforest-tree")
    predictions: dict[str, list[Detection]] = {}
    for image_path in sorted(images.glob("*.tif")):
        if only_stems is not None and image_path.stem not in only_stems:
            continue
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
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--all-tiles", action="store_true", help="Score every tile instead of the held-out val set."
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    images = Path("data/sample") if args.smoke else Path(args.images)
    annotations = Path("data/sample") if args.smoke else Path(args.annotations)

    if args.smoke:
        predictions = stub_predictions(annotations)
        truth = load_ground_truth(annotations)
    else:
        val_stems = (
            None
            if args.all_tiles
            else val_tile_stems(images, annotations, val_fraction=args.val_fraction, seed=args.seed)
        )
        predictions = deepforest_predictions(images, only_stems=val_stems)
        full_truth = load_ground_truth(annotations)
        truth = (
            full_truth
            if val_stems is None
            else {name: boxes for name, boxes in full_truth.items() if Path(name).stem in val_stems}
        )
        print(f"scoring DeepForest on {len(truth)} held-out val tiles")
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
    inference = "whole-image, CPU" + ("" if args.all_tiles else " (val tiles)")
    row = result_to_model_row(NAME, result, inference=inference)
    metrics["models"] = [m for m in metrics["models"] if m["name"] != NAME] + [row]
    path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print("wrote results/metrics.json")


if __name__ == "__main__":
    main()
