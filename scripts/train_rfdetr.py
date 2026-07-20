"""Fine-tune RF-DETR on the NEON benchmark and score it — headless.

Same protocol as ``train_yolo.py``: identical site-disjoint split (seed 0),
identical scorer, so the README row is directly comparable. Uses the COCO
dataset layout that RF-DETR's trainer expects (``train``/``valid`` folders with
``_annotations.coco.json``).

Run (GPU):   uv run python scripts/train_rfdetr.py --epochs 60
Smoke (CPU): uv run python scripts/train_rfdetr.py --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # Windows cp1252 consoles

# Keep every model/weights cache inside the repo (gitignored), not the machine.
_CACHE = str(Path(__file__).resolve().parents[1] / ".cache")
os.environ.setdefault("HF_HOME", _CACHE + "/huggingface")
os.environ.setdefault("TORCH_HOME", _CACHE + "/torch")

from urban_canopy.dataset import build_coco_dataset, discover_pairs
from urban_canopy.evaluate import Detection
from urban_canopy.labels import Box, from_coco
from urban_canopy.predictions import result_to_model_row, score_predictions

RESULTS = Path("results")
NAME = "RF-DETR (fine-tuned)"


def valid_ground_truth(dataset_dir: Path) -> dict[str, list[Box]]:
    payload = json.loads(
        (dataset_dir / "valid" / "_annotations.coco.json").read_text(encoding="utf-8")
    )
    return {a.image_name: list(a.boxes) for a in from_coco(payload)}


def predict_valid(
    model: object, dataset_dir: Path, *, threshold: float
) -> dict[str, list[Detection]]:
    from PIL import Image

    predictions: dict[str, list[Detection]] = {}
    for image_path in sorted((dataset_dir / "valid").glob("*.png")):
        with Image.open(image_path) as image:
            result = model.predict(image.convert("RGB"), threshold=threshold)  # type: ignore[attr-defined]
        detections = []
        for box, score in zip(result.xyxy, result.confidence, strict=True):
            detections.append(Detection(Box(*[float(v) for v in box]), float(score)))
        predictions[image_path.name] = detections
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", default="data/raw/evaluation/RGB")
    parser.add_argument("--annotations", default="data/raw/annotations")
    parser.add_argument("--out", default="data/coco")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--smoke", action="store_true", help="1 CPU epoch on the sample tiles.")
    args = parser.parse_args()

    if args.smoke:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # before torch initializes CUDA
        images = annotations = Path("data/sample")
        out, epochs, batch = Path("data/coco_smoke"), 1, 2
        tile = 320
    else:
        images, annotations = Path(args.images), Path(args.annotations)
        out, epochs, batch = Path(args.out), args.epochs, args.batch
        tile = 640

    pairs = discover_pairs(images, annotations)
    if not pairs:
        raise SystemExit(f"no image/annotation pairs under {images} — run download_neon.py first")
    manifest = build_coco_dataset(
        pairs,
        out,
        val_fraction=0.5 if args.smoke else 0.2,
        seed=0,
        tile_size=tile,
        overlap=tile // 5,
    )
    if set(manifest.train_sites) & set(manifest.val_sites):
        raise SystemExit("site leakage in split — aborting")
    print(f"dataset: {manifest.n_train_images} train / {manifest.n_val_images} valid tiles")

    from rfdetr import RFDETRSmall

    model = RFDETRSmall()
    model.train(
        dataset_dir=str(out),
        epochs=epochs,
        batch_size=batch,
        grad_accum_steps=2,
        lr=args.lr,
        output_dir="runs/rfdetr",
        num_workers=0,  # Windows-safe
    )

    predictions = predict_valid(model, out, threshold=0.05)
    truth = valid_ground_truth(out)
    result = score_predictions(predictions, truth, score_threshold=args.threshold)
    print(
        f"{NAME}: mAP50={result.map_50} mAP50-95={result.map_50_95} "
        f"P={result.precision} R={result.recall} by_size={result.recall_by_size}"
    )

    if args.smoke:
        print("smoke OK: dataset → train → predict → score wiring intact")
        return
    metrics_path = RESULTS / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    row = result_to_model_row(NAME, result, inference="tile (640)")
    metrics["models"] = [m for m in metrics["models"] if m["name"] != NAME] + [row]
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print("wrote results/metrics.json — run `canopy make-table` to refresh the README")


if __name__ == "__main__":
    main()
