"""Fine-tune a YOLO detector on the NEON benchmark and score it — headless.

Primary local-GPU entry point (RTX 4080 SUPER). Reuses the tested
``urban_canopy`` package for dataset building, sliced inference and scoring; the
only untested surface is the Ultralytics ``train``/``val`` call itself, which the
``--smoke`` path exercises on CPU over the two committed sample tiles.

Run (GPU):   uv run python scripts/train_yolo.py --model yolo26s.pt --epochs 80
Smoke (CPU): uv run python scripts/train_yolo.py --smoke
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Keep every model/weights cache inside the repo (gitignored), not the machine.
_CACHE = str(Path(__file__).resolve().parents[1] / ".cache")
os.environ.setdefault("HF_HOME", _CACHE + "/huggingface")
os.environ.setdefault("TORCH_HOME", _CACHE + "/torch")
os.environ.setdefault("YOLO_CONFIG_DIR", _CACHE + "/ultralytics")

from urban_canopy.dataset import build_yolo_dataset, discover_pairs
from urban_canopy.evaluate import Detection
from urban_canopy.labels import Box, from_yolo_text
from urban_canopy.predictions import result_to_model_row, score_predictions

RESULTS = Path("results")


def val_ground_truth(dataset_dir: Path) -> dict[str, list[Box]]:
    """Reconstruct val ground-truth boxes (pixels) from the YOLO label files."""
    truth: dict[str, list[Box]] = {}
    from PIL import Image

    for image_path in sorted((dataset_dir / "images" / "val").glob("*.png")):
        with Image.open(image_path) as image:
            width, height = image.size
        label_path = dataset_dir / "labels" / "val" / f"{image_path.stem}.txt"
        text = label_path.read_text(encoding="utf-8") if label_path.exists() else ""
        annotation = from_yolo_text(text, image_name=image_path.name, width=width, height=height)
        truth[image_path.name] = list(annotation.boxes)
    return truth


def predict_val(model: object, dataset_dir: Path, *, conf: float) -> dict[str, list[Detection]]:
    """Whole-image prediction on each val tile (they are already ≤ imgsz)."""
    predictions: dict[str, list[Detection]] = {}
    for image_path in sorted((dataset_dir / "images" / "val").glob("*.png")):
        result = model.predict(str(image_path), conf=conf, verbose=False)[0]  # type: ignore[attr-defined]
        predictions[image_path.name] = [
            Detection(Box(*box), float(score))
            for box, score in zip(
                result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), strict=True
            )
        ]
    return predictions


def update_metrics(name: str, row: dict[str, object]) -> None:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "metrics.json"
    metrics = json.loads(path.read_text(encoding="utf-8"))
    metrics["models"] = [m for m in metrics["models"] if m["name"] != name] + [row]
    metrics["dataset"]["status"] = "measured"
    path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo26s.pt", help="Ultralytics weights to start from.")
    parser.add_argument("--name", default="YOLO26-s (fine-tuned)", help="Metrics row name.")
    parser.add_argument("--images", default="data/raw/evaluation/RGB")
    parser.add_argument("--annotations", default="data/raw/annotations")
    parser.add_argument("--out", default="data/yolo")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--device", default=None, help="'0' for GPU, 'cpu', or auto.")
    parser.add_argument("--smoke", action="store_true", help="1 CPU epoch on the sample tiles.")
    args = parser.parse_args()

    from ultralytics import YOLO

    if args.smoke:
        images = annotations = Path("data/sample")
        out = Path("data/yolo_smoke")
        epochs, imgsz, batch, device, model_name = 1, 320, 2, "cpu", "yolo11n.pt"
    else:
        images, annotations, out = Path(args.images), Path(args.annotations), Path(args.out)
        epochs, imgsz, batch = args.epochs, args.imgsz, args.batch
        device, model_name = args.device, args.model

    pairs = discover_pairs(images, annotations)
    if not pairs:
        raise SystemExit(f"no image/annotation pairs under {images} — run download_neon.py first")

    manifest = build_yolo_dataset(
        pairs,
        out,
        val_fraction=0.5 if args.smoke else 0.2,
        seed=0,
        tile_size=imgsz,
        overlap=imgsz // 5,
    )
    if set(manifest.train_sites) & set(manifest.val_sites):
        raise SystemExit("site leakage in split — aborting")
    print(f"dataset: {manifest.n_train_images} train / {manifest.n_val_images} val tiles")

    model = YOLO(model_name)
    model.train(
        data=str(out / "data.yaml"),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        mosaic=0.0,  # OFF: mosaic hurts small, dense objects
        workers=0,  # Windows-safe
        seed=0,
        project="runs",
        name=out.name,
        exist_ok=True,
        verbose=False,
    )

    predictions = predict_val(model, out, conf=args.conf)
    truth = val_ground_truth(out)
    result = score_predictions(predictions, truth, score_threshold=args.conf)
    print(
        f"{args.name}: mAP50={result.map_50} mAP50-95={result.map_50_95} "
        f"P={result.precision} R={result.recall} by_size={result.recall_by_size}"
    )

    if not args.smoke:
        update_metrics(args.name, result_to_model_row(args.name, result, inference="tile (≤imgsz)"))
        print("wrote results/metrics.json — run `canopy make-table` to refresh the README")
    else:
        print("smoke OK: dataset → train → predict → score wiring intact")


if __name__ == "__main__":
    main()
