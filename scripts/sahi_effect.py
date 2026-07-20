"""Quantify the SAHI (sliced) inference effect for the results table.

Compares whole-image inference against SAHI sliced inference on the same
held-out val tiles, with the same trained model and the same scorer, and writes
the numbers into ``results/metrics.json``. Because the NEON benchmark tiles are
only 400 px, slicing uses a 256 px window (a real 2x2 grid over each tile) so
the effect is actually exercised — on full orthophotos the tile size would be
640 and the effect much larger.

Run: uv run python scripts/sahi_effect.py --weights runs/detect/runs/yolo/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

_CACHE = str(Path(__file__).resolve().parents[1] / ".cache")
os.environ.setdefault("YOLO_CONFIG_DIR", _CACHE + "/ultralytics")

from urban_canopy.evaluate import Detection, evaluate
from urban_canopy.labels import Box, from_yolo_text
from urban_canopy.sliced import sliced_predict
from urban_canopy.tiling import TileWindow

RESULTS = Path("results")


def load_val(dataset_dir: Path) -> list[tuple[Path, list[Box]]]:
    from PIL import Image

    items = []
    for image_path in sorted((dataset_dir / "images" / "val").glob("*.png")):
        with Image.open(image_path) as image:
            width, height = image.size
        label = dataset_dir / "labels" / "val" / f"{image_path.stem}.txt"
        text = label.read_text(encoding="utf-8") if label.exists() else ""
        annotation = from_yolo_text(text, image_name=image_path.name, width=width, height=height)
        items.append((image_path, list(annotation.boxes)))
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="runs/detect/runs/yolo/weights/best.pt")
    parser.add_argument("--dataset", default="data/yolo")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--slice", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--name", default="YOLO11-s", help="Model label for the metrics block.")
    args = parser.parse_args()

    from PIL import Image
    from ultralytics import YOLO

    model = YOLO(args.weights)
    items = load_val(Path(args.dataset))
    if not items:
        raise SystemExit(f"no val tiles under {args.dataset}/images/val — run train_yolo.py first")

    def detect_crop(image: Image.Image, window: TileWindow) -> list[Detection]:
        crop = image.crop((window.x0, window.y0, window.x1, window.y1))
        out = model.predict(crop, conf=0.05, device=args.device, verbose=False)[0]
        return [
            Detection(Box(*b), float(s))
            for b, s in zip(out.boxes.xyxy.tolist(), out.boxes.conf.tolist(), strict=True)
        ]

    whole_preds, sliced_preds, truths = [], [], []
    for image_path, boxes in items:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            out = model.predict(image, conf=args.conf, device=args.device, verbose=False)[0]
            whole = [
                Detection(Box(*b), float(s))
                for b, s in zip(out.boxes.xyxy.tolist(), out.boxes.conf.tolist(), strict=True)
            ]
            sliced = sliced_predict(
                width,
                height,
                lambda w, im=image: detect_crop(im, w),
                tile_size=args.slice,
                overlap=args.overlap,
            )
        whole_preds.append(whole)
        sliced_preds.append([d for d in sliced if d.score >= args.conf])
        truths.append(boxes)

    whole_map = evaluate(whole_preds, truths).map_50
    sliced_map = evaluate(sliced_preds, truths).map_50
    print(
        f"SAHI effect: whole-image mAP50={whole_map} -> "
        f"sliced({args.slice}/{args.overlap}) mAP50={sliced_map}"
    )

    path = RESULTS / "metrics.json"
    metrics = json.loads(path.read_text(encoding="utf-8"))
    metrics["sahi_effect"] = {
        "status": "measured",
        "model": args.name,
        "slice": f"{args.slice}/{args.overlap}",
        "whole_image_map_50": whole_map,
        "sliced_map_50": sliced_map,
    }
    path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print("wrote results/metrics.json")


if __name__ == "__main__":
    main()
