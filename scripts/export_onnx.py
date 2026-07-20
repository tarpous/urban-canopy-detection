"""Export a trained YOLO detector to ONNX for the in-browser (static) demo.

The static Hugging Face Space in ``app/static`` runs this ONNX model with ONNX
Runtime Web (WASM); YOLO26's NMS-free head means the browser only has to
threshold the ``[1, 300, 6]`` output (x1, y1, x2, y2, score, class). Copies the
model next to the page so ``app/static`` is a self-contained deployable folder.

Run: uv run python scripts/export_onnx.py --weights runs/detect/runs/yolo26/weights/best.pt
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

_CACHE = str(Path(__file__).resolve().parents[1] / ".cache")
os.environ.setdefault("YOLO_CONFIG_DIR", _CACHE + "/ultralytics")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="runs/detect/runs/yolo26/weights/best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out", default="app/static/best.onnx")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    onnx_path = model.export(
        format="onnx", imgsz=args.imgsz, opset=17, simplify=True, dynamic=False
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx_path, out)
    print(f"exported {onnx_path} -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
