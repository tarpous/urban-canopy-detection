"""Gradio demo for the Hugging Face Space: upload an aerial image → crowns.

CPU-only. The heavy lifting (SAHI slicing, NMS, GeoJSON export) is the same
tested ``urban_canopy`` code the notebooks use; only the model load and the
Gradio glue live here. Weights are expected at ``weights/best.pt`` (committed
to the Space via git-lfs or downloaded on startup); if absent, the app runs in
a synthetic-demo mode so the Space still boots and the UI is inspectable.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import gradio as gr
from PIL import Image, ImageDraw

from urban_canopy.evaluate import Detection
from urban_canopy.labels import Box
from urban_canopy.sliced import sliced_predict
from urban_canopy.tiling import TileWindow

WEIGHTS = Path("weights/best.pt")
TILE_SIZE, OVERLAP = 640, 128


def _load_model():
    if not WEIGHTS.exists():
        return None
    from ultralytics import YOLO

    return YOLO(str(WEIGHTS))


MODEL = _load_model()


def _detector_for(image: Image.Image):
    def detect(window: TileWindow) -> list[Detection]:
        crop = image.crop((window.x0, window.y0, window.x1, window.y1))
        if MODEL is None:
            # Synthetic fallback: a coarse grid of boxes so the UI is demoable.
            step = 80
            return [
                Detection(Box(x, y, x + 40, y + 40), 0.5)
                for x in range(0, crop.width - 40, step)
                for y in range(0, crop.height - 40, step)
            ]
        result = MODEL.predict(crop, imgsz=TILE_SIZE, conf=0.15, verbose=False)[0]
        return [
            Detection(Box(*box), float(score))
            for box, score in zip(
                result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), strict=True
            )
        ]

    return detect


def predict(image: Image.Image) -> tuple[Image.Image, str, str]:
    """Run sliced detection; return an annotated image, a count, and GeoJSON path."""
    image = image.convert("RGB")
    detections = sliced_predict(
        image.width, image.height, _detector_for(image), tile_size=TILE_SIZE, overlap=OVERLAP
    )

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for detection in detections:
        box = detection.box
        draw.rectangle([box.xmin, box.ymin, box.xmax, box.ymax], outline="#C1121F", width=3)

    features = [
        {
            "type": "Feature",
            "properties": {"crown_id": index, "score": round(detection.score, 3)},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [detection.box.xmin, detection.box.ymin],
                        [detection.box.xmax, detection.box.ymin],
                        [detection.box.xmax, detection.box.ymax],
                        [detection.box.xmin, detection.box.ymax],
                        [detection.box.xmin, detection.box.ymin],
                    ]
                ],
            },
        }
        for index, detection in enumerate(detections)
    ]
    geojson = {"type": "FeatureCollection", "features": features}
    out = Path(tempfile.mkdtemp()) / "crowns_pixels.geojson"
    out.write_text(json.dumps(geojson, indent=2), encoding="utf-8")

    note = "" if MODEL is not None else " (synthetic demo — no weights loaded)"
    return annotated, f"{len(detections)} crowns detected{note}", str(out)


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Aerial RGB image"),
    outputs=[
        gr.Image(type="pil", label="Detected crowns"),
        gr.Textbox(label="Crown count"),
        gr.File(label="Detections (GeoJSON, pixel coordinates)"),
    ],
    title="Urban canopy detection",
    description=(
        "Upload an aerial RGB image; the model tiles it (SAHI), detects tree crowns, "
        "and returns boxes, a count, and a downloadable GeoJSON. Model: YOLO26-s "
        "fine-tuned on the NEON benchmark. Code: github.com/tarpous/urban-canopy-detection"
    ),
    flagging_mode="never",
)


if __name__ == "__main__":
    demo.launch()
