"""Sliced (SAHI-style) inference: tile a big image, detect per tile, merge back.

Full orthophotos exceed a detector's input size, and downscaling them to fit
erases exactly the small crowns that matter. The SAHI recipe — slice with
overlap, run the detector per slice, map boxes back to full-image coordinates,
and de-duplicate overlaps with NMS — recovers small-object recall. This module
owns the geometry and merging; the actual per-tile detector is injected as a
callable so the whole flow is unit-tested without any model weights, and the
notebooks pass a real YOLO/RF-DETR predictor for the same code path.
"""

from __future__ import annotations

from collections.abc import Callable

from urban_canopy.evaluate import Detection, iou
from urban_canopy.labels import Box
from urban_canopy.tiling import TileWindow, compute_tile_grid

#: A per-tile detector: given an (image_id, window) it returns tile-local boxes.
TileDetector = Callable[[TileWindow], list[Detection]]


def offset_detection(detection: Detection, window: TileWindow) -> Detection:
    """Shift a tile-local detection into full-image pixel coordinates."""
    box = detection.box
    return Detection(
        Box(
            xmin=box.xmin + window.x0,
            ymin=box.ymin + window.y0,
            xmax=box.xmax + window.x0,
            ymax=box.ymax + window.y0,
        ),
        detection.score,
    )


def non_max_suppression(
    detections: list[Detection], *, iou_threshold: float = 0.5
) -> list[Detection]:
    """Greedy NMS: keep the highest-scoring box, drop its high-IoU neighbours."""
    ordered = sorted(detections, key=lambda d: d.score, reverse=True)
    kept: list[Detection] = []
    for candidate in ordered:
        if all(iou(candidate.box, chosen.box) < iou_threshold for chosen in kept):
            kept.append(candidate)
    return kept


def sliced_predict(
    width: int,
    height: int,
    detector: TileDetector,
    *,
    tile_size: int = 640,
    overlap: int = 128,
    iou_threshold: float = 0.5,
) -> list[Detection]:
    """Run ``detector`` over an overlapping tile grid and merge to full image.

    Boxes from every slice are offset into full-image coordinates and passed
    through NMS so a crown straddling a slice seam is reported once.
    """
    windows = compute_tile_grid(width, height, tile_size=tile_size, overlap=overlap)
    merged: list[Detection] = []
    for window in windows:
        for detection in detector(window):
            merged.append(offset_detection(detection, window))
    return non_max_suppression(merged, iou_threshold=iou_threshold)


def whole_image_baseline(detector: Callable[[], list[Detection]]) -> list[Detection]:
    """Trivial helper: whole-image inference, for the SAHI-vs-whole comparison."""
    return detector()
