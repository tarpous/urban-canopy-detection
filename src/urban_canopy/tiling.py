"""Tiling of large aerial images into overlapping crops, with box handling.

Full orthophotos are far bigger than detector inputs; training and inference
run on overlapping tiles. Boxes are reassigned to every tile they remain
sufficiently visible in — a crown chopped to a sliver at a tile edge would
teach the model to hallucinate partial trees, so visibility is thresholded.
"""

from __future__ import annotations

from dataclasses import dataclass

from urban_canopy.labels import Box, ImageAnnotation


@dataclass(frozen=True, slots=True)
class TileWindow:
    """One tile's pixel window in the source image (half-open, x1/y1 exclusive)."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def compute_tile_grid(width: int, height: int, *, tile_size: int, overlap: int) -> list[TileWindow]:
    """Overlapping tile windows covering the image; edge tiles snap inward.

    Snapping the last row/column to the image edge keeps every tile exactly
    ``tile_size`` pixels (detectors want fixed input) at the cost of extra
    overlap there. Images smaller than one tile yield a single full-image
    window.
    """
    if tile_size <= 0 or not 0 <= overlap < tile_size:
        raise ValueError(f"need 0 <= overlap < tile_size, got {overlap=} {tile_size=}")
    if width <= tile_size and height <= tile_size:
        return [TileWindow(0, 0, width, height)]

    stride = tile_size - overlap
    xs = list(range(0, max(width - tile_size, 0) + 1, stride))
    ys = list(range(0, max(height - tile_size, 0) + 1, stride))
    if xs[-1] + tile_size < width:
        xs.append(width - tile_size)
    if ys[-1] + tile_size < height:
        ys.append(height - tile_size)
    return [
        TileWindow(x, y, min(x + tile_size, width), min(y + tile_size, height))
        for y in ys
        for x in xs
    ]


def clip_box_to_window(box: Box, window: TileWindow) -> Box | None:
    """Box intersected with a window, in tile-local coordinates (or None)."""
    xmin = max(box.xmin, float(window.x0))
    ymin = max(box.ymin, float(window.y0))
    xmax = min(box.xmax, float(window.x1))
    ymax = min(box.ymax, float(window.y1))
    if xmax <= xmin or ymax <= ymin:
        return None
    return Box(
        xmin=xmin - window.x0, ymin=ymin - window.y0, xmax=xmax - window.x0, ymax=ymax - window.y0
    )


def tile_annotation(
    annotation: ImageAnnotation,
    *,
    tile_size: int,
    overlap: int,
    min_visibility: float = 0.4,
) -> list[ImageAnnotation]:
    """Split one annotated image into per-tile annotations.

    A box is kept in a tile when the visible fraction of its original area is
    at least ``min_visibility``; the same crown may legitimately appear in
    several overlapping tiles.
    """
    tiles = []
    stem, dot, suffix = annotation.image_name.rpartition(".")
    base = stem if dot else annotation.image_name
    extension = f".{suffix}" if dot else ""
    for window in compute_tile_grid(
        annotation.width, annotation.height, tile_size=tile_size, overlap=overlap
    ):
        kept = []
        for box in annotation.boxes:
            clipped = clip_box_to_window(box, window)
            if clipped is not None and box.area > 0 and clipped.area / box.area >= min_visibility:
                kept.append(clipped)
        tiles.append(
            ImageAnnotation(
                image_name=f"{base}_x{window.x0}_y{window.y0}{extension}",
                width=window.width,
                height=window.height,
                boxes=tuple(kept),
            )
        )
    return tiles
