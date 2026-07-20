"""Render the README figures from a trained detector on a held-out tile.

Produces two real-data artifacts, both committed to ``docs/img``:

- ``detection_example.png`` — a validation tile (a site the model never trained
  on) with the detected crown boxes drawn on it: the demo visual.
- ``canopy_density.png`` + ``docs/geo/canopy_density.gpkg`` — the detections
  georeferenced to the tile's UTM CRS, aggregated into a 10 m crown-density grid
  (``geo.canopy_density_grid``) and drawn as a choropleth, exercising the full
  pixel → CRS → GeoPackage path on real data.

Run: uv run python scripts/make_figures.py --weights runs/detect/runs/yolo26/weights/best.pt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

_CACHE = str(Path(__file__).resolve().parents[1] / ".cache")
os.environ.setdefault("YOLO_CONFIG_DIR", _CACHE + "/ultralytics")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from urban_canopy.dataset import discover_pairs
from urban_canopy.evaluate import Detection
from urban_canopy.geo import (
    GeoReference,
    canopy_density_grid,
    detections_to_geodataframe,
    write_geopackage,
)
from urban_canopy.labels import Box
from urban_canopy.splits import split_by_site

IMG = Path("docs/img")
GEO = Path("docs/geo")
ACCENT = "#C1121F"


def val_tile_paths(images: Path, annotations: Path, *, seed: int = 0) -> list[Path]:
    pairs = discover_pairs(images, annotations)
    by_name = {p[0].name: p[0] for p in pairs}
    split = split_by_site(sorted(by_name), val_fraction=0.2, seed=seed)
    return [by_name[n] for n in split.val]


def detect(model: object, tile: Path, *, conf: float) -> list[Detection]:
    from PIL import Image

    with Image.open(tile) as image:
        result = model.predict(image.convert("RGB"), conf=conf, device="cpu", verbose=False)  # type: ignore[attr-defined]
        out = result[0]
    return [
        Detection(Box(*b), float(s))
        for b, s in zip(out.boxes.xyxy.tolist(), out.boxes.conf.tolist(), strict=True)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="runs/detect/runs/yolo26/weights/best.pt")
    parser.add_argument("--images", default="data/raw/evaluation/RGB")
    parser.add_argument("--annotations", default="data/raw/annotations")
    parser.add_argument("--conf", type=float, default=0.2)
    args = parser.parse_args()

    import numpy as np
    from PIL import Image
    from ultralytics import YOLO

    IMG.mkdir(parents=True, exist_ok=True)
    GEO.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)

    tiles = val_tile_paths(Path(args.images), Path(args.annotations))
    if not tiles:
        raise SystemExit("no val tiles found — run download_neon.py first")
    # Pick the held-out tile with the most detected crowns for a clear figure.
    best_tile, best_dets = max(
        ((t, detect(model, t, conf=args.conf)) for t in tiles), key=lambda pair: len(pair[1])
    )
    print(f"figure tile: {best_tile.name} ({len(best_dets)} crowns)")

    # --- Figure 1: detections drawn on the tile ---
    with Image.open(best_tile) as image:
        rgb = np.asarray(image.convert("RGB"))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb)
    for d in best_dets:
        b = d.box
        ax.add_patch(
            plt.Rectangle((b.xmin, b.ymin), b.width, b.height, fill=False, edgecolor=ACCENT, lw=1.5)
        )
    ax.set_title(
        f"{best_tile.stem}: {len(best_dets)} tree crowns detected (held-out site)", fontsize=10
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(IMG / "detection_example.png", dpi=150, facecolor="white")
    plt.close(fig)

    # --- Figure 2: georeferenced 10 m crown-density grid ---
    ref = GeoReference.from_raster(best_tile)
    frame = detections_to_geodataframe(best_dets, ref)
    grid = canopy_density_grid(frame, cell_size_m=10.0)
    write_geopackage(grid, GEO / "canopy_density.gpkg", layer="density")

    fig, ax = plt.subplots(figsize=(6, 5))
    if not grid.empty:
        grid.plot(
            column="crowns_per_ha",
            cmap="YlGn",
            edgecolor="#888",
            linewidth=0.4,
            legend=True,
            ax=ax,
            legend_kwds={"label": "crowns / hectare"},
        )
        frame.centroid.plot(ax=ax, color=ACCENT, markersize=6)
    ax.set_title("Crown-density grid (10 m cells, UTM) over the tile", fontsize=10)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(IMG / "canopy_density.png", dpi=150, facecolor="white")
    plt.close(fig)
    print(
        f"wrote {IMG / 'detection_example.png'}, {IMG / 'canopy_density.png'}, "
        f"{GEO / 'canopy_density.gpkg'}"
    )


if __name__ == "__main__":
    main()
