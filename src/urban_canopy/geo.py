"""Georeferencing detections: pixel boxes → CRS polygons → GeoJSON / GeoPackage.

The affine transform and CRS come straight from the source GeoTIFF (NEON tiles
are UTM). The round-trip guarantee — pixel → world → pixel is exact to within
half a pixel, enforced in the tests — is the whole point: a detector's boxes
are worthless downstream if the georeferencing quietly drifts. A canopy-density
grid aggregates crown counts into fixed-size cells for the choropleth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.transform import Affine, rowcol, xy
from shapely.geometry import box as shapely_box

from urban_canopy.evaluate import Detection
from urban_canopy.labels import Box


@dataclass(frozen=True, slots=True)
class GeoReference:
    """The affine transform + CRS needed to place pixels in the world."""

    transform: Affine
    crs: str

    @classmethod
    def from_raster(cls, path: Path) -> GeoReference:
        with rasterio.open(path) as raster:
            crs = raster.crs
            return cls(transform=raster.transform, crs=str(crs) if crs else "EPSG:4326")


def pixel_box_to_polygon(box: Box, ref: GeoReference) -> object:
    """Convert a pixel-space box to a shapely polygon in the raster CRS."""
    x0, y0 = xy(ref.transform, box.ymin, box.xmin, offset="ul")
    x1, y1 = xy(ref.transform, box.ymax, box.xmax, offset="ul")
    return shapely_box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def world_to_pixel(x: float, y: float, ref: GeoReference) -> tuple[float, float]:
    """Inverse of the corner mapping: world coordinates → (col, row) in pixels."""
    row, col = rowcol(ref.transform, x, y, op=lambda value: value)
    return col, row


def detections_to_geodataframe(detections: list[Detection], ref: GeoReference) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame of crown polygons with scores, in the raster CRS."""
    records = [
        {"crown_id": index, "score": round(detection.score, 4)}
        for index, detection in enumerate(detections)
    ]
    geometries = [pixel_box_to_polygon(detection.box, ref) for detection in detections]
    return gpd.GeoDataFrame(records, geometry=geometries, crs=ref.crs)


def write_geojson(frame: gpd.GeoDataFrame, path: Path) -> Path:
    """Write crowns as GeoJSON (reprojected to WGS84, the GeoJSON standard CRS)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_crs("EPSG:4326").to_file(path, driver="GeoJSON")
    return path


def write_geopackage(frame: gpd.GeoDataFrame, path: Path, *, layer: str = "crowns") -> Path:
    """Write crowns to a GeoPackage layer in the native CRS."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_file(path, layer=layer, driver="GPKG")
    return path


def canopy_density_grid(frame: gpd.GeoDataFrame, *, cell_size_m: float = 100.0) -> gpd.GeoDataFrame:
    """Aggregate crown centroids into a fixed grid: count + crowns/hectare.

    Runs in the frame's projected CRS so ``cell_size_m`` is real metres; the
    caller reprojects for display. Empty input yields an empty grid, not an error.
    """
    if frame.empty:
        return gpd.GeoDataFrame({"count": [], "crowns_per_ha": []}, geometry=[], crs=frame.crs)

    centroids = frame.geometry.centroid
    minx, miny, _, _ = centroids.total_bounds
    cell_area_ha = (cell_size_m**2) / 10_000.0

    # Bin each centroid by floor division from a fixed origin; a point on a
    # lower/left cell edge belongs to exactly one cell (no boundary ambiguity,
    # no double counting), which geometric ``within`` tests cannot guarantee.
    counts_by_cell: dict[tuple[int, int], int] = {}
    for point in centroids:
        col = math.floor((point.x - minx) / cell_size_m)
        row = math.floor((point.y - miny) / cell_size_m)
        counts_by_cell[(col, row)] = counts_by_cell.get((col, row), 0) + 1

    cells = []
    counts = []
    for (col, row), count in sorted(counts_by_cell.items()):
        cx0 = minx + col * cell_size_m
        cy0 = miny + row * cell_size_m
        cells.append(shapely_box(cx0, cy0, cx0 + cell_size_m, cy0 + cell_size_m))
        counts.append(count)
    return gpd.GeoDataFrame(
        {"count": counts, "crowns_per_ha": [round(n / cell_area_ha, 2) for n in counts]},
        geometry=cells,
        crs=frame.crs,
    )
