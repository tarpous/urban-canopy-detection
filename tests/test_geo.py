"""Georeferencing tests: the pixel↔world round-trip and vector outputs."""

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from urban_canopy.evaluate import Detection
from urban_canopy.geo import (
    GeoReference,
    canopy_density_grid,
    detections_to_geodataframe,
    pixel_box_to_polygon,
    world_to_pixel,
    write_geojson,
    write_geopackage,
)
from urban_canopy.labels import Box

# NEON-like UTM tile: 400x400 px at 0.1 m/px, origin at a plausible UTM easting/northing.
ORIGIN_X, ORIGIN_Y, RES = 315000.0, 4094000.0, 0.1
TRANSFORM = Affine.translation(ORIGIN_X, ORIGIN_Y) * Affine.scale(RES, -RES)


@pytest.fixture
def geotiff(tmp_path: Path) -> Path:
    path = tmp_path / "tile.tif"
    data = np.zeros((3, 400, 400), dtype=np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=400,
        width=400,
        count=3,
        dtype="uint8",
        crs="EPSG:32617",
        transform=TRANSFORM,
    ) as dst:
        dst.write(data)
    return path


@pytest.fixture
def ref(geotiff: Path) -> GeoReference:
    return GeoReference.from_raster(geotiff)


class TestGeoReference:
    def test_reads_transform_and_crs(self, ref: GeoReference) -> None:
        assert ref.crs == "EPSG:32617"
        assert ref.transform.c == pytest.approx(ORIGIN_X)
        assert ref.transform.f == pytest.approx(ORIGIN_Y)


class TestRoundTrip:
    @pytest.mark.parametrize(
        "box", [Box(0, 0, 40, 40), Box(120, 200, 180, 260), Box(399, 0, 400, 1)]
    )
    def test_pixel_to_world_to_pixel_within_half_a_pixel(self, box: Box, ref: GeoReference) -> None:
        polygon = pixel_box_to_polygon(box, ref)
        minx, miny, maxx, maxy = polygon.bounds
        col0, row0 = world_to_pixel(minx, maxy, ref)  # upper-left corner
        col1, row1 = world_to_pixel(maxx, miny, ref)  # lower-right corner
        assert col0 == pytest.approx(box.xmin, abs=0.5)
        assert row0 == pytest.approx(box.ymin, abs=0.5)
        assert col1 == pytest.approx(box.xmax, abs=0.5)
        assert row1 == pytest.approx(box.ymax, abs=0.5)

    def test_polygon_has_real_world_extent(self, ref: GeoReference) -> None:
        polygon = pixel_box_to_polygon(Box(0, 0, 100, 100), ref)
        # 100 px * 0.1 m/px = 10 m square = 100 m² area.
        assert polygon.area == pytest.approx(100.0, rel=1e-6)


class TestVectorOutputs:
    def test_geojson_is_valid_wgs84_featurecollection(
        self, ref: GeoReference, tmp_path: Path
    ) -> None:
        detections = [Detection(Box(10, 10, 50, 50), 0.9), Detection(Box(100, 100, 140, 160), 0.7)]
        frame = detections_to_geodataframe(detections, ref)
        out = write_geojson(frame, tmp_path / "crowns.geojson")
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["type"] == "FeatureCollection"
        assert len(payload["features"]) == 2
        # GeoJSON must be WGS84: UTM zone 17N lon/lat is roughly (-81, 37).
        lon, lat = payload["features"][0]["geometry"]["coordinates"][0][0]
        assert -85 < lon < -78
        assert 36 < lat < 38
        assert payload["features"][0]["properties"]["score"] == pytest.approx(0.9)

    def test_geopackage_layer_written(self, ref: GeoReference, tmp_path: Path) -> None:
        import geopandas as gpd

        frame = detections_to_geodataframe([Detection(Box(0, 0, 40, 40), 0.8)], ref)
        out = write_geopackage(frame, tmp_path / "crowns.gpkg")
        loaded = gpd.read_file(out, layer="crowns")
        assert len(loaded) == 1
        assert str(loaded.crs).endswith("32617")


class TestDensityGrid:
    def test_counts_crowns_per_cell(self, ref: GeoReference) -> None:
        # Three crowns within a 10 m span → one 100 m cell holding all three.
        detections = [
            Detection(Box(10, 10, 30, 30), 0.9),
            Detection(Box(40, 40, 60, 60), 0.9),
            Detection(Box(70, 70, 90, 90), 0.9),
        ]
        frame = detections_to_geodataframe(detections, ref)
        grid = canopy_density_grid(frame, cell_size_m=100.0)
        assert grid["count"].sum() == 3
        assert grid["crowns_per_ha"].max() == pytest.approx(3.0)

    def test_empty_input_yields_empty_grid(self, ref: GeoReference) -> None:
        import geopandas as gpd

        empty = gpd.GeoDataFrame({"crown_id": [], "score": []}, geometry=[], crs=ref.crs)
        grid = canopy_density_grid(empty)
        assert grid.empty
