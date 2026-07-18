"""Tiling tests: grid geometry, edge snapping, and box reassignment."""

import pytest

from urban_canopy.labels import Box, ImageAnnotation
from urban_canopy.tiling import TileWindow, clip_box_to_window, compute_tile_grid, tile_annotation


class TestGrid:
    def test_exact_cover_without_remainder(self) -> None:
        grid = compute_tile_grid(100, 100, tile_size=50, overlap=0)
        assert len(grid) == 4
        assert grid[0] == TileWindow(0, 0, 50, 50)
        assert grid[-1] == TileWindow(50, 50, 100, 100)

    def test_overlap_produces_stride(self) -> None:
        grid = compute_tile_grid(100, 40, tile_size=40, overlap=10)
        xs = sorted({window.x0 for window in grid})
        assert xs == [0, 30, 60]  # stride 30, last tile snapped to 60 = 100 - 40

    def test_edge_tiles_snap_to_image_boundary(self) -> None:
        grid = compute_tile_grid(110, 40, tile_size=40, overlap=0)
        xs = sorted({window.x0 for window in grid})
        assert xs == [0, 40, 70]  # 70 + 40 = 110: full-size tile, extra overlap
        assert all(window.width == 40 for window in grid)

    def test_small_image_yields_single_window(self) -> None:
        assert compute_tile_grid(30, 20, tile_size=64, overlap=8) == [TileWindow(0, 0, 30, 20)]

    def test_invalid_parameters_raise(self) -> None:
        with pytest.raises(ValueError, match="overlap"):
            compute_tile_grid(100, 100, tile_size=50, overlap=50)


class TestBoxClipping:
    window = TileWindow(50, 50, 150, 150)

    def test_inside_box_becomes_tile_local(self) -> None:
        clipped = clip_box_to_window(Box(60, 70, 100, 120), self.window)
        assert clipped == Box(10.0, 20.0, 50.0, 70.0)

    def test_straddling_box_is_clipped(self) -> None:
        clipped = clip_box_to_window(Box(0, 0, 80, 80), self.window)
        assert clipped == Box(0.0, 0.0, 30.0, 30.0)

    def test_outside_box_is_dropped(self) -> None:
        assert clip_box_to_window(Box(0, 0, 40, 40), self.window) is None


class TestTileAnnotation:
    def test_boxes_land_in_the_right_tiles_with_visibility_filter(self) -> None:
        annotation = ImageAnnotation(
            "2019_TEAK_1.tif",
            100,
            100,
            (
                Box(5, 5, 45, 45),  # fully inside tile (0,0)
                Box(40, 40, 60, 60),  # centre straddle: 25% per tile → dropped at 0.4
            ),
        )
        tiles = tile_annotation(annotation, tile_size=50, overlap=0, min_visibility=0.4)
        assert [tile.image_name for tile in tiles] == [
            "2019_TEAK_1_x0_y0.tif",
            "2019_TEAK_1_x50_y0.tif",
            "2019_TEAK_1_x0_y50.tif",
            "2019_TEAK_1_x50_y50.tif",
        ]
        counts = [len(tile.boxes) for tile in tiles]
        assert counts == [1, 0, 0, 0]

    def test_lower_visibility_keeps_straddlers_in_all_tiles(self) -> None:
        annotation = ImageAnnotation("2019_TEAK_1.tif", 100, 100, (Box(40, 40, 60, 60),))
        tiles = tile_annotation(annotation, tile_size=50, overlap=0, min_visibility=0.2)
        assert [len(tile.boxes) for tile in tiles] == [1, 1, 1, 1]

    def test_overlapping_tiles_can_share_a_crown(self) -> None:
        annotation = ImageAnnotation("2019_TEAK_1.tif", 100, 50, (Box(30, 10, 45, 40),))
        tiles = tile_annotation(annotation, tile_size=50, overlap=25, min_visibility=0.9)
        with_box = [tile for tile in tiles if tile.boxes]
        assert len(with_box) == 2  # visible in x0=0 and x0=25 windows
