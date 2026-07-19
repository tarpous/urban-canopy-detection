"""Sliced-inference tests: offsetting, NMS, and the merge with a fake detector."""

from urban_canopy.evaluate import Detection
from urban_canopy.labels import Box
from urban_canopy.sliced import (
    non_max_suppression,
    offset_detection,
    sliced_predict,
)
from urban_canopy.tiling import TileWindow


def det(xmin: float, ymin: float, xmax: float, ymax: float, score: float) -> Detection:
    return Detection(Box(xmin, ymin, xmax, ymax), score)


class TestOffset:
    def test_tile_local_box_becomes_global(self) -> None:
        shifted = offset_detection(det(10, 20, 30, 40, 0.9), TileWindow(100, 200, 200, 300))
        assert shifted.box == Box(110, 220, 130, 240)
        assert shifted.score == 0.9


class TestNms:
    def test_keeps_highest_and_drops_overlap(self) -> None:
        detections = [det(0, 0, 40, 40, 0.9), det(2, 2, 42, 42, 0.7), det(500, 500, 540, 540, 0.6)]
        kept = non_max_suppression(detections, iou_threshold=0.5)
        assert len(kept) == 2
        assert kept[0].score == 0.9
        assert {round(d.box.xmin) for d in kept} == {0, 500}

    def test_disjoint_boxes_all_survive(self) -> None:
        detections = [det(0, 0, 10, 10, 0.9), det(100, 100, 110, 110, 0.8)]
        assert len(non_max_suppression(detections)) == 2

    def test_empty_input(self) -> None:
        assert non_max_suppression([]) == []


class TestSlicedPredict:
    def test_merges_tiles_and_dedupes_seam_crown(self) -> None:
        # A crown centred on the seam between two overlapping tiles is detected
        # in both; after offsetting, the two boxes coincide → NMS keeps one.
        def detector(window: TileWindow) -> list[Detection]:
            # Emit a box at global (90,10)-(110,30) whenever the window covers it.
            gx0, gy0, gx1, gy1 = 90, 10, 110, 30
            if window.x0 <= gx0 and window.x1 >= gx1 and window.y1 >= gy1:
                return [
                    det(gx0 - window.x0, gy0 - window.y0, gx1 - window.x0, gy1 - window.y0, 0.9)
                ]
            return []

        merged = sliced_predict(200, 100, detector, tile_size=120, overlap=60)
        assert len(merged) == 1
        assert merged[0].box == Box(90, 10, 110, 30)

    def test_distinct_crowns_in_different_tiles_are_all_returned(self) -> None:
        # One crown near each corner; each falls in exactly one tile.
        crowns = [(10, 10), (180, 10), (10, 90), (180, 90)]

        def detector(window: TileWindow) -> list[Detection]:
            out = []
            for cx, cy in crowns:
                if window.x0 <= cx < window.x1 and window.y0 <= cy < window.y1:
                    out.append(
                        det(
                            cx - window.x0,
                            cy - window.y0,
                            cx - window.x0 + 5,
                            cy - window.y0 + 5,
                            0.8,
                        )
                    )
            return out

        merged = sliced_predict(200, 100, detector, tile_size=100, overlap=0)
        assert len(merged) == 4

    def test_no_detections_yields_empty(self) -> None:
        assert sliced_predict(200, 200, lambda _window: [], tile_size=100, overlap=0) == []
