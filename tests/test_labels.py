"""Converter tests: VOC ↔ YOLO ↔ COCO, with byte-exact text round-trips."""

from pathlib import Path

import pytest

from urban_canopy.labels import (
    Box,
    ImageAnnotation,
    coco_json,
    from_coco,
    from_yolo_text,
    parse_voc_xml,
    to_coco,
    to_yolo_text,
    write_yolo_labels,
)

FIXTURES = Path(__file__).parent / "fixtures"
VOC = FIXTURES / "2019_TEAK_4_315000_4094000_image_crop.xml"


@pytest.fixture
def annotation() -> ImageAnnotation:
    return parse_voc_xml(VOC)


class TestVocParsing:
    def test_parses_geometry_and_boxes(self, annotation: ImageAnnotation) -> None:
        assert annotation.image_name == "2019_TEAK_4_315000_4094000_image_crop.tif"
        assert (annotation.width, annotation.height) == (400, 400)
        assert len(annotation.boxes) == 3
        assert annotation.boxes[0] == Box(12.0, 30.0, 110.0, 128.0)

    def test_missing_size_is_an_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.xml"
        bad.write_text("<annotation><filename>x.tif</filename></annotation>", encoding="utf-8")
        with pytest.raises(ValueError, match="size"):
            parse_voc_xml(bad)


class TestYoloRoundTrip:
    def test_voc_to_yolo_to_boxes_is_lossless_within_precision(
        self, annotation: ImageAnnotation
    ) -> None:
        text = to_yolo_text(annotation)
        recovered = from_yolo_text(text, image_name=annotation.image_name, width=400, height=400)
        for original, back in zip(annotation.boxes, recovered.boxes, strict=True):
            assert back.xmin == pytest.approx(original.xmin, abs=0.001 * 400)
            assert back.ymax == pytest.approx(original.ymax, abs=0.001 * 400)

    def test_yolo_text_round_trip_is_byte_exact(self, annotation: ImageAnnotation) -> None:
        text = to_yolo_text(annotation)
        recovered = from_yolo_text(text, image_name="x.tif", width=400, height=400)
        assert to_yolo_text(recovered) == text

    def test_empty_annotation_yields_empty_file(self) -> None:
        empty = ImageAnnotation("empty.tif", 400, 400, ())
        assert to_yolo_text(empty) == ""
        assert from_yolo_text("", image_name="empty.tif", width=400, height=400).boxes == ()

    def test_garbage_lines_raise_with_line_number(self) -> None:
        with pytest.raises(ValueError, match="line 2"):
            from_yolo_text(
                "0 0.5 0.5 0.1 0.1\n1 0.5 0.5 0.1 0.1\n",
                image_name="x.tif",
                width=400,
                height=400,
            )

    def test_write_yolo_labels_names_files_after_images(
        self, annotation: ImageAnnotation, tmp_path: Path
    ) -> None:
        paths = write_yolo_labels([annotation], tmp_path)
        assert [path.name for path in paths] == ["2019_TEAK_4_315000_4094000_image_crop.txt"]
        assert paths[0].read_text(encoding="utf-8") == to_yolo_text(annotation)


class TestCocoRoundTrip:
    def test_coco_json_round_trip_is_byte_exact(self, annotation: ImageAnnotation) -> None:
        payload = to_coco([annotation])
        text = coco_json(payload)
        assert coco_json(to_coco(from_coco(payload))) == text

    def test_coco_structure(self, annotation: ImageAnnotation) -> None:
        payload = to_coco([annotation, ImageAnnotation("2019_NIWO_1.tif", 400, 400, ())])
        images = payload["images"]
        annotations = payload["annotations"]
        assert isinstance(images, list)
        assert isinstance(annotations, list)
        assert len(images) == 2
        assert len(annotations) == 3
        assert {entry["image_id"] for entry in annotations} == {1}
        assert payload["categories"] == [{"id": 1, "name": "Tree"}]

    def test_boxes_survive_coco(self, annotation: ImageAnnotation) -> None:
        recovered = from_coco(to_coco([annotation]))[0]
        for original, back in zip(annotation.boxes, recovered.boxes, strict=True):
            assert back.xmin == pytest.approx(original.xmin, abs=0.01)
            assert back.area == pytest.approx(original.area, rel=0.001)
