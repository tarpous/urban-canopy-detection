"""Dataset-builder tests running end-to-end on the committed NEON samples."""

import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from urban_canopy.dataset import build_coco_dataset, build_yolo_dataset, discover_pairs
from urban_canopy.labels import from_yolo_text, parse_voc_xml

SAMPLES = Path("data/sample")


@pytest.fixture(scope="module")
def pairs() -> list[tuple[Path, Path]]:
    found = discover_pairs(SAMPLES, SAMPLES)
    assert len(found) == 2  # BLAN + SJER, both annotated
    return found


class TestDiscovery:
    def test_pairs_match_by_stem(self, pairs: list[tuple[Path, Path]]) -> None:
        for image_path, xml_path in pairs:
            assert image_path.stem == xml_path.stem

    def test_sample_annotations_agree_with_pixels(self, pairs: list[tuple[Path, Path]]) -> None:
        for image_path, xml_path in pairs:
            annotation = parse_voc_xml(xml_path)
            with Image.open(image_path) as image:
                assert image.size == (annotation.width, annotation.height)


class TestYoloBuild:
    def test_full_yolo_layout_from_samples(
        self, pairs: list[tuple[Path, Path]], tmp_path: Path
    ) -> None:
        manifest = build_yolo_dataset(pairs, tmp_path, val_fraction=0.5, seed=0)
        assert manifest.n_train_images == 1
        assert manifest.n_val_images == 1
        assert set(manifest.train_sites) | set(manifest.val_sites) == {"BLAN", "SJER"}
        assert not set(manifest.train_sites) & set(manifest.val_sites)

        data = yaml.safe_load((tmp_path / "data.yaml").read_text(encoding="utf-8"))
        assert data["names"] == {0: "Tree"}
        for side, count in [("train", manifest.n_train_boxes), ("val", manifest.n_val_boxes)]:
            labels = list((tmp_path / "labels" / side).glob("*.txt"))
            assert len(labels) == 1
            parsed = from_yolo_text(
                labels[0].read_text(encoding="utf-8"), image_name="x", width=400, height=400
            )
            assert len(parsed.boxes) == count
            images = list((tmp_path / "images" / side).iterdir())
            assert len(images) == 1

    def test_tiled_build_writes_real_crops(
        self, pairs: list[tuple[Path, Path]], tmp_path: Path
    ) -> None:
        manifest = build_yolo_dataset(
            pairs, tmp_path, val_fraction=0.5, seed=0, tile_size=200, overlap=0
        )
        train_images = sorted((tmp_path / "images" / "train").glob("*.png"))
        assert len(train_images) == 4  # 400x400 sample → 2x2 tiles of 200
        with Image.open(train_images[0]) as crop:
            assert crop.size == (200, 200)
        assert manifest.n_train_images == 4
        # tiling redistributes boxes but never invents extras beyond duplicates
        assert manifest.n_train_boxes >= 1


class TestCocoBuild:
    def test_roboflow_style_layout(self, pairs: list[tuple[Path, Path]], tmp_path: Path) -> None:
        manifest = build_coco_dataset(pairs, tmp_path, val_fraction=0.5, seed=0)
        for side in ("train", "valid"):
            payload = json.loads(
                (tmp_path / side / "_annotations.coco.json").read_text(encoding="utf-8")
            )
            assert payload["categories"] == [{"id": 1, "name": "Tree"}]
            assert len(payload["images"]) == 1
            image_entry = payload["images"][0]
            assert (tmp_path / side / image_entry["file_name"]).exists()
        total_boxes = manifest.n_train_boxes + manifest.n_val_boxes
        assert total_boxes == 33 + 8  # BLAN + SJER crown counts
