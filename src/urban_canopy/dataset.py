"""Assemble detector-ready datasets from NEON images + VOC annotations.

This is the module the fine-tune notebooks call: everything here runs and is
tested locally on the committed samples, so the only untested surface on the
GPU side is the trainer invocation itself. Two layouts are produced from the
same site-disjoint split:

- **YOLO** (ultralytics): ``images/{train,val}`` + ``labels/{train,val}`` +
  ``data.yaml``;
- **COCO** (RF-DETR / roboflow convention): ``{train,valid}/`` folders each
  holding the images plus ``_annotations.coco.json``.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image

from urban_canopy.labels import (
    ImageAnnotation,
    coco_json,
    parse_voc_xml,
    to_coco,
    to_yolo_text,
)
from urban_canopy.splits import Split, split_by_site
from urban_canopy.tiling import compute_tile_grid, tile_annotation

Pair = tuple[Path, Path]  # (image, voc xml)


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """What a build produced — recorded into results JSONs by the notebooks."""

    n_train_images: int
    n_val_images: int
    n_train_boxes: int
    n_val_boxes: int
    train_sites: tuple[str, ...]
    val_sites: tuple[str, ...]


def discover_pairs(images_dir: Path, annotations_dir: Path) -> list[Pair]:
    """Match images to annotation files by stem; unannotated images are skipped."""
    xml_by_stem = {path.stem: path for path in sorted(annotations_dir.glob("*.xml"))}
    pairs = []
    for image_path in sorted(images_dir.glob("*.tif")) + sorted(images_dir.glob("*.png")):
        xml_path = xml_by_stem.get(image_path.stem)
        if xml_path is not None:
            pairs.append((image_path, xml_path))
    return pairs


def _load(pairs: list[Pair]) -> list[tuple[Path, ImageAnnotation]]:
    loaded = []
    for image_path, xml_path in pairs:
        annotation = parse_voc_xml(xml_path)
        # Trust the pixels over the XML header if they disagree.
        with Image.open(image_path) as image:
            width, height = image.size
        if (width, height) != (annotation.width, annotation.height):
            annotation = ImageAnnotation(annotation.image_name, width, height, annotation.boxes)
        loaded.append((image_path, annotation))
    return loaded


def _split_pairs(
    loaded: list[tuple[Path, ImageAnnotation]], *, val_fraction: float, seed: int
) -> tuple[Split, dict[str, tuple[Path, ImageAnnotation]]]:
    by_name = {image_path.name: (image_path, annotation) for image_path, annotation in loaded}
    split = split_by_site(sorted(by_name), val_fraction=val_fraction, seed=seed)
    return split, by_name


def _tile_to_disk(
    image_path: Path,
    annotation: ImageAnnotation,
    out_images: Path,
    *,
    tile_size: int,
    overlap: int,
    min_visibility: float,
) -> list[ImageAnnotation]:
    """Crop tiles out of one image, write them, return their annotations."""
    tiled = tile_annotation(
        annotation, tile_size=tile_size, overlap=overlap, min_visibility=min_visibility
    )
    grid = compute_tile_grid(
        annotation.width, annotation.height, tile_size=tile_size, overlap=overlap
    )
    with Image.open(image_path) as image:
        for window, tile in zip(grid, tiled, strict=True):
            crop = image.crop((window.x0, window.y0, window.x1, window.y1))
            crop.save(out_images / f"{Path(tile.image_name).stem}.png")
    return [
        ImageAnnotation(f"{Path(tile.image_name).stem}.png", tile.width, tile.height, tile.boxes)
        for tile in tiled
    ]


def _materialize(
    names: tuple[str, ...],
    by_name: dict[str, tuple[Path, ImageAnnotation]],
    out_images: Path,
    *,
    tile_size: int | None,
    overlap: int,
    min_visibility: float,
) -> list[ImageAnnotation]:
    """Copy (or tile) the images of one split side; return final annotations."""
    out_images.mkdir(parents=True, exist_ok=True)
    annotations: list[ImageAnnotation] = []
    for name in names:
        image_path, annotation = by_name[name]
        if tile_size is None:
            shutil.copy2(image_path, out_images / image_path.name)
            annotations.append(annotation)
        else:
            annotations.extend(
                _tile_to_disk(
                    image_path,
                    annotation,
                    out_images,
                    tile_size=tile_size,
                    overlap=overlap,
                    min_visibility=min_visibility,
                )
            )
    return annotations


def _manifest(
    split: Split, train: list[ImageAnnotation], val: list[ImageAnnotation]
) -> DatasetManifest:
    return DatasetManifest(
        n_train_images=len(train),
        n_val_images=len(val),
        n_train_boxes=sum(len(annotation.boxes) for annotation in train),
        n_val_boxes=sum(len(annotation.boxes) for annotation in val),
        train_sites=tuple(sorted(split.train_sites)),
        val_sites=tuple(sorted(split.val_sites)),
    )


def build_yolo_dataset(
    pairs: list[Pair],
    out_dir: Path,
    *,
    val_fraction: float = 0.2,
    seed: int = 0,
    tile_size: int | None = None,
    overlap: int = 64,
    min_visibility: float = 0.4,
) -> DatasetManifest:
    """Site-disjoint YOLO layout with ``data.yaml`` (ultralytics convention)."""
    split, by_name = _split_pairs(_load(pairs), val_fraction=val_fraction, seed=seed)
    sides: dict[str, list[ImageAnnotation]] = {}
    for side, names in [("train", split.train), ("val", split.val)]:
        annotations = _materialize(
            names,
            by_name,
            out_dir / "images" / side,
            tile_size=tile_size,
            overlap=overlap,
            min_visibility=min_visibility,
        )
        labels_dir = out_dir / "labels" / side
        labels_dir.mkdir(parents=True, exist_ok=True)
        for annotation in annotations:
            stem = Path(annotation.image_name).stem
            (labels_dir / f"{stem}.txt").write_text(to_yolo_text(annotation), encoding="utf-8")
        sides[side] = annotations

    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "Tree"},
    }
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    return _manifest(split, sides["train"], sides["val"])


def build_coco_dataset(
    pairs: list[Pair],
    out_dir: Path,
    *,
    val_fraction: float = 0.2,
    seed: int = 0,
    tile_size: int | None = None,
    overlap: int = 64,
    min_visibility: float = 0.4,
) -> DatasetManifest:
    """Site-disjoint COCO layout (``train``/``valid`` folders, roboflow style)."""
    split, by_name = _split_pairs(_load(pairs), val_fraction=val_fraction, seed=seed)
    sides: dict[str, list[ImageAnnotation]] = {}
    for side, names in [("train", split.train), ("valid", split.val)]:
        side_dir = out_dir / side
        annotations = _materialize(
            names,
            by_name,
            side_dir,
            tile_size=tile_size,
            overlap=overlap,
            min_visibility=min_visibility,
        )
        (side_dir / "_annotations.coco.json").write_text(
            coco_json(to_coco(annotations)), encoding="utf-8"
        )
        sides[side] = annotations
    return _manifest(split, sides["train"], sides["valid"])
