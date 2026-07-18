"""Label formats and converters: Pascal-VOC XML ↔ YOLO ↔ COCO.

The NEON tree-crown benchmark annotates crowns as Pascal-VOC XML boxes; YOLO
training wants normalized ``class cx cy w h`` text files; COCO JSON is the
lingua franca of the DETR family. Conversions are canonical and round-trip
**byte-exactly** in the text domain (``to_yolo(from_yolo(text)) == text`` and
likewise for COCO with sorted keys), which is what the test suite enforces —
a silent half-pixel drift in a converter is the classic way detection metrics
go quietly wrong.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

#: Single-class problem: every box is a tree crown.
CLASS_NAME = "Tree"
CLASS_ID = 0


@dataclass(frozen=True, slots=True)
class Box:
    """Axis-aligned box in pixel coordinates (VOC convention, inclusive)."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def area(self) -> float:
        return max(self.width, 0.0) * max(self.height, 0.0)


@dataclass(frozen=True, slots=True)
class ImageAnnotation:
    """All crown boxes of one image plus the image geometry."""

    image_name: str
    width: int
    height: int
    boxes: tuple[Box, ...]


def parse_voc_xml(path: Path) -> ImageAnnotation:
    """Parse one NEON-style Pascal-VOC annotation file."""
    root = ET.parse(path).getroot()
    filename = root.findtext("filename", default=path.stem)
    size = root.find("size")
    if size is None:
        raise ValueError(f"{path}: missing <size> element")
    width = int(size.findtext("width", default="0"))
    height = int(size.findtext("height", default="0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"{path}: invalid image size {width}x{height}")

    boxes = []
    for obj in root.iter("object"):
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        boxes.append(
            Box(
                xmin=float(bndbox.findtext("xmin", default="0")),
                ymin=float(bndbox.findtext("ymin", default="0")),
                xmax=float(bndbox.findtext("xmax", default="0")),
                ymax=float(bndbox.findtext("ymax", default="0")),
            )
        )
    return ImageAnnotation(image_name=filename, width=width, height=height, boxes=tuple(boxes))


def to_yolo_text(annotation: ImageAnnotation) -> str:
    """YOLO label text: one ``class cx cy w h`` line per box, normalized, 6 dp."""
    lines = []
    for box in annotation.boxes:
        cx = (box.xmin + box.xmax) / 2.0 / annotation.width
        cy = (box.ymin + box.ymax) / 2.0 / annotation.height
        w = box.width / annotation.width
        h = box.height / annotation.height
        lines.append(f"{CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")


def from_yolo_text(text: str, *, image_name: str, width: int, height: int) -> ImageAnnotation:
    """Inverse of :func:`to_yolo_text` for the same image geometry."""
    boxes = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        fields = stripped.split()
        if len(fields) != 5 or fields[0] != str(CLASS_ID):
            raise ValueError(f"line {line_no}: expected '{CLASS_ID} cx cy w h', got {line!r}")
        cx, cy, w, h = (float(value) for value in fields[1:])
        boxes.append(
            Box(
                xmin=(cx - w / 2.0) * width,
                ymin=(cy - h / 2.0) * height,
                xmax=(cx + w / 2.0) * width,
                ymax=(cy + h / 2.0) * height,
            )
        )
    return ImageAnnotation(image_name=image_name, width=width, height=height, boxes=tuple(boxes))


def to_coco(annotations: list[ImageAnnotation]) -> dict[str, object]:
    """COCO detection payload (single ``Tree`` category, xywh boxes)."""
    images = []
    coco_annotations = []
    annotation_id = 1
    for image_id, annotation in enumerate(annotations, start=1):
        images.append(
            {
                "id": image_id,
                "file_name": annotation.image_name,
                "width": annotation.width,
                "height": annotation.height,
            }
        )
        for box in annotation.boxes:
            bbox = [
                round(box.xmin, 2),
                round(box.ymin, 2),
                round(box.width, 2),
                round(box.height, 2),
            ]
            coco_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": bbox,
                    # Area from the *rounded* bbox keeps serialization idempotent:
                    # to_coco(from_coco(p)) must reproduce p byte-for-byte.
                    "area": round(bbox[2] * bbox[3], 2),
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    return {
        "images": images,
        "annotations": coco_annotations,
        "categories": [{"id": 1, "name": CLASS_NAME}],
    }


def from_coco(payload: dict[str, object]) -> list[ImageAnnotation]:
    """Inverse of :func:`to_coco`."""
    images = payload["images"]
    raw_annotations = payload["annotations"]
    if not isinstance(images, list) or not isinstance(raw_annotations, list):
        raise ValueError("malformed COCO payload: images/annotations must be lists")

    boxes_by_image: dict[int, list[Box]] = {}
    for entry in raw_annotations:
        xmin, ymin, width, height = entry["bbox"]
        boxes_by_image.setdefault(int(entry["image_id"]), []).append(
            Box(xmin=xmin, ymin=ymin, xmax=xmin + width, ymax=ymin + height)
        )
    return [
        ImageAnnotation(
            image_name=str(image["file_name"]),
            width=int(image["width"]),
            height=int(image["height"]),
            boxes=tuple(boxes_by_image.get(int(image["id"]), [])),
        )
        for image in images
    ]


def coco_json(payload: dict[str, object]) -> str:
    """Canonical COCO JSON text (sorted keys) so round-trips are byte-exact."""
    return json.dumps(payload, sort_keys=True, indent=2) + "\n"


def write_yolo_labels(annotations: list[ImageAnnotation], out_dir: Path) -> list[Path]:
    """Write one YOLO ``.txt`` per image; returns the written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for annotation in annotations:
        stem = Path(annotation.image_name).stem
        path = out_dir / f"{stem}.txt"
        path.write_text(to_yolo_text(annotation), encoding="utf-8")
        written.append(path)
    return written
