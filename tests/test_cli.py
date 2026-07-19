"""CLI tests exercised on the committed samples (no GPU, no network)."""

import json
from pathlib import Path

from typer.testing import CliRunner

from urban_canopy.cli import app
from urban_canopy.evaluate import Detection
from urban_canopy.labels import Box, parse_voc_xml
from urban_canopy.predictions import detections_to_json

SAMPLES = Path("data/sample")
runner = CliRunner()


def write_perfect_predictions(path: Path) -> None:
    """Predictions that exactly match the sample ground truth."""
    per_image = {}
    for xml_path in SAMPLES.glob("*.xml"):
        annotation = parse_voc_xml(xml_path)
        per_image[annotation.image_name] = [Detection(box, 0.9) for box in annotation.boxes]
    path.write_text(detections_to_json(per_image), encoding="utf-8")


class TestBuildDataset:
    def test_yolo_layout_from_samples(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "build-dataset",
                "--images",
                str(SAMPLES),
                "--annotations",
                str(SAMPLES),
                "--out",
                str(tmp_path / "ds"),
                "--layout",
                "yolo",
                "--val-fraction",
                "0.5",
            ],
        )
        assert result.exit_code == 0
        assert (tmp_path / "ds" / "data.yaml").exists()
        assert "1 train / 1 val" in result.output


class TestScore:
    def test_perfect_predictions_score_one(self, tmp_path: Path) -> None:
        preds = tmp_path / "preds.json"
        write_perfect_predictions(preds)
        result = runner.invoke(
            app, ["score", str(preds), "--annotations", str(SAMPLES), "--name", "test"]
        )
        assert result.exit_code == 0
        assert "mAP50=1.0" in result.output

    def test_update_metrics_writes_row(self, tmp_path: Path, monkeypatch) -> None:
        annotations = SAMPLES.resolve()  # resolve before chdir
        preds = tmp_path / "preds.json"
        write_perfect_predictions(preds)
        monkeypatch.chdir(tmp_path)
        (tmp_path / "results").mkdir()
        (tmp_path / "results" / "metrics.json").write_text(
            json.dumps({"dataset": {}, "models": [], "sahi_effect": {}}), encoding="utf-8"
        )
        result = runner.invoke(
            app,
            [
                "score",
                str(preds),
                "--annotations",
                str(annotations),
                "--name",
                "YOLO26-s",
                "--update-metrics",
            ],
        )
        assert result.exit_code == 0
        metrics = json.loads((tmp_path / "results" / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["models"][0]["name"] == "YOLO26-s"
        assert metrics["models"][0]["status"] == "measured"


class TestToGeojson:
    def test_missing_image_key_errors(self, tmp_path: Path) -> None:
        preds = tmp_path / "preds.json"
        preds.write_text(detections_to_json({"other.tif": [Detection(Box(0, 0, 1, 1), 0.5)]}))
        result = runner.invoke(
            app,
            [
                "to-geojson",
                str(preds),
                "--raster",
                str(SAMPLES / "BLAN_005_2019.tif"),
                "--image-name",
                "absent.tif",
            ],
        )
        assert result.exit_code == 1
        assert "not in predictions" in result.output
