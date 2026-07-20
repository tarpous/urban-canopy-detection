"""``canopy`` — CPU-side command line for data prep, scoring and geo export.

Training lives in ``scripts/train_*.py`` (local GPU); everything here runs on a
laptop: build detector-ready datasets, score a predictions file against the
benchmark, refresh the README table, and georeference predictions to GeoJSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    help="Tree-crown detection: data prep, scoring, geo export.", no_args_is_help=True
)


@app.command("build-dataset")
def build_dataset(
    images: Annotated[Path, typer.Option(exists=True, help="Directory of image tiles.")],
    annotations: Annotated[Path, typer.Option(exists=True, help="Directory of VOC XML files.")],
    out: Annotated[Path, typer.Option(help="Output dataset directory.")],
    layout: Annotated[str, typer.Option(help="'yolo' or 'coco'.")] = "yolo",
    tile_size: Annotated[int, typer.Option(help="Tile size (0 = whole images).")] = 0,
    overlap: Annotated[int, typer.Option(help="Tile overlap in pixels.")] = 64,
    val_fraction: Annotated[float, typer.Option(help="Validation fraction (by site).")] = 0.2,
    seed: Annotated[int, typer.Option(help="Split seed.")] = 0,
) -> None:
    """Build a site-disjoint YOLO or COCO dataset from images + VOC annotations."""
    from urban_canopy.dataset import build_coco_dataset, build_yolo_dataset, discover_pairs

    pairs = discover_pairs(images, annotations)
    if not pairs:
        typer.echo("no matching image/annotation pairs found", err=True)
        raise typer.Exit(code=1)
    builder = build_yolo_dataset if layout == "yolo" else build_coco_dataset
    manifest = builder(
        pairs,
        out,
        val_fraction=val_fraction,
        seed=seed,
        tile_size=tile_size or None,
        overlap=overlap,
    )
    typer.echo(
        f"{layout} dataset → {out}: {manifest.n_train_images} train / "
        f"{manifest.n_val_images} val images, sites "
        f"train={list(manifest.train_sites)} val={list(manifest.val_sites)}"
    )


@app.command()
def score(
    predictions: Annotated[Path, typer.Argument(exists=True, help="Predictions JSON.")],
    annotations: Annotated[Path, typer.Option(exists=True, help="VOC ground-truth directory.")],
    name: Annotated[str, typer.Option(help="Model name for the metrics row.")] = "model",
    inference: Annotated[str, typer.Option(help="Inference description.")] = "whole-image",
    update_metrics: Annotated[
        bool, typer.Option("--update-metrics", help="Write the row into results/metrics.json.")
    ] = False,
) -> None:
    """Score a predictions file against the benchmark ground truth."""
    from urban_canopy.predictions import (
        detections_from_json,
        load_ground_truth,
        result_to_model_row,
        score_predictions,
    )

    preds = detections_from_json(predictions.read_text(encoding="utf-8"))
    truth = load_ground_truth(annotations)
    result = score_predictions(preds, truth)
    typer.echo(
        f"{name}: mAP50={result.map_50} mAP50-95={result.map_50_95} "
        f"P={result.precision} R={result.recall} recall_by_size={result.recall_by_size}"
    )
    if update_metrics:
        metrics_path = Path("results/metrics.json")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        row = result_to_model_row(name, result, inference=inference)
        metrics["models"] = [model for model in metrics["models"] if model["name"] != name] + [row]
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        typer.echo(f"updated {metrics_path}")


@app.command("to-geojson")
def to_geojson(
    predictions: Annotated[Path, typer.Argument(exists=True, help="Predictions JSON.")],
    raster: Annotated[
        Path, typer.Option(exists=True, help="Source GeoTIFF for the CRS/transform.")
    ],
    image_name: Annotated[str, typer.Option(help="Which image key in the predictions to export.")],
    out: Annotated[Path, typer.Option(help="Output GeoJSON path.")] = Path("crowns.geojson"),
) -> None:
    """Georeference one image's predictions to WGS84 GeoJSON."""
    from urban_canopy.geo import GeoReference, detections_to_geodataframe, write_geojson
    from urban_canopy.predictions import detections_from_json

    preds = detections_from_json(predictions.read_text(encoding="utf-8"))
    if image_name not in preds:
        typer.echo(f"{image_name!r} not in predictions (have {list(preds)[:5]}…)", err=True)
        raise typer.Exit(code=1)
    ref = GeoReference.from_raster(raster)
    frame = detections_to_geodataframe(preds[image_name], ref)
    write_geojson(frame, out)
    typer.echo(f"wrote {len(frame)} crowns → {out}")


@app.command("make-table")
def make_table() -> None:
    """Regenerate the README results table from results/metrics.json."""
    import runpy

    runpy.run_path("scripts/make_results_table.py", run_name="__main__")


if __name__ == "__main__":
    app()
