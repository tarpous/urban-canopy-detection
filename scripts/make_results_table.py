"""Render the README model-comparison table from ``results/metrics.json``.

Numbers are never hand-typed: the notebooks and the baseline script write
``results/metrics.json``; this script turns it into the markdown injected
between the ``<!-- results:begin -->`` / ``<!-- results:end -->`` markers.
Runs fine on the placeholder file too, so the README is honest ("pending")
before any GPU run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

RESULTS = Path("results")


def cell(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render(metrics: dict) -> str:
    dataset = metrics["dataset"]
    lines = [
        f"**Benchmark:** {dataset['benchmark']} · **split:** {dataset['split']}",
        "",
        "| Model | mAP@50 | mAP@[.5:.95] | P | R | R small | R med | R large | Inference |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model in metrics["models"]:
        pending = " 🕒" if model["status"] != "measured" else ""
        lines.append(
            f"| {model['name']}{pending} "
            f"| {cell(model['map_50'])} | {cell(model['map_50_95'])} "
            f"| {cell(model['precision'])} | {cell(model['recall'])} "
            f"| {cell(model['recall_small'])} | {cell(model['recall_medium'])} "
            f"| {cell(model['recall_large'])} | {model['inference']} |"
        )
    sahi = metrics["sahi_effect"]
    lines.append("")
    lines.append(
        f"**SAHI effect ({sahi['model']}):** whole-image mAP@50 "
        f"{cell(sahi['whole_image_map_50'])} → sliced {cell(sahi['sliced_map_50'])}."
    )
    if any(model["status"] != "measured" for model in metrics["models"]):
        lines.append("")
        lines.append("🕒 = awaiting the T4 fine-tune run; see `notebooks/`.")
    return "\n".join(lines) + "\n"


def inject_into_readme(table_markdown: str, readme: Path = Path("README.md")) -> None:
    begin, end = "<!-- results:begin -->", "<!-- results:end -->"
    content = readme.read_text(encoding="utf-8")
    if begin not in content or end not in content:
        print(f"markers not found in {readme}; skipped injection")
        return
    head, rest = content.split(begin, 1)
    _, tail = rest.split(end, 1)
    readme.write_text(f"{head}{begin}\n{table_markdown}{end}{tail}", encoding="utf-8")
    print(f"injected results into {readme}")


def main() -> None:
    metrics = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
    table = render(metrics)
    (RESULTS / "table.md").write_text(table, encoding="utf-8")
    inject_into_readme(table)
    print(table)


if __name__ == "__main__":
    main()
