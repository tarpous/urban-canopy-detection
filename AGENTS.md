# AGENTS.md

## Setup

- Python 3.12, uv-managed. Create the environment once: `uv venv .venv --python 3.12`, then activate it
  (`source .venv/Scripts/activate` on Windows, `source .venv/bin/activate` on Linux/macOS) and run `uv sync`.
- Every `python`/`pip` command runs inside the activated `.venv`. Add dependencies with `uv add <pkg>`
  (or `uv add --dev` for tooling) so `pyproject.toml` and `uv.lock` stay in sync.

## Commands

- Lint: `uv run ruff check .` and `uv run ruff format --check .` (fix with `ruff check --fix` / `ruff format`)
- Types: `uv run mypy` (strict, scoped to `src/`)
- Tests: `uv run pytest` (offline, runs from `data/sample/` + `tests/fixtures/`)
- CI (`.github/workflows/ci.yml`) runs exactly these gates on ubuntu-latest / Python 3.12.
- When chaining gates in a shell, use `set -o pipefail` — piping through `tail` otherwise masks failures.

## Layout

- `src/urban_canopy/` — package code, fully typed (`py.typed`): label converters, tiling,
  splits, evaluation, geospatial outputs
- `notebooks/` — parameterized Colab/Kaggle T4 fine-tune notebooks (outputs cleared);
  metrics are exported to `results/*.json`, never copied by hand
- `data/sample/` — small committed fixtures (≤5 MB); `data/raw/` is gitignored and
  populated by `scripts/download_neon.py`
- `scripts/` — dataset download + results-table generation
- `results/` — generated metrics; the only source of README numbers

## CLI

`uv run canopy` (see `--help`): `build-dataset` (VOC → YOLO/COCO, site-disjoint),
`score` (predictions JSON vs benchmark, `--update-metrics` writes results/metrics.json),
`to-geojson` (georeference one image's predictions), `make-table` (regenerate README table).

## Conventions

- Conventional-commit messages (`feat:`, `fix:`, `test:`, `docs:`, `chore:`, `ci:`). No co-author trailers.
- GPU work never runs locally — training/fine-tuning lives in `notebooks/` (ruff-excluded) for Colab/Kaggle T4.
- Train/val splits are by NEON site (geographic blocking); the leakage test must always pass.
- README numbers come only from `results/metrics.json` via `scripts/make_results_table.py`.
- The Gradio app (`app/`) reuses the tested package and boots in a synthetic mode without weights.
- No personal details of the repo owner in committed content.
