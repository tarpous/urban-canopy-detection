"""Training-wiring smoke tests (marked `smoke`).

These drive the training scripts' ``--smoke`` path end-to-end on CPU over the
committed sample tiles: dataset build → 1-epoch train → predict → score. They
prove the GPU scripts stay wired to the tested package without needing a GPU or
the multi-GB benchmark download. Skipped automatically when the `train` group
(ultralytics/torch) is not installed, so the default CPU CI stays lean; a
separate CI job installs `--group train` and runs `pytest -m smoke`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

REPO = Path(__file__).resolve().parents[1]

ultralytics = pytest.importorskip("ultralytics", reason="train group not installed")


def run_script(*script_args: str) -> subprocess.CompletedProcess[str]:
    # Ultralytics prints Unicode progress bars; force UTF-8 decoding so a
    # Windows cp1252 console default doesn't crash the capture.
    return subprocess.run(
        [sys.executable, *script_args],
        cwd=REPO,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
    )


def test_train_yolo_smoke_wiring() -> None:
    result = run_script("scripts/train_yolo.py", "--smoke")
    assert result.returncode == 0, result.stderr[-2000:]
    assert "smoke OK" in result.stdout


def test_run_baseline_smoke_wiring() -> None:
    result = run_script("scripts/run_baseline.py", "--smoke")
    assert result.returncode == 0, result.stderr[-2000:]
    assert "smoke OK" in result.stdout


def test_train_rfdetr_smoke_wiring() -> None:
    # RF-DETR is an optional extra (`uv pip install rfdetr`), not in the locked
    # train group; skip when it isn't installed so the lean CI job stays green.
    pytest.importorskip("rfdetr", reason="rfdetr not installed (optional extra)")
    result = run_script("scripts/train_rfdetr.py", "--smoke")
    assert result.returncode == 0, result.stderr[-3000:]
    assert "smoke OK" in result.stdout
