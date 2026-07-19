"""Download the NeonTreeEvaluation benchmark (weecology), pinned and verified.

Sources, pinned 2026-07-18:

- Zenodo record 5914554, v0.2.2 ("Data for the NeonTreeEvaluation Benchmark"):
  ``annotations.zip`` (0.6 MB), ``evaluation.zip`` (3.9 GB, RGB+HSI+LiDAR+CHM),
  ``training.zip`` (4.5 GB). MD5 checksums below come from the Zenodo API.
- Individual evaluation RGB tiles are also served from the benchmark's GitHub
  repository, which is how the two committed ``data/sample`` tiles were made.

The laptop path only ever needs ``--annotations`` (+ the committed samples);
the full zips are meant for Colab/Kaggle, where the fine-tune notebooks call
this script with ``--evaluation``/``--training``.
"""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

import requests

ZENODO_RECORD = "5914554"
ZENODO_FILES: dict[str, str] = {
    # name -> md5 (Zenodo API, record v0.2.2)
    "annotations.zip": "f577f0948a259d474ee0199fb8d76524",
    "evaluation.zip": "f0c444fc59cf0115ce8c981a576fab77",
    "training.zip": "8d71412f3dfe9f055e8183f5faed905f",
}
GITHUB_RAW = "https://raw.githubusercontent.com/weecology/NeonTreeEvaluation/master"
SAMPLE_FILES = [
    "evaluation/RGB/BLAN_005_2019.tif",
    "annotations/BLAN_005_2019.xml",
    "evaluation/RGB/2018_SJER_3_252000_4104000_image_628.tif",
    "annotations/2018_SJER_3_252000_4104000_image_628.xml",
]

RAW = Path("data/raw")


def zenodo_url(name: str) -> str:
    return f"https://zenodo.org/records/{ZENODO_RECORD}/files/{name}?download=1"


def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path, expected_md5: str | None = None) -> Path:
    if dest.exists() and (expected_md5 is None or md5_of(dest) == expected_md5):
        print(f"{dest} already present")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url} -> {dest}")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        part = dest.with_suffix(dest.suffix + ".part")
        with part.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
        part.replace(dest)
    if expected_md5 is not None:
        actual = md5_of(dest)
        if actual != expected_md5:
            dest.unlink()
            raise RuntimeError(f"{dest}: md5 {actual} != expected {expected_md5}")
        print(f"{dest}: md5 verified")
    return dest


def fetch_zip(name: str, extract: bool) -> None:
    dest = download(zenodo_url(name), RAW / name, ZENODO_FILES[name])
    if extract:
        target = RAW / name.removesuffix(".zip")
        print(f"extracting {dest} -> {target}")
        with zipfile.ZipFile(dest) as archive:
            archive.extractall(target)


def fetch_samples() -> None:
    for relative in SAMPLE_FILES:
        name = Path(relative).name
        download(f"{GITHUB_RAW}/{relative}", Path("data/sample") / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", action="store_true", help="(re)fetch the committed samples")
    parser.add_argument("--annotations", action="store_true", help="annotations.zip (0.6 MB)")
    parser.add_argument("--evaluation", action="store_true", help="evaluation.zip (3.9 GB)")
    parser.add_argument("--training", action="store_true", help="training.zip (4.5 GB)")
    parser.add_argument("--no-extract", action="store_true", help="keep zips unextracted")
    arguments = parser.parse_args()

    if not any(
        [arguments.samples, arguments.annotations, arguments.evaluation, arguments.training]
    ):
        parser.error("pick at least one of --samples/--annotations/--evaluation/--training")
    if arguments.samples:
        fetch_samples()
    for flag, name in [
        (arguments.annotations, "annotations.zip"),
        (arguments.evaluation, "evaluation.zip"),
        (arguments.training, "training.zip"),
    ]:
        if flag:
            fetch_zip(name, extract=not arguments.no_extract)


if __name__ == "__main__":
    main()
