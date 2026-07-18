"""Geographic train/val splitting by NEON site, with a leakage guarantee.

Overlapping aerial tiles from the same site are near-duplicates; a random
tile-level split leaks them across train and validation and inflates every
metric — the classic remote-sensing mistake. Splitting by *site* (the NEON
4-letter location code embedded in every filename) blocks that: all tiles of
a site land on the same side, and :func:`assert_site_disjoint` is enforced in
the test suite and re-checked inside the training notebooks.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import dataclass

#: NEON site codes are exactly four capital letters delimited by non-letters
#: (e.g. ``2018_TEAK_3_315000_4094000_image_crop.tif`` → ``TEAK``).
_SITE_PATTERN = re.compile(r"(?<![A-Za-z])([A-Z]{4})(?![A-Za-z])")


def site_of(name: str) -> str:
    """NEON site code embedded in a tile/annotation filename."""
    match = _SITE_PATTERN.search(name)
    if match is None:
        raise ValueError(f"no NEON site code found in {name!r}")
    return match.group(1)


@dataclass(frozen=True, slots=True)
class Split:
    train: tuple[str, ...]
    val: tuple[str, ...]

    @property
    def train_sites(self) -> frozenset[str]:
        return frozenset(site_of(name) for name in self.train)

    @property
    def val_sites(self) -> frozenset[str]:
        return frozenset(site_of(name) for name in self.val)


def split_by_site(names: list[str], *, val_fraction: float = 0.2, seed: int = 0) -> Split:
    """Deterministic site-level split targeting ``val_fraction`` of images.

    Sites are shuffled with the seed and assigned to validation until the
    image fraction is reached; every remaining site trains. With few sites the
    achieved fraction can overshoot — sites are atomic by design.
    """
    if not names:
        raise ValueError("nothing to split")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    by_site: dict[str, list[str]] = defaultdict(list)
    for name in names:
        by_site[site_of(name)].append(name)
    if len(by_site) < 2:
        raise ValueError("need at least two sites for a site-disjoint split")

    sites = sorted(by_site)
    random.Random(seed).shuffle(sites)

    target = val_fraction * len(names)
    val_names: list[str] = []
    val_sites = []
    for site in sites:
        if len(val_names) >= target:
            break
        val_sites.append(site)
        val_names.extend(by_site[site])
    if len(val_sites) == len(sites):  # never let validation swallow everything
        dropped = val_sites.pop()
        val_names = [name for name in val_names if site_of(name) != dropped]

    val_set = set(val_names)
    train_names = [name for name in names if name not in val_set]
    split = Split(train=tuple(train_names), val=tuple(val_names))
    assert_site_disjoint(split)
    return split


def assert_site_disjoint(split: Split) -> None:
    """Raise if any site appears on both sides — the leakage guarantee."""
    shared = split.train_sites & split.val_sites
    if shared:
        raise ValueError(f"site leakage between train and val: {sorted(shared)}")
