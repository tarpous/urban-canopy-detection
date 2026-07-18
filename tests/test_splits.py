"""Site-split tests: extraction, determinism, and the leakage guarantee."""

import pytest

from urban_canopy.splits import Split, assert_site_disjoint, site_of, split_by_site


def names_for(site: str, count: int) -> list[str]:
    return [f"2019_{site}_{index}_315000_4094000_image_crop.tif" for index in range(count)]


class TestSiteExtraction:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("2018_TEAK_3_315000_4094000_image_crop.tif", "TEAK"),
            ("NIWO_043_2019.xml", "NIWO"),
            ("evaluation/MLBS_071_2020.tif", "MLBS"),
        ],
    )
    def test_known_neon_patterns(self, name: str, expected: str) -> None:
        assert site_of(name) == expected

    def test_missing_site_raises(self) -> None:
        with pytest.raises(ValueError, match="no NEON site code"):
            site_of("image_0001.tif")

    def test_longer_uppercase_runs_are_not_sites(self) -> None:
        assert site_of("ORTHO_TEAK_1.tif") == "TEAK"


class TestSplitBySite:
    def test_split_is_site_disjoint_and_deterministic(self) -> None:
        names = names_for("TEAK", 30) + names_for("NIWO", 30) + names_for("SJER", 40)
        first = split_by_site(names, val_fraction=0.3, seed=7)
        second = split_by_site(names, val_fraction=0.3, seed=7)
        assert first == second
        assert first.train_sites & first.val_sites == frozenset()
        assert len(first.train) + len(first.val) == len(names)
        assert len(first.val) >= 0.3 * len(names)

    def test_different_seeds_can_differ(self) -> None:
        names = names_for("TEAK", 10) + names_for("NIWO", 10) + names_for("SJER", 10)
        splits = {split_by_site(names, val_fraction=0.3, seed=seed).val_sites for seed in range(6)}
        assert len(splits) > 1

    def test_validation_never_swallows_all_sites(self) -> None:
        names = names_for("TEAK", 5) + names_for("NIWO", 5)
        split = split_by_site(names, val_fraction=0.9, seed=0)
        assert split.train
        assert split.val

    def test_single_site_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least two sites"):
            split_by_site(names_for("TEAK", 10))


class TestLeakageGuard:
    def test_overlapping_sites_are_caught(self) -> None:
        leaky = Split(
            train=tuple(names_for("TEAK", 3) + names_for("NIWO", 3)),
            val=tuple(names_for("TEAK", 2)),
        )
        with pytest.raises(ValueError, match=r"leakage.*TEAK"):
            assert_site_disjoint(leaky)
