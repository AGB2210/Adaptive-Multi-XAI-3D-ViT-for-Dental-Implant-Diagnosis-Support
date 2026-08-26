"""Localisation metrics: do they say what they claim to say?"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.xai.localization import (
    competing_structure_ratio,
    enrichment,
    localization_scores,
    mass_inside,
    overlap_at_topk,
    pointing_game,
)

SHAPE = (16, 16, 16)


def blob(centre, radius=2, shape=SHAPE):
    grid = np.indices(shape).astype(np.float32)
    d = sum((grid[i] - centre[i]) ** 2 for i in range(3))
    return d <= radius**2


@pytest.fixture
def mask():
    return blob((4, 4, 4))


class TestPointingGame:
    def test_hit_when_the_peak_is_inside(self, mask):
        s = np.zeros(SHAPE, dtype=np.float32)
        s[4, 4, 4] = 1.0
        assert pointing_game(s, mask) is True

    def test_miss_when_the_peak_is_outside(self, mask):
        s = np.zeros(SHAPE, dtype=np.float32)
        s[12, 12, 12] = 1.0
        assert pointing_game(s, mask) is False

    def test_empty_mask_raises_rather_than_scoring_zero(self):
        with pytest.raises(ValueError):
            pointing_game(np.ones(SHAPE), np.zeros(SHAPE, dtype=bool))

    def test_accepts_a_torch_tensor(self, mask):
        s = torch.zeros(SHAPE)
        s[4, 4, 4] = 1.0
        assert pointing_game(s, torch.from_numpy(mask)) is True


class TestEnrichment:
    def test_uniform_saliency_scores_one(self, mask):
        """The definition of chance: a flat map is 1.0, whatever the mask size."""
        assert enrichment(np.ones(SHAPE, dtype=np.float32), mask) == pytest.approx(1.0, rel=1e-6)

    def test_perfectly_concentrated_saliency_scores_one_over_chance(self, mask):
        s = mask.astype(np.float32)
        assert enrichment(s, mask) == pytest.approx(1.0 / mask.mean(), rel=1e-6)

    def test_saliency_entirely_outside_scores_zero(self, mask):
        s = (~mask).astype(np.float32)
        assert enrichment(s, mask) == pytest.approx(0.0)

    def test_a_smaller_target_gives_a_larger_score_for_the_same_hit(self):
        """Chance correction is the point: hitting a rare target is worth more."""
        big, small = blob((8, 8, 8), radius=5), blob((8, 8, 8), radius=1)
        s = small.astype(np.float32)
        assert enrichment(s, small) > enrichment(s, big)

    def test_negative_saliency_is_clipped_not_summed(self, mask):
        """Attribution can be signed; mass is about magnitude of support."""
        s = np.full(SHAPE, -1.0, dtype=np.float32)
        s[mask] = 1.0
        assert mass_inside(s, mask) == pytest.approx(1.0)


class TestOverlap:
    def test_identical_regions_give_iou_one(self):
        m = blob((8, 8, 8), radius=3)
        k = m.mean()
        out = overlap_at_topk(m.astype(np.float32), m, k_fraction=k)
        assert out["iou"] == pytest.approx(1.0, abs=0.05)
        assert out["dice"] == pytest.approx(1.0, abs=0.05)

    def test_disjoint_regions_give_zero(self, mask):
        s = blob((12, 12, 12)).astype(np.float32)
        out = overlap_at_topk(s, mask, k_fraction=mask.mean())
        assert out["iou"] == pytest.approx(0.0)

    def test_topk_size_is_fixed_regardless_of_how_peaked_the_map_is(self, mask):
        """A fixed k is what keeps methods comparable."""
        peaked = np.zeros(SHAPE, dtype=np.float32)
        peaked[4, 4, 4] = 99.0
        diffuse = np.random.default_rng(0).random(SHAPE).astype(np.float32)
        a = overlap_at_topk(peaked, mask, 0.01)["topk_voxels"]
        b = overlap_at_topk(diffuse, mask, 0.01)["topk_voxels"]
        assert a == b


class TestCompetingStructure:
    def test_above_one_when_saliency_prefers_the_target(self):
        target, other = blob((4, 4, 4)), blob((12, 12, 12))
        s = target.astype(np.float32) + 0.01  # a little mass everywhere
        assert competing_structure_ratio(s, target, other) > 1.0

    def test_infinite_when_the_competitor_gets_no_mass_at_all(self):
        """The strongest possible result, and it must not read as "absent"."""
        target, other = blob((4, 4, 4)), blob((12, 12, 12))
        assert competing_structure_ratio(target.astype(np.float32), target, other) == float("inf")

    def test_nan_when_the_map_avoids_both_structures(self):
        target, other = blob((4, 4, 4)), blob((12, 12, 12))
        s = blob((8, 0, 0)).astype(np.float32)
        assert np.isnan(competing_structure_ratio(s, target, other))

    def test_below_one_when_saliency_prefers_the_neighbour(self):
        """The failure the project must be able to detect: an implant
        explanation that is really pointing at the crown."""
        target, other = blob((4, 4, 4)), blob((12, 12, 12))
        s = other.astype(np.float32)
        assert competing_structure_ratio(s, target, other) < 1.0

    def test_nan_when_the_competing_structure_is_absent(self, mask):
        empty = np.zeros(SHAPE, dtype=bool)
        assert np.isnan(competing_structure_ratio(mask.astype(np.float32), mask, empty))


def test_localization_scores_reports_every_metric(mask):
    s = mask.astype(np.float32)
    out = localization_scores(s, mask)
    for key in ("pointing_hit", "mass_inside", "enrichment", "mask_fraction",
                "iou", "dice", "topk_voxels", "mask_voxels"):
        assert key in out
    assert out["pointing_hit"] is True
    assert out["mask_voxels"] == int(mask.sum())
