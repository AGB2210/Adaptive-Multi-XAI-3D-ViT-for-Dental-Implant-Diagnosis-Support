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
        """The function's own contract, exercised on a RAW signed map.

        Note this clip never fires in the pipeline: every method's output goes
        through `SaliencyMethod.attribute`, which is non-negative by the time it
        arrives -- Grad-CAM by ReLU, IG and GradientSHAP by `.abs()`, rollout by
        construction. Those are three different meanings of "mass", and the
        difference belongs beside any claim that one method localises better
        than another. See `mass_inside`'s docstring.
        """
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


class TestTheCeilingControlPremise:
    """What the not-yet-written achievable-ceiling control must be built on.

    The control is meant to put a denominator under every enrichment figure by
    running the methods on the synthetic planted-signal task, where the correct
    answer is known exactly. Two premises have to hold before its numbers
    transfer to the real localisation table, and only one of them is obvious.
    """

    def test_the_millimetre_heads_get_continuous_targets_not_zero_and_one(self):
        """A regression head trained on 0/1 would make the ceiling meaningless.

        `make_hybrid_dataset` builds each millimetre column as a linear function
        of one planted binary signal, scaled to a clinical range, precisely so
        the gate trains a regression head on lengths rather than on labels. This
        pins that, because it is the kind of thing that quietly reverts.
        """
        from tests.synthetic import make_hybrid_dataset

        _, y = make_hybrid_dataset(40, seed=0, shape=(32, 32, 32))
        assert set(np.unique(y[:, 0])) <= {0.0, 1.0}, "the binary column should be binary"
        for j in (1, 2):
            col = y[:, j]
            assert len(np.unique(np.round(col, 3))) > 30, (
                f"millimetre column {j} takes {len(np.unique(np.round(col, 3)))} "
                f"distinct values across 40 cases -- a regression head is being "
                f"trained on something close to labels, and an enrichment "
                f"ceiling measured on it would not transfer"
            )
            assert col.std() > 1.0 and col.min() > 0.0

    def test_the_planted_blob_is_a_far_easier_target_than_the_canal(self):
        """The premise that does NOT hold, quantified.

        Enrichment is bounded by resolution: a method that can only place mass
        at token granularity cannot concentrate it inside a structure thinner
        than a token. So a ceiling only transfers between two targets of similar
        SHAPE, not merely similar volume.

        A token here is `patch_size` 4 times the conv stem's stride 2, so 8
        input voxels = 2.4 mm at 0.3 mm spacing.

            planted blob   0.71% of the patch, compact sphere, ~2.9 tokens across
            nerve canal    0.48% of the patch, long tube,      ~0.94 tokens across

        Comparable fractions, and the reason they are not interchangeable is the
        aspect ratio. A ceiling measured on the blob would be optimistic and
        would make GradCAM and rollout look worse than the resolution allows.
        Whoever writes the control must plant a TUBE of the canal's cross
        section, not the existing sphere.
        """
        from tests.synthetic import signal_mask

        patch, spacing, token_voxels = 96, 0.3, 8
        blob = signal_mask(np.array([1, 0, 0]), shape=(patch, patch, patch))
        blob_fraction = float(blob.mean())
        blob_radius = (3 * blob.sum() / (4 * np.pi)) ** (1 / 3)

        canal_fraction = 0.00478           # measured, run_localization prints it
        cross_section = canal_fraction * patch ** 3 / patch
        canal_radius = (cross_section / np.pi) ** 0.5

        assert 0.5 < blob_fraction / canal_fraction < 2.0, (
            "volumes are comparable -- that is what makes the shape difference "
            "easy to miss"
        )
        blob_tokens = 2 * blob_radius / token_voxels
        canal_tokens = 2 * canal_radius / token_voxels
        assert blob_tokens > 2.0
        assert canal_tokens < 1.0, (
            f"the canal is {canal_tokens:.2f} tokens across against the blob's "
            f"{blob_tokens:.2f}. If this ever stops holding, re-derive the "
            f"ceiling argument rather than assuming it still applies"
        )
        assert spacing * token_voxels == pytest.approx(2.4)
