"""The hand-crafted geometric estimator, against geometry with a known answer.

A baseline nobody has verified is worse than no baseline: if it under-performs,
the paper cannot tell "a ruler cannot do this" from "the ruler is bent". So
every test here builds a phantom whose true crest-to-canal distance and ridge
width are known by construction, and checks the estimator recovers them.

Tolerances are in voxels and stated as such. At 0.3 mm spacing one voxel is
0.3 mm, and the estimator locates a boundary to the nearest slice, so anything
tighter than a voxel would be testing rounding rather than method.
"""

from __future__ import annotations

import numpy as np

from src.models.geometric import measure_patch, otsu

SPACING = (0.3, 0.3, 0.3)


def phantom(n=96, crest=70, canal_top=40, canal_thick=8, width_vox=30, seed=0):
    """A bright bone block with a dark canal inside it, and air above the crest.

    Mandibular convention: bone below, crest at the top, canal below the crest.
    Returns (patch, true_height_mm, true_width_mm).
    """
    rng = np.random.default_rng(seed)
    patch = rng.normal(-1.0, 0.05, size=(n, n, n)).astype(np.float32)   # air
    cx = n // 2
    half = width_vox // 2
    patch[cx - half:cx + half, :, :crest] = 1.0                          # bone slab
    patch[cx - half:cx + half, :, canal_top - canal_thick:canal_top] = -1.0   # canal
    patch += rng.normal(0.0, 0.02, size=patch.shape).astype(np.float32)
    # crest is the last bone slice (crest-1); the canal roof is canal_top-1
    height_mm = (crest - 1 - (canal_top - 1)) * SPACING[2]
    return patch.astype(np.float32), height_mm, width_vox * SPACING[0]


class TestOtsu:
    def test_it_splits_a_two_peak_histogram_between_the_peaks(self):
        rng = np.random.default_rng(0)
        v = np.concatenate([rng.normal(-1, 0.1, 2000), rng.normal(1, 0.1, 2000)])
        assert -0.5 < otsu(v) < 0.5

    def test_a_constant_volume_has_no_threshold(self):
        assert np.isnan(otsu(np.ones(500, dtype=np.float32)))


class TestMeasurePatch:
    def test_it_recovers_a_known_crest_to_canal_distance(self):
        patch, height_mm, _ = phantom()
        got = measure_patch(patch, "lower", SPACING)
        assert got["limiter"] == "nerve"
        assert abs(got["available_height_mm"] - height_mm) <= 2 * SPACING[2], (
            f"measured {got['available_height_mm']:.2f} mm against a built-in "
            f"{height_mm:.2f} mm"
        )

    def test_it_recovers_a_known_ridge_width(self):
        patch, _, width_mm = phantom(width_vox=24)
        got = measure_patch(patch, "lower", SPACING)
        assert abs(got["ridge_width_mm"] - width_mm) <= 2 * SPACING[0], (
            f"measured {got['ridge_width_mm']:.2f} mm against a built-in {width_mm:.2f} mm"
        )

    def test_the_measurement_tracks_the_canal_when_it_moves(self):
        """The estimator must respond to the geometry, not to the patch size."""
        seen = []
        for canal_top in (30, 40, 50):
            patch, height_mm, _ = phantom(canal_top=canal_top)
            got = measure_patch(patch, "lower", SPACING)
            seen.append((height_mm, got["available_height_mm"]))
        truth = [a for a, _ in seen]
        pred = [b for _, b in seen]
        assert truth == sorted(truth, reverse=True)
        assert pred == sorted(pred, reverse=True), (
            f"height must fall as the canal rises toward the crest; got {pred}"
        )

    def test_no_canal_falls_back_to_bone_extent_rather_than_inventing_one(self):
        patch, _, _ = phantom(canal_thick=0)
        got = measure_patch(patch, "lower", SPACING)
        assert got["limiter"] == "bone_extent"
        assert np.isfinite(got["available_height_mm"])

    def test_a_patch_with_no_bone_is_unmeasured_not_guessed(self):
        rng = np.random.default_rng(1)
        patch = rng.normal(-1.0, 0.05, size=(96, 96, 96)).astype(np.float32)
        got = measure_patch(patch, "lower", SPACING)
        assert np.isnan(got["available_height_mm"]), (
            "a baseline that always answers is worse than one that says when it could not"
        )

    def test_the_upper_jaw_measures_in_the_other_direction(self):
        """The sinus sits ABOVE the maxillary crest, the canal BELOW the
        mandibular one, so the crest is the opposite end of the bone column."""
        patch, _, _ = phantom()
        lower = measure_patch(patch, "lower", SPACING)
        upper = measure_patch(patch, "upper", SPACING)
        assert lower["crest_z"] > upper["crest_z"]

    def test_it_is_deterministic(self):
        patch, _, _ = phantom(seed=3)
        a = measure_patch(patch, "lower", SPACING)
        b = measure_patch(patch, "lower", SPACING)
        assert a == b
