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

from src.models.geometric import bone_threshold, measure_patch, otsu

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


def cortical_phantom(n=96, crest=70, canal_top=40, canal_thick=8, width_vox=30,
                     plate_vox=3, seed=1):
    """The phantom that matters: THREE tissue classes, not two.

    A real patch is mostly soft tissue, with a thin bright cortical shell around
    a ridge whose interior is trabecular bone barely above soft tissue, and a
    canal darker than either. One threshold cannot separate three classes, and
    which two it merges decides whether the estimator measures a ridge or a
    single cortical plate.
    """
    rng = np.random.default_rng(seed)
    patch = rng.normal(0.0, 0.15, size=(n, n, n)).astype(np.float32)   # soft tissue
    cx = n // 2
    half = width_vox // 2
    patch[cx - half:cx + half, :, :crest] = 0.45                       # trabecular
    patch[cx - half:cx - half + plate_vox, :, :crest] = 2.5            # buccal cortex
    patch[cx + half - plate_vox:cx + half, :, :crest] = 2.5            # lingual cortex
    patch[cx - half:cx + half, :, crest - 3:crest] = 2.5               # crestal cortex
    if canal_thick:
        patch[cx - half:cx + half, :, canal_top - canal_thick:canal_top] = -0.2
    patch += rng.normal(0.0, 0.03, size=patch.shape).astype(np.float32)
    height_mm = (crest - 1 - (canal_top - 1)) * SPACING[2]
    return patch.astype(np.float32), height_mm, width_vox * SPACING[0]


class TestCorticalRidges:
    """The failure that a single threshold produces on real bone.

    Otsu over a soft-tissue-dominated patch lands high -- measured at 1.52 in
    z-scored units on the phantom below -- and cuts between dense CORTICAL bone
    and everything else, so the trabecular bone filling the ridge falls on the
    background side. A run through the centre then reads cortex, marrow, cortex,
    and what gets measured is ONE CORTICAL PLATE: 0.9 mm where the ridge is
    9.0 mm.

    It is the same fault v2.0.0 fixed in the mask pipeline from the other side,
    where a bone-only slab had a hole at every occupied site and a 6.00 mm ridge
    measured 1.80 mm. Here it arrives through intensity rather than labels, and
    it was found by review rather than by this suite -- which is why the phantom
    now lives here.
    """

    def test_a_single_otsu_lands_above_trabecular_bone(self):
        """Both halves of the fix, pinned.

        The first pass must still land high -- that is the fault, and if it
        stops happening the regressions below stop testing anything. The
        corrected threshold must then land between soft tissue and trabecular
        bone instead.
        """
        patch, _, _ = cortical_phantom()
        naive = otsu(patch)
        assert naive > 1.0, (
            f"a single Otsu returned {naive:.2f}; the three-class phantom was "
            f"built so it splits off cortex alone"
        )
        assert float((patch > naive).mean()) < 0.10, "cortex is a small minority class"

        corrected = bone_threshold(patch, SPACING)
        assert 0.0 < corrected < 0.45, (
            f"the corrected threshold is {corrected:.2f}; it should sit between "
            f"soft tissue at 0.0 and trabecular bone at 0.45"
        )
        assert float((patch > corrected).mean()) > 0.05

    def test_the_ridge_is_measured_whole_not_one_cortical_plate(self):
        patch, _, width_mm = cortical_phantom()
        got = measure_patch(patch, "lower", SPACING)
        assert abs(got["ridge_width_mm"] - width_mm) <= 3 * SPACING[0], (
            f"measured {got['ridge_width_mm']:.2f} mm against a built-in "
            f"{width_mm:.2f} mm. A value near one plate thickness means the "
            f"threshold landed above trabecular bone again, so the probe is "
            f"reading cortex alone -- look at bone_threshold's second pass, not "
            f"at the probe"
        )

    def test_the_height_survives_a_cortical_ridge_too(self):
        patch, height_mm, _ = cortical_phantom()
        got = measure_patch(patch, "lower", SPACING)
        assert got["limiter"] == "nerve"
        assert abs(got["available_height_mm"] - height_mm) <= 3 * SPACING[2], (
            f"measured {got['available_height_mm']:.2f} mm against a built-in "
            f"{height_mm:.2f} mm"
        )

    def test_the_canal_is_still_found_once_trabecular_bone_counts_as_bone(self):
        """Lowering the threshold must not swallow the canal along with marrow.

        The correction includes trabecular bone, which sits only just above soft
        tissue -- and the canal is darker than either, so a threshold pushed too
        far down would take the canal with it and the estimator would report a
        solid column. NOTE: this is a THRESHOLD, not a morphological fill. The
        per-slice cavity fill was considered and rejected, because a 96-voxel
        patch is a short segment of the arch rather than a closed ring and
        whether the cortex encloses anything in-plane depends on where round the
        jaw the site sits.

        With no canal the estimator must fall back to bone extent; with one it
        must find it. The two cases therefore have to differ.
        """
        with_canal, _, _ = cortical_phantom()
        without, _, _ = cortical_phantom(canal_thick=0)
        a = measure_patch(with_canal, "lower", SPACING)
        b = measure_patch(without, "lower", SPACING)
        assert a["limiter"] == "nerve" and b["limiter"] == "bone_extent"
        assert a["available_height_mm"] < b["available_height_mm"]


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
