"""Anatomy masks cut to the same patch as the model's input.

The alignment here is load-bearing. A mask offset by a few voxels scores the
explanation against anatomy the model was not shown, and every number downstream
still looks reasonable.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np

from src.data.implant_sites import IAC, LOWER_JAW, UPPER_JAW
from src.data.site_dataset import cut_patch, patch_centre
from src.xai.site_masks import SITE_STRUCTURES, describe_coverage, patch_masks


def write_mask(tmp_path, flipped: bool):
    """A toy jaw. `flipped` builds it in ToothFairy3's own convention, where the
    z index increases DOWNWARD and the maxilla therefore sits at low z."""
    m = np.zeros((60, 60, 60), dtype=np.int16)
    if flipped:
        m[10:50, 10:50, 5:15] = UPPER_JAW      # maxilla high in the head, low z
        m[10:50, 10:50, 40:55] = LOWER_JAW
        m[28:32, 28:32, 44:48] = IAC[0]
    else:
        m[10:50, 10:50, 45:55] = UPPER_JAW
        m[10:50, 10:50, 5:20] = LOWER_JAW
        m[28:32, 28:32, 12:16] = IAC[0]
    path = tmp_path / ("flip.nii.gz" if flipped else "plain.nii.gz")
    nib.save(nib.Nifti1Image(m, np.eye(4)), str(path))
    return path


SITE = {"site_x": 30.0, "site_y": 30.0}


class TestPatchMasks:
    def test_returns_every_named_structure(self, tmp_path):
        path = write_mask(tmp_path, flipped=False)
        masks = patch_masks(path, {**SITE, "site_z": 14.0}, patch_size=16)
        assert set(masks) == set(SITE_STRUCTURES)

    def test_masks_are_the_patch_size(self, tmp_path):
        path = write_mask(tmp_path, flipped=False)
        masks = patch_masks(path, {**SITE, "site_z": 14.0}, patch_size=16)
        for m in masks.values():
            assert m.shape == (16, 16, 16)
            assert m.dtype == bool

    def test_the_nerve_is_found_inside_the_patch(self, tmp_path):
        path = write_mask(tmp_path, flipped=False)
        masks = patch_masks(path, {**SITE, "site_z": 14.0}, patch_size=16)
        assert masks["nerve"].any()
        assert masks["jawbone"].any()

    def test_the_z_flip_is_applied_like_the_input(self, tmp_path):
        """The site coordinates were measured on the oriented mask. If this did
        not flip identically, the patch would be cut from the opposite end of
        the head."""
        path = write_mask(tmp_path, flipped=True)
        # After orienting, the canal that sat at raw z 44-48 lands at 60-1-47..
        masks = patch_masks(path, {**SITE, "site_z": 60 - 1 - 46.0}, patch_size=16)
        assert masks["nerve"].any(), "the flip was not applied to the mask"

    def test_a_patch_away_from_the_canal_has_no_nerve(self, tmp_path):
        """Anterior mandibular sites genuinely have no canal above them, and
        that must read as absent rather than as a miss."""
        path = write_mask(tmp_path, flipped=False)
        masks = patch_masks(path, {"site_x": 15.0, "site_y": 15.0, "site_z": 14.0},
                            patch_size=8)
        assert not masks["nerve"].any()

    def test_a_missing_z_falls_back_to_the_middle_rather_than_crashing(self, tmp_path):
        path = write_mask(tmp_path, flipped=False)
        masks = patch_masks(path, {**SITE, "site_z": float("nan")}, patch_size=16)
        assert masks["jawbone"].shape == (16, 16, 16)


class TestCoverage:
    def test_reports_the_fraction_each_structure_occupies(self, tmp_path):
        path = write_mask(tmp_path, flipped=False)
        masks = patch_masks(path, {**SITE, "site_z": 14.0}, patch_size=16)
        cov = describe_coverage(masks)
        assert 0.0 < cov["nerve"] < cov["jawbone"] <= 1.0

    def test_the_nerve_is_the_rarer_target(self, tmp_path):
        """Enrichment is a ratio against exactly this. Comparing enrichment on a
        structure filling 30% of the patch to one filling 0.5% without the
        denominator is meaningless."""
        path = write_mask(tmp_path, flipped=False)
        cov = describe_coverage(patch_masks(path, {**SITE, "site_z": 14.0}, patch_size=16))
        assert cov["nerve"] < 0.2


# --- The mask box must BE the model's box, not a second guess at it -----------
#
# `patch_centre` pushes the input box a quarter of a patch toward the structure
# that decides the answer -- 24 voxels at patch_size 96, which is 7.2 mm at
# 0.3 mm. `patch_masks` used to build its own centre from raw `site_z`, so it
# honoured the z flip but not the shift, and every localisation number was
# scored against anatomy 7.2 mm from where the model was looking. Every test
# above this line passed throughout, because they call `patch_masks` with a
# hand-made row and never compare it against the model's own path.


class TestTheMaskBoxIsTheModelsBox:
    def test_the_crop_matches_patch_centre_voxel_for_voxel(self, tmp_path):
        path = write_mask(tmp_path, flipped=False)
        # site_z must sit where the structure actually is. At z=30 the nerve
        # falls outside BOTH the shifted and unshifted boxes, so the comparison
        # would be all-False against all-False and would pass either way.
        row = {**SITE, "site_z": 14.0, "jaw": "lower"}
        patch_size = 16

        got = patch_masks(path, row, patch_size, {"nerve": IAC})["nerve"]
        assert got.any(), "phantom mis-sited: the test proves nothing on an empty mask"

        volume = np.asarray(nib.load(str(path)).dataobj)
        centre = patch_centre(row, volume.shape, patch_size)
        want = cut_patch(np.isin(volume, IAC).astype(np.uint8),
                         centre, patch_size).astype(bool)

        assert np.array_equal(got, want), (
            "the mask crop drifted from patch_centre -- a second implementation "
            "of the crop is exactly what went wrong")

    def test_the_jaw_shift_reaches_the_mask(self, tmp_path):
        """Guard the guard: if the shift were ignored these would be identical."""
        path = write_mask(tmp_path, flipped=False)
        lower = {**SITE, "site_z": 30.0, "jaw": "lower"}
        upper = {**SITE, "site_z": 30.0, "jaw": "upper"}

        got_lower = patch_masks(path, lower, 16, {"bone": (LOWER_JAW, UPPER_JAW)})["bone"]
        got_upper = patch_masks(path, upper, 16, {"bone": (LOWER_JAW, UPPER_JAW)})["bone"]

        assert not np.array_equal(got_lower, got_upper), (
            "jaw is not reaching patch_centre -- the quarter-shift is ignored")
