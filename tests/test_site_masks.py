"""Anatomy masks cut to the same patch as the model's input.

The alignment here is load-bearing. A mask offset by a few voxels scores the
explanation against anatomy the model was not shown, and every number downstream
still looks reasonable.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np

from src.data.implant_sites import IAC, LOWER_JAW, UPPER_JAW
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
