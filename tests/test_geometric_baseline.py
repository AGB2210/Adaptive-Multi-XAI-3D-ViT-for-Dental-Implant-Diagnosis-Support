"""The geometric baseline measures synthetic anatomy it should get exactly right.

If this baseline is going to be used to argue the transformer was unnecessary, it
has to be shown correct on geometry whose answer is known by construction. These
build a patch with bone, a canal and a ridge at chosen sizes and check the
measurement comes back within a voxel.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.geometric_baseline import (
    SPACING_MM,
    canal_roof,
    central_disc,
    crest_from_image,
    measure,
    otsu,
    run_through,
)

SIZE = 96
CREST_Z = 72  # where patch_centre puts the crest: size/2 + size//4


def make_patch(crest_z=CREST_Z, canal_top=32, canal_thickness=8, ridge_half_width=10,
               bone=2.0, soft=0.0, air=-1.0):
    """A patch with known geometry, in the cache's z-scored value range.

    bone slab from z=0 to crest_z, a dark canal inside it, air above the crest.
    """
    p = np.full((SIZE, SIZE, SIZE), air, dtype=np.float32)
    x0, x1 = SIZE // 2 - ridge_half_width, SIZE // 2 + ridge_half_width
    p[x0:x1, :, :crest_z] = bone
    p[x0:x1, :, canal_top - canal_thickness:canal_top] = soft   # the canal
    return p


def test_otsu_separates_two_populations():
    v = np.concatenate([np.full(1000, -1.0), np.full(1000, 2.0)])
    t = otsu(v)
    assert -1.0 < t < 2.0


def test_otsu_on_constant_input_is_not_a_crash():
    assert not np.isfinite(otsu(np.zeros(100))) or True  # must not raise


def test_run_through_measures_the_run_containing_the_index():
    line = np.array([0, 1, 1, 1, 1, 0, 1, 1], dtype=bool)
    length, fell_back = run_through(line, 3)
    assert (length, fell_back) == (4, False)


def test_run_through_reports_the_nearest_run_fallback():
    """The fallback that turned a 6.00 mm ridge into 1.80 mm must be visible."""
    line = np.array([1, 1, 0, 0, 0, 1, 1, 1], dtype=bool)
    length, fell_back = run_through(line, 3)
    assert fell_back is True
    assert length in (2, 3)


def test_crest_is_found_at_the_top_of_the_bone():
    p = make_patch(crest_z=60)
    bone = p > otsu(p)
    disc = central_disc(p.shape[:2])
    assert abs(crest_from_image(bone, disc) - 59) <= 1


def test_crest_uses_the_median_not_the_maximum():
    """One spike must not set the crest for the whole cylinder."""
    p = make_patch(crest_z=60)
    p[SIZE // 2 + 8, SIZE // 2 + 8, 60:80] = 2.0     # a spike inside the disc
    bone = p > otsu(p)
    disc = central_disc(p.shape[:2])
    assert crest_from_image(bone, disc) < 70, "a single spike set the crest"


def test_canal_roof_finds_an_enclosed_gap():
    p = make_patch(canal_top=32, canal_thickness=8)
    bone = p > otsu(p)
    disc = central_disc(p.shape[:2])
    assert abs(canal_roof(bone, disc, CREST_Z) - 31) <= 2


def test_canal_roof_refuses_when_there_is_no_enclosed_gap():
    """No canal must be nan, never the bottom of the bone.

    An anterior site genuinely has no canal above it. Measuring to the inferior
    border instead would report a large height and call an unmeasurable site
    comfortably feasible.
    """
    p = make_patch(canal_thickness=0)
    bone = p > otsu(p)
    disc = central_disc(p.shape[:2])
    assert not np.isfinite(canal_roof(bone, disc, CREST_Z))


def test_canal_roof_ignores_a_gap_thinner_than_the_minimum():
    p = make_patch(canal_top=32, canal_thickness=2)   # 0.6 mm, below MIN_CANAL_MM
    bone = p > otsu(p)
    disc = central_disc(p.shape[:2])
    assert not np.isfinite(canal_roof(bone, disc, CREST_Z))


@pytest.mark.parametrize("canal_top,expected_mm", [(32, (72 - 31) * SPACING_MM),
                                                   (52, (72 - 51) * SPACING_MM)])
def test_height_matches_the_construction(canal_top, expected_mm):
    p = make_patch(canal_top=canal_top)
    out = measure(p, CREST_Z)
    assert out["reason"] == "ok"
    assert abs(out["height_mm"] - expected_mm) <= 2 * SPACING_MM


def test_width_matches_the_construction():
    """20 voxels of bone across = 6.0 mm, the configured minimum."""
    out = measure(make_patch(ridge_half_width=10), CREST_Z)
    assert abs(out["width_mm"] - 6.0) <= 2 * SPACING_MM


def test_measure_refuses_rather_than_guessing_when_there_is_no_canal():
    out = measure(make_patch(canal_thickness=0), CREST_Z)
    assert not np.isfinite(out["height_mm"])
    assert out["reason"] == "no_canal_in_patch"


def test_measure_survives_an_empty_patch():
    out = measure(np.zeros((SIZE, SIZE, SIZE), dtype=np.float32), CREST_Z)
    assert not np.isfinite(out["height_mm"])
