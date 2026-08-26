"""Locating a tooth site when the tooth is gone.

Validated against real anatomy by leave-one-out on 10 ToothFairy3 scans: hide a
tooth that IS present, fit the arch to the rest, and measure how far off the
prediction lands. Median 1.9 mm, 88.8% within 5 mm, against teeth 7-10 mm wide.
These tests hold the behaviour that produced that.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.dental_arch import (
    ARCH,
    LOWER_ARCH,
    UPPER_ARCH,
    arch_index,
    fit_arch,
    site_positions,
    within_volume,
)
from src.data.implant_sites import LOWER_JAW, UPPER_JAW


def parabolic_jaw(teeth, jaw, x0=60.0, y0=40.0, span=40.0, depth=25.0):
    """Centroids on a parabola, the shape a real dental arch approximates."""
    order = ARCH[jaw]
    out = {}
    for t in teeth:
        i = order.index(t)
        u = (i - (len(order) - 1) / 2) / ((len(order) - 1) / 2)   # -1..1
        out[t] = np.array([x0 + span * u, y0 + depth * u * u, 50.0])
    return out


class TestArchIndex:
    def test_runs_right_to_left_across_the_midline(self):
        """FDI is not numeric order around the arch: 17..11 then 21..27."""
        assert arch_index(17) == 0
        assert arch_index(11) == 6
        assert arch_index(21) == 7
        assert arch_index(27) == 13

    def test_each_jaw_is_indexed_independently(self):
        assert arch_index(47) == arch_index(17) == 0

    def test_a_third_molar_is_not_a_scored_site(self):
        """18/28/38/48 are excluded by decision; asking for one is a bug."""
        with pytest.raises(ValueError, match="28 sites"):
            arch_index(38)


class TestFitArch:
    def test_recovers_a_held_out_tooth(self):
        """The leave-one-out check that was run on real scans, in miniature."""
        full = parabolic_jaw(UPPER_ARCH, "upper")
        held = 24
        curve, n = fit_arch({t: c for t, c in full.items() if t != held}, "upper")
        px, py = curve(arch_index(held))
        assert abs(px - full[held][0]) < 1.0
        assert abs(py - full[held][1]) < 1.0
        assert n == 13

    def test_extrapolates_past_the_end_of_the_anchors(self):
        """45% of missing sites have a neighbour on one side only, so the fit
        must reach beyond the teeth that defined it."""
        anchors = parabolic_jaw((17, 16, 15, 14, 13), "upper")
        curve, _ = fit_arch(anchors, "upper")
        px, _ = curve(arch_index(27))
        assert np.isfinite(px)

    def test_only_uses_teeth_from_the_requested_jaw(self):
        mixed = {**parabolic_jaw(UPPER_ARCH, "upper"), **parabolic_jaw(LOWER_ARCH, "lower")}
        _, n = fit_arch(mixed, "upper")
        assert n == len(UPPER_ARCH)

    def test_no_teeth_in_this_jaw_returns_none(self):
        assert fit_arch(parabolic_jaw(LOWER_ARCH, "lower"), "upper") is None

    def test_degree_drops_rather_than_overfitting(self):
        """Two anchors cannot support a quadratic; asking for one must not blow
        up or invent curvature that no data implies."""
        curve, n = fit_arch(parabolic_jaw((13, 23), "upper"), "upper", degree=2)
        assert n == 2
        assert np.isfinite(curve(0)[0])

    def test_a_single_anchor_is_a_constant_not_an_error(self):
        curve, n = fit_arch(parabolic_jaw((13,), "upper"), "upper")
        assert n == 1
        assert curve(0) == curve(13)


class TestSitePositions:
    def make(self, upper_teeth=(), lower_teeth=()):
        mask = np.zeros((120, 120, 80), dtype=np.int16)
        # so orientation resolves: maxilla above mandible
        mask[10:110, 10:110, 60:70] = UPPER_JAW
        mask[10:110, 10:110, 10:20] = LOWER_JAW
        for jaw, teeth in (("upper", upper_teeth), ("lower", lower_teeth)):
            for t, c in parabolic_jaw(teeth, jaw).items():
                x, y = int(c[0]), int(c[1])
                z = 62 if jaw == "upper" else 12
                mask[x - 2:x + 3, y - 2:y + 3, z:z + 4] = t
        return mask

    def test_every_scored_site_gets_an_entry(self):
        out = site_positions(self.make(UPPER_ARCH, LOWER_ARCH))
        assert set(out) == set(UPPER_ARCH) | set(LOWER_ARCH)
        assert len(out) == 28

    def test_plenty_of_teeth_is_the_teeth_method(self):
        out = site_positions(self.make(UPPER_ARCH, LOWER_ARCH))
        assert all(v["method"] == "teeth" for v in out.values())

    def test_one_or_two_teeth_is_flagged_sparse_not_silently_trusted(self):
        """A position extrapolated from two anchors is far weaker than one from
        twelve, and the row has to say so or nobody can filter on it."""
        out = site_positions(self.make((13, 23), LOWER_ARCH))
        assert out[17]["method"] == "sparse"
        assert out[47]["method"] == "teeth"

    def test_an_edentulous_jaw_borrows_the_other_one(self):
        """30.7% of missing sites are in a fully edentulous jaw -- the patients
        who most need implants. Dropping them would bias the dataset against
        its own clinical purpose."""
        out = site_positions(self.make((), LOWER_ARCH))
        assert all(out[t]["method"] == "opposite_jaw" for t in UPPER_ARCH)
        assert all(np.isfinite(out[t]["xy"]).all() for t in UPPER_ARCH)

    def test_no_teeth_anywhere_fails_loudly_per_site(self):
        out = site_positions(self.make((), ()))
        assert all(v["method"] == "failed" and v["xy"] is None for v in out.values())

    def test_anchor_count_is_recorded(self):
        out = site_positions(self.make(UPPER_ARCH, LOWER_ARCH))
        assert out[16]["anchors"] == len(UPPER_ARCH)


class TestWithinVolume:
    def test_a_position_inside_the_scan_passes(self):
        assert within_volume((50.0, 50.0), (120, 120, 80))

    def test_an_extrapolated_position_outside_the_scan_is_rejected(self):
        """Extrapolating an arch past its anchors can land outside the field of
        view. That is a site we never saw, not a site with no bone -- measuring
        it would score whatever happens to sit at the clipped edge."""
        assert not within_volume((-5.0, 50.0), (120, 120, 80))
        assert not within_volume((50.0, 500.0), (120, 120, 80))

    def test_none_is_rejected(self):
        assert not within_volume(None, (120, 120, 80))

    def test_the_margin_excludes_the_very_edge(self):
        assert not within_volume((0.0, 50.0), (120, 120, 80), margin=2)
