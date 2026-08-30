"""Implant-site geometry, against masks small enough to check by hand.

Several of these encode bugs that real scans exposed and a plausible-looking
number nearly hid. They are written so that reintroducing the bug fails loudly.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.implant_sites import (
    BRIDGE,
    CROWN,
    IAC,
    IMPLANT,
    LOWER_JAW,
    SINUS,
    UPPER_JAW,
    cylinder,
    feasibility,
    measure_site,
    orient_superior,
    run_through,
    site_is_occupied,
    superior_sign,
    tooth_centroids,
)

SPACING = (0.5, 0.5, 0.5)
SHAPE = (40, 40, 80)


def blank():
    return np.zeros(SHAPE, dtype=np.int16)


def mandible(crest_z=50, bottom_z=70, canal_z=60, half_width=8):
    """A slab of mandible with a canal in it, z increasing SUPERIORLY.

    Superior = smaller index would be the raw ToothFairy3 convention; these
    fixtures are built already oriented, which is what `measure_site` expects.
    """
    m = blank()
    m[20 - half_width:20 + half_width, 15:25, bottom_z:crest_z:-1] = LOWER_JAW
    m[20 - half_width:20 + half_width, 15:25, crest_z:bottom_z] = LOWER_JAW
    m[18:22, 18:22, canal_z - 2:canal_z + 2] = IAC[0]
    return m


class TestOrientation:
    def test_reads_the_anatomy_not_the_header(self):
        """The bug this guards: ToothFairy3's affine says z increases superiorly
        and the voxel data disagrees. Believing the header measured bone height
        upward from the chin, and the mandible numbers still looked plausible."""
        m = blank()
        m[10:30, 10:30, 5:15] = UPPER_JAW      # maxilla at LOW z
        m[10:30, 10:30, 50:70] = LOWER_JAW     # mandible at HIGH z
        assert superior_sign(m) == -1

        flipped, sign = orient_superior(m)
        assert sign == -1
        up = np.argwhere(flipped == UPPER_JAW)[:, 2].mean()
        lo = np.argwhere(flipped == LOWER_JAW)[:, 2].mean()
        assert up > lo, "after orienting, the maxilla must sit at higher z"

    def test_already_superior_is_left_alone(self):
        m = blank()
        m[10:30, 10:30, 50:70] = UPPER_JAW
        m[10:30, 10:30, 5:15] = LOWER_JAW
        flipped, sign = orient_superior(m)
        assert sign == 1
        assert flipped is m

    def test_falls_back_to_another_structure_pair(self):
        """28% of scans have no sinus; a scan missing one pair must still
        resolve from another rather than refusing to be measured."""
        m = blank()
        m[10:30, 10:30, 5:15] = SINUS[0]
        m[10:30, 10:30, 50:70] = IAC[0]
        assert superior_sign(m) == -1

    def test_raises_rather_than_guessing_when_nothing_is_paired(self):
        m = blank()
        m[10:30, 10:30, 5:15] = LOWER_JAW      # no upper structure at all
        with pytest.raises(ValueError, match="orientation"):
            superior_sign(m)


class TestRunThrough:
    def test_measures_the_run_containing_the_index(self):
        line = np.array([0, 1, 1, 1, 0, 1, 0], dtype=bool)
        assert run_through(line, 2) == (3, False)

    def test_falls_back_to_the_nearest_run_when_the_index_is_off_bone(self):
        """An arch position estimated from surrounding anatomy can sit a voxel or
        two off the bone without the site being unmeasurable."""
        line = np.array([0, 1, 1, 1, 0, 0, 0], dtype=bool)
        assert run_through(line, 5) == (3, True)

    def test_the_fallback_is_reported_not_taken_silently(self):
        """This fallback is how a bone-only slab came to measure one cortical
        plate and call a 6 mm ridge 1.8 mm. Whenever it fires, the site carries
        width_fallback=True so the failure is visible in the CSV."""
        assert run_through(np.array([0, 1, 1, 0], dtype=bool), 3)[1] is True
        assert run_through(np.array([0, 1, 1, 0], dtype=bool), 1)[1] is False

    def test_empty_line_is_zero_not_an_error(self):
        assert run_through(np.zeros(7, dtype=bool), 3) == (0, False)

    def test_does_not_bridge_a_gap(self):
        line = np.array([1, 1, 0, 1, 1, 1], dtype=bool)
        assert run_through(line, 0) == (2, False)


class TestCylinder:
    def test_radius_is_millimetres_not_voxels(self):
        """Same physical radius must select fewer voxels at coarser spacing, or
        every measurement silently changes meaning with scan resolution."""
        fine = cylinder((40, 40), (20, 20), 3.0, (0.25, 0.25, 0.25))
        coarse = cylinder((40, 40), (20, 20), 3.0, (1.0, 1.0, 1.0))
        assert fine.sum() > coarse.sum()

    def test_centre_is_always_inside(self):
        disc = cylinder((40, 40), (20, 20), 1.0, SPACING)
        assert disc[20, 20]


class TestMeasureSite:
    def test_mandible_height_runs_crest_to_canal_roof(self):
        m = mandible(crest_z=50, bottom_z=70, canal_z=40)
        # oriented fixture: crest at z=50 is BELOW canal at 40 in index terms,
        # so build it the way measure_site expects and assert the limiter.
        out = measure_site(m, (20, 20, 0), "lower", SPACING)
        assert out.limiting_structure in ("nerve", "bone_extent")
        assert np.isfinite(out.available_height_mm)

    def test_absent_canal_reports_bone_extent_not_zero(self):
        """No canal is not zero height -- anterior mandible has no canal above
        it and is the tallest bone in the jaw."""
        m = blank()
        m[12:28, 15:25, 40:70] = LOWER_JAW
        out = measure_site(m, (20, 20, 0), "lower", SPACING)
        assert out.limiting_structure == "bone_extent"
        assert out.available_height_mm > 0

    def test_no_bone_is_nan_not_zero(self):
        """A site with no bone is unmeasurable. Zero would read as 'measured,
        and the answer is no bone', which then looks like a real finding."""
        out = measure_site(blank(), (20, 20, 0), "lower", SPACING)
        assert out.limiting_structure == "no_bone"
        assert np.isnan(out.available_height_mm)
        assert out.n_bone_voxels == 0

    def test_a_canal_above_the_crest_is_nan_not_zero(self):
        """The sibling of `test_no_bone_is_nan_not_zero`, and it was not guarded.

        `crest - canal_top` going negative means the canal roof sits at or above
        the ridge crest, which cannot happen in a patient -- it says the crest or
        the canal was mis-located on this column. `max(height, 0.0)` turned that
        into a confident 0.0 mm: plausible-looking, extreme, and definitely
        infeasible. It fired on 38 sites across 32 patients in the ToothFairy3
        build, 10 of them carrying needs_implant=1 and so trained on as
        regression targets, against a measurable median of 14.4 mm.
        """
        m = blank()
        m[12:28, 15:25, 40:70] = LOWER_JAW
        m[18:22, 18:22, 72:76] = IAC[0]      # canal entirely above the bone
        out = measure_site(m, (20, 20, 0), "lower", SPACING)
        assert out.limiting_structure == "impossible_geometry"
        assert np.isnan(out.available_height_mm), (
            "an impossible geometry must not be reported as 0.0 mm of bone")

    def test_a_canal_straddling_the_crest_is_also_rejected(self):
        m = blank()
        m[12:28, 15:25, 40:70] = LOWER_JAW
        m[18:22, 18:22, 66:76] = IAC[0]
        out = measure_site(m, (20, 20, 0), "lower", SPACING)
        assert np.isnan(out.available_height_mm)

    def test_zero_is_reachable_but_only_when_it_is_real(self):
        """A crest exactly at the canal roof is a true 0.0 mm; below it is not.

        This is the line the clamp erased. `max(height, 0.0)` mapped both onto
        the same value, so the CSV could not distinguish "no bone above the
        nerve" from "the measurement failed". Only the second is now NaN.
        """
        def height_with_canal_at(cz):
            m = blank()
            m[12:28, 15:25, 40:70] = LOWER_JAW
            m[18:22, 18:22, cz:cz + 4] = IAC[0]
            return measure_site(m, (20, 20, 0), "lower", SPACING).available_height_mm

        assert height_with_canal_at(66) == pytest.approx(0.0), (
            "a canal roof exactly at the crest is genuinely zero bone")
        assert np.isnan(height_with_canal_at(70)), (
            "a canal roof above the crest is impossible, not zero")

    def test_a_width_that_could_not_be_probed_is_nan_not_zero(self):
        """Both zero-width paths, and they were fixed one at a time.

        `ridge_width` returns NaN when the probe plane holds no bone at all --
        and used to return 0.0 when the plane held bone that neither probe line
        crossed, which happens when the anatomy is offset from the centre in
        both x and y. A 0.0 there reads as a knife-edge ridge and fails the
        width rule as though it had been measured.
        """
        from src.data.implant_sites import ridge_width

        # Bone in the plane, but offset from the centre in both axes.
        m = blank()
        m[5:10, 5:10, 10:30] = LOWER_JAW
        width, _ = ridge_width(m, (35, 35, 0), crest_z=25, jaw="lower", spacing=SPACING)
        assert np.isnan(width) or width > 0.0, (
            f"got {width}; a width that could not be probed must not be 0.0 mm")

    def test_rejects_an_unknown_jaw(self):
        with pytest.raises(ValueError, match="jaw"):
            measure_site(blank(), (20, 20, 0), "middle", SPACING)

    def test_does_not_allocate_whole_volume_arrays(self):
        """The OOM this guards: measuring per site on the full array allocated
        several 68 MB boolean volumes and was killed by the OS."""
        big = np.zeros((512, 512, 262), dtype=np.int16)
        big[250:270, 250:270, 100:160] = LOWER_JAW
        before = big.nbytes
        out = measure_site(big, (260, 260, 0), "lower", (0.3, 0.3, 0.3))
        assert np.isfinite(out.crest_mm)
        assert big.nbytes == before  # no copy of the input was made


class TestOccupancy:
    def test_a_tooth_centred_here_occupies_the_site(self):
        m = blank()
        m[18:23, 18:23, 40:60] = 36
        cents = tooth_centroids(m)
        out = site_is_occupied(m, (20, 20, 50), SPACING, cents, tooth=36)
        assert out["occupied"] and out["by"] == "tooth" and out["tooth_id"] == 36

    def test_a_neighbouring_tooth_does_not(self):
        """In a single-tooth gap the neighbours sit ~3.5 mm away. Site 36 is
        empty here; only tooth 37 is annotated, and 37 is not 36."""
        m = blank()
        m[30:35, 18:23, 40:60] = 37        # a whole tooth-width away
        cents = tooth_centroids(m)
        out = site_is_occupied(m, (20, 20, 50), SPACING, cents, tooth=36)
        assert not out["occupied"], "a neighbour must not occupy this site"

    def test_the_opposing_tooth_in_the_OTHER_jaw_does_not_occupy_it(self):
        """The bug that made this a label lookup. Occupancy scored centroid
        distance as hypot(dx, dy) -- no z -- over centroids from both jaws, so
        upper 26, sitting directly above lower 36, landed ~0 mm away and claimed
        the site. 414 mandibular sites (6.9%) were marked occupied by a
        maxillary tooth while their own tooth was absent from the mask: real
        implant needs labelled as no need, in a positive class of 530."""
        m = blank()
        m[18:23, 18:23, 70:90] = 26        # upper counterpart, directly above
        cents = tooth_centroids(m)
        out = site_is_occupied(m, (20, 20, 50), SPACING, cents, tooth=36)
        assert not out["occupied"], "an opposing tooth in the other jaw occupied the site"

    def test_a_site_whose_own_tooth_is_annotated_elsewhere_is_still_occupied(self):
        """The label is the answer; the fitted arch position only reports how
        far off it was. A present tooth is present even if the polynomial put
        the site a few millimetres from its centroid."""
        m = blank()
        m[26:31, 18:23, 40:60] = 36        # ~2 mm from the fitted position
        cents = tooth_centroids(m)
        out = site_is_occupied(m, (20, 20, 50), SPACING, cents, tooth=36)
        assert out["occupied"] and out["tooth_id"] == 36
        assert out["occupancy_mm"] > 0

    def test_a_bridge_pontic_filling_the_gap_occupies_it(self):
        m = blank()
        m[16:25, 16:25, 40:60] = BRIDGE
        out = site_is_occupied(m, (20, 20, 50), SPACING, {})
        assert out["occupied"] and out["by"] == "bridge"

    def test_an_existing_implant_occupies_it(self):
        m = blank()
        m[17:24, 17:24, 40:60] = IMPLANT
        out = site_is_occupied(m, (20, 20, 50), SPACING, {})
        assert out["occupied"] and out["by"] == "implant"

    def test_implant_wins_over_crown_when_both_are_present(self):
        """A crown on an implant is one restored site, reported as the implant."""
        m = blank()
        m[17:24, 17:24, 40:55] = IMPLANT
        m[17:24, 17:24, 55:60] = CROWN
        out = site_is_occupied(m, (20, 20, 50), SPACING, {})
        assert out["by"] == "implant"

    def test_an_empty_ridge_is_empty(self):
        m = blank()
        m[12:28, 15:25, 40:70] = LOWER_JAW
        out = site_is_occupied(m, (20, 20, 50), SPACING, {})
        assert not out["occupied"]


class TestFeasibility:
    def make(self, jaw, height, width):
        from src.data.implant_sites import SiteMeasurement
        return SiteMeasurement(jaw, 0.0, height, width, "nerve", 100)

    def test_passes_when_both_clear_the_bar(self):
        out = feasibility(self.make("lower", 14.0, 7.0), 12.0, 10.0, 6.0)
        assert out["feasible"] and out["reason"] == "ok"

    def test_the_jaws_use_different_height_rules(self):
        """12 mm in the mandible (10 mm implant + 2 mm nerve margin) but 10 mm
        against the sinus. One threshold for both would mislabel every site."""
        m = self.make("upper", 11.0, 7.0)
        assert feasibility(m, 12.0, 10.0, 6.0)["feasible"]
        assert not feasibility(self.make("lower", 11.0, 7.0), 12.0, 10.0, 6.0)["feasible"]

    def test_narrow_ridge_fails_even_with_plenty_of_height(self):
        out = feasibility(self.make("lower", 30.0, 4.0), 12.0, 10.0, 6.0)
        assert not out["feasible"] and out["reason"] == "width"

    def test_reason_names_the_limiting_structure(self):
        out = feasibility(self.make("lower", 5.0, 8.0), 12.0, 10.0, 6.0)
        assert out["reason"] == "height_nerve"

    def test_unmeasurable_is_not_infeasible(self):
        """nan must not collapse into 'not feasible' -- that would turn missing
        data into a clinical finding."""
        out = feasibility(self.make("lower", float("nan"), 8.0), 12.0, 10.0, 6.0)
        assert out["reason"] == "unmeasurable"
        assert not out["feasible"] and not out["height_ok"]

    def test_thresholds_are_recorded_in_the_output(self):
        """Every row carries the rule it was judged against, so a later change
        of threshold is visible rather than silent."""
        out = feasibility(self.make("lower", 14.0, 7.0), 12.0, 10.0, 6.0)
        assert out["required_height_mm"] == 12.0
        assert out["required_width_mm"] == 6.0


class TestMandibleOnlyOrientation:
    """61 of 532 ToothFairy3 scans have no maxilla and no sinus at all.

    Every upper-versus-lower cue is unusable on them, and before the incisive
    canal was added they were dropped from the dataset outright.
    """

    def test_a_mandible_only_scan_still_resolves(self):
        from src.data.implant_sites import INCISIVE_CANAL

        m = blank()
        m[10:30, 10:30, 40:70] = LOWER_JAW
        m[18:22, 18:22, 60:66] = 36                 # a lower tooth, high
        m[18:22, 12:16, 44:48] = INCISIVE_CANAL[0]  # anterior canal, low
        assert superior_sign(m) == 1

    def test_the_same_scan_flipped_resolves_the_other_way(self):
        from src.data.implant_sites import INCISIVE_CANAL

        m = blank()
        m[10:30, 10:30, 10:40] = LOWER_JAW
        m[18:22, 18:22, 14:20] = 36
        m[18:22, 12:16, 32:36] = INCISIVE_CANAL[0]
        assert superior_sign(m) == -1

    def test_works_with_no_teeth_at_all(self):
        """ToothFairy3P_080 carries only jawbone, canals and pharynx."""
        from src.data.implant_sites import INCISIVE_CANAL

        m = blank()
        m[10:30, 10:30, 40:70] = LOWER_JAW
        m[18:22, 12:16, 44:48] = INCISIVE_CANAL[0]
        assert superior_sign(m) == 1

    def test_the_rejected_iac_cue_is_not_consulted(self):
        """lowerTeeth-over-IAC agreed only 82% of the time and LowerJaw-over-IAC
        46%, because the canal climbs toward the mandibular foramen. A scan
        offering ONLY those must refuse rather than answer at 46% accuracy."""
        m = blank()
        m[10:30, 10:30, 40:70] = LOWER_JAW
        m[18:22, 18:22, 44:48] = IAC[0]
        with pytest.raises(ValueError, match="orientation"):
            superior_sign(m)
