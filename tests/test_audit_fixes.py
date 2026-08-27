"""Regressions for an external audit of the label and XAI pipeline.

Every test here corresponds to a fault that was live in a tagged release, and
every one of them was invisible to the suite that existed at the time. They are
grouped by what made them invisible, because that is the reusable lesson: each
component was correct in isolation and wrong in composition, or correct in code
and contradicted by the config that drove it.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.implant_sites import (
    LOWER_JAW,
    ridge_width,
    run_through,
    site_is_occupied,
    tooth_centroids,
)
from src.data.site_dataset import patch_centre
from src.utils.config import load_config
from src.xai.runner import patients_of, select_cases, xai_setting

SPACING = (0.3, 0.3, 0.3)


class TestOccupancyIsALabelLookup:
    """Occupancy scored `hypot(dx, dy)` over centroids from BOTH jaws. z was
    absent, so an upper molar sitting directly above its lower counterpart
    landed ~0 mm away and claimed the site."""

    def jaw_with(self, *teeth):
        m = np.zeros((80, 80, 120), dtype=np.int16)
        m[30:50, 30:50, 20:60] = LOWER_JAW
        for label, z0 in teeth:
            m[38:44, 38:44, z0:z0 + 18] = label
        return m

    def test_the_opposing_upper_tooth_does_not_occupy_a_lower_site(self):
        """414 mandibular sites (6.9%) were marked occupied by a maxillary
        tooth. Spot-checked against the raw masks, the site's own lower tooth
        was absent in four of four: real implant needs recorded as no need."""
        m = self.jaw_with((26, 80))              # upper 26, directly above
        out = site_is_occupied(m, (41, 41, 0), SPACING, tooth_centroids(m), tooth=36)
        assert not out["occupied"]

    def test_its_own_tooth_does_occupy_it(self):
        m = self.jaw_with((36, 30))
        out = site_is_occupied(m, (41, 41, 0), SPACING, tooth_centroids(m), tooth=36)
        assert out["occupied"] and out["tooth_id"] == 36

    def test_a_neighbour_does_not_occupy_it(self):
        """Lower incisors are ~5.3 mm wide against a 4.0 mm reach, so the
        distance test could not separate a neighbour from the site either."""
        m = self.jaw_with((37, 30))
        out = site_is_occupied(m, (41, 41, 0), SPACING, tooth_centroids(m), tooth=36)
        assert not out["occupied"]

    def test_the_answer_does_not_depend_on_the_fitted_position(self):
        """The arch polynomial carries a residual. The label does not."""
        m = self.jaw_with((36, 30))
        cents = tooth_centroids(m)
        near = site_is_occupied(m, (41, 41, 0), SPACING, cents, tooth=36)
        far = site_is_occupied(m, (48, 48, 0), SPACING, cents, tooth=36)
        assert near["occupied"] and far["occupied"]
        assert far["occupancy_mm"] > near["occupancy_mm"]      # reported, not used


class TestRidgeWidthMeasuresTheRidge:
    """The width slab was bone only. A root is a separate label, so an occupied
    site was probed through a hole, `run_through` took the nearest run, and one
    cortical plate got measured in isolation."""

    def ridge(self, tooth: bool):
        m = np.zeros((120, 120, 120), dtype=np.int16)
        m[50:70, 40:100, 20:80] = LOWER_JAW          # 20 voxels = 6.0 mm across
        if tooth:
            m[56:64, 66:74, 60:80] = 36
        return m

    def test_a_tooth_in_the_socket_does_not_shrink_the_ridge(self):
        """6.00 mm of bone measured 1.80 mm with a root in it. It fired on every
        occupied site -- 92% of the cohort -- and left the CSV with a median
        width of 3.00 mm where a tooth was present against 11.10 mm where one
        was missing: backwards, since an edentulous site has resorbed and a
        dentate one has not."""
        empty, _ = ridge_width(self.ridge(False), (60, 70, 0), 80, "lower", SPACING)
        held, _ = ridge_width(self.ridge(True), (60, 70, 0), 80, "lower", SPACING)
        assert empty == pytest.approx(6.0, abs=0.31)
        assert held == pytest.approx(empty, abs=0.31), "a root shrank the ridge"

    def test_the_fallback_is_reported(self):
        """This fallback is the mechanism. Whenever it fires the site carries
        width_fallback=True, so the failure is visible in the CSV."""
        assert run_through(np.array([0, 1, 1, 0], dtype=bool), 3)[1] is True
        assert run_through(np.array([0, 1, 1, 0], dtype=bool), 1)[1] is False


class TestPatchFraming:
    """A centred 96^3 box at 0.3 mm reached 14.4 mm below the crest against a
    12.0 mm threshold, leaving 26.2% of sites needing an implant with their
    limiting structure outside the input."""

    ROW = {"site_x": 50.0, "site_y": 50.0, "site_z": 60.0, "jaw": "lower"}

    def test_the_box_is_pushed_below_the_crest_in_the_mandible(self):
        _, _, z = patch_centre(self.ROW, (200, 200, 200), 96)
        assert z == 60.0 - 24, "the mandibular box was not shifted toward the canal"

    def test_and_above_it_in_the_maxilla(self):
        _, _, z = patch_centre({**self.ROW, "jaw": "upper"}, (200, 200, 200), 96)
        assert z == 60.0 + 24, "the maxillary box must reach toward the sinus"

    def test_reach_below_the_crest_clears_the_threshold_with_room(self):
        cfg = load_config("configs/sites.yaml")
        size, mm = cfg.model.img_size, 0.3
        below = (size // 2 + size // 4) * mm
        threshold = cfg.sites.min_height_mandible_mm
        assert below > threshold + 5.0, (
            f"reach {below:.1f} mm leaves too little margin over the {threshold} mm rule")

    def test_x_and_y_are_untouched(self):
        x, y, _ = patch_centre(self.ROW, (200, 200, 200), 96)
        assert (x, y) == (50.0, 50.0)

    def test_a_missing_z_still_falls_back_to_the_middle(self):
        _, _, z = patch_centre({**self.ROW, "site_z": float("nan")}, (200, 200, 200), 96)
        assert z == 100.0


class TestXaiSettingsAreRead:
    """`configs/sites.yaml` carried an `xai:` block for a whole release and no
    script read it. Every run used its own argparse default instead."""

    def test_the_config_wins_over_the_fallback(self):
        cfg = load_config("configs/sites.yaml")
        assert xai_setting(cfg, "shap_samples", None, 24) == cfg.xai.shap_samples
        assert xai_setting(cfg, "shap_samples", None, 24) != 24

    def test_an_explicit_flag_wins_over_the_config(self):
        cfg = load_config("configs/sites.yaml")
        assert xai_setting(cfg, "shap_samples", 8, 24) == 8

    def test_the_fallback_applies_only_when_neither_exists(self):
        cfg = load_config("configs/sites.yaml")
        assert xai_setting(cfg, "not_a_real_key", None, 7) == 7

    @pytest.mark.parametrize("key,floor", [
        ("shap_samples", 128), ("faithfulness_cases", 100),
        ("localization_cases", 100), ("adaptive_cases", 100),
        ("randomization_cases", 10), ("ig_steps", 128),
    ])
    def test_the_shipped_values_are_large_enough_to_quote(self, key, floor):
        """GradientSHAP at 24 samples measured a relative standard error of
        0.5689 -- not converged, not quotable. These are the values that make a
        claim, not the ones that finish overnight on a laptop."""
        cfg = load_config("configs/sites.yaml")
        assert getattr(cfg.xai, key) >= floor


class TestCaseSelection:
    """`ids[:n]` is not a sample. The list is in (patient, tooth) order and a
    patient contributes ~14 mandibular sites, so n=20 was two patients."""

    IDS = [f"P{p:02d}#{t}" for p in range(1, 21) for t in range(31, 45)]

    def test_head_truncation_would_have_given_two_patients(self):
        assert len(set(patients_of(self.IDS[:20]))) == 2

    def test_sampling_spreads_across_patients(self):
        ids, _ = select_cases(self.IDS, np.zeros((len(self.IDS), 2), np.float32), 20, seed=0)
        assert len(set(patients_of(ids))) >= 10

    def test_it_is_reproducible(self):
        y = np.zeros((len(self.IDS), 2), np.float32)
        a, _ = select_cases(self.IDS, y, 20, seed=3)
        b, _ = select_cases(self.IDS, y, 20, seed=3)
        assert a == b

    def test_labels_travel_with_their_cases(self):
        y = np.arange(len(self.IDS), dtype=np.float32)[:, None].repeat(2, 1)
        ids, ys = select_cases(self.IDS, y, 12, seed=1)
        for cid, row in zip(ids, ys):
            assert row[0] == float(self.IDS.index(cid))

    def test_asking_for_more_than_exists_is_not_an_error(self):
        ids, _ = select_cases(self.IDS[:5], np.zeros((5, 2), np.float32), 99, seed=0)
        assert len(ids) == 5


class TestAugmentationMatchesTheCache:
    def test_the_site_task_does_not_mirror_an_unverified_axis(self):
        """augment.py mirrors axis 0 as the sagittal plane, which holds only on
        a cache reoriented to RAS+. The site cache keeps the label's own frame,
        so axis 0 is whatever the scan stored."""
        cfg = load_config("configs/sites.yaml")
        assert cfg.augment.flip_lr_prob == 0.0

    def test_it_does_not_interpolate_away_the_resolution_it_paid_for(self):
        """ndimage.rotate(order=1) blurs the native 0.3 mm detail that is the
        entire reason this task caches 52 GB."""
        cfg = load_config("configs/sites.yaml")
        assert cfg.augment.rotate_deg == 0.0 or cfg.augment.rotate_prob == 0.0

    def test_translation_survives(self):
        """Still needed: every patch is cut to the same anatomy by construction,
        so absolute position within the box is a shortcut worth breaking."""
        cfg = load_config("configs/sites.yaml")
        assert cfg.augment.translate_voxels > 0
