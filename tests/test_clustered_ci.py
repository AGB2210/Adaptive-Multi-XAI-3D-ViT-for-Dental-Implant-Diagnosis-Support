"""Patient-clustered bootstrap intervals.

Every XAI figure in this project is measured over SITES, and a patient
contributes up to fourteen of them. The interval that matters is the one over
patients, and the difference is not cosmetic: row resampling narrows it by
roughly the square root of the sites-per-patient ratio and says nothing about
having done so.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.xai.runner import ci_table, clustered_ci, patients_of


def sites(n_patients=20, per_patient=14, spread=1.0, within=0.05, seed=0):
    """Sites whose value is almost entirely determined by their patient.

    This is the structure the real data has -- anatomy, field of view, scanner
    and annotator are shared within a scan -- and it is the structure that makes
    row resampling wrong.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        level = rng.normal(0.0, spread)
        for t in range(per_patient):
            rows.append({"patient_id": f"P{p:03d}",
                         "value": level + rng.normal(0.0, within)})
    return pd.DataFrame(rows)


def row_ci(frame, col="value", n_boot=4000, seed=1337):
    """The naive interval, for comparison only. Never report this one."""
    v = frame[col].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws = [np.median(rng.choice(v, v.size, replace=True)) for _ in range(n_boot)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


class TestItActuallyClusters:
    def test_the_clustered_interval_is_wider_than_the_row_interval(self):
        """The whole point, on data whose correlation is known by construction."""
        frame = sites()
        _, lo, hi = clustered_ci(frame, "value")
        r_lo, r_hi = row_ci(frame)
        assert (hi - lo) > 3 * (r_hi - r_lo), (
            f"clustered width {hi - lo:.3f} against row width {r_hi - r_lo:.3f}. "
            f"With 14 near-identical sites per patient the row interval should "
            f"be far too narrow; if it is not, the grouping is not happening"
        )

    def test_independent_sites_make_the_two_agree(self):
        """The converse: with no within-patient correlation there is nothing to
        correct for, and a clustering that still inflated would be wrong."""
        frame = sites(n_patients=60, per_patient=1, spread=1.0, within=1.0, seed=3)
        _, lo, hi = clustered_ci(frame, "value")
        r_lo, r_hi = row_ci(frame)
        assert 0.5 < (hi - lo) / (r_hi - r_lo) < 2.0

    def test_the_point_estimate_is_the_plain_statistic(self):
        frame = sites()
        point, _, _ = clustered_ci(frame, "value")
        assert point == float(np.median(frame["value"].to_numpy()))

    def test_the_interval_brackets_the_point(self):
        point, lo, hi = clustered_ci(sites(), "value")
        assert lo <= point <= hi

    def test_it_is_reproducible(self):
        frame = sites()
        assert clustered_ci(frame, "value") == clustered_ci(frame, "value")


class TestItRefusesRatherThanInvents:
    def test_all_nan_gives_nan_not_zero(self):
        """A missing interval and an interval of zero width are opposite claims."""
        frame = pd.DataFrame({"patient_id": ["P1", "P1"], "value": [np.nan, np.nan]})
        point, lo, hi = clustered_ci(frame, "value")
        assert np.isnan(point) and np.isnan(lo) and np.isnan(hi)

    def test_partial_nans_are_dropped_not_propagated(self):
        frame = pd.DataFrame({"patient_id": ["P1", "P1", "P2", "P2"],
                              "value": [1.0, np.nan, 3.0, 3.0]})
        point, lo, hi = clustered_ci(frame, "value")
        assert np.isfinite([point, lo, hi]).all()

    def test_one_patient_gives_an_interval_of_that_patient_only(self):
        frame = pd.DataFrame({"patient_id": ["P1"] * 10, "value": np.arange(10.0)})
        point, lo, hi = clustered_ci(frame, "value")
        assert lo == hi == point, (
            "with a single cluster every resample is that cluster, so the "
            "interval is degenerate -- which is the honest answer, not a bug"
        )


class TestCiTable:
    def test_it_reports_one_row_per_method_with_counts(self):
        frames = []
        for i, method in enumerate(["gradcam", "integrated_gradients", "attention_rollout"]):
            f = sites(n_patients=12, seed=i)
            f["method"] = method
            frames.append(f)
        table = ci_table(pd.concat(frames), "value")

        assert list(table.index) == sorted(table.index, key=lambda m: table.loc[m, "value"])
        assert set(table.columns) == {"value", "ci_lo", "ci_hi", "patients", "n"}
        assert (table["patients"] == 12).all()
        assert (table["n"] == 12 * 14).all()
        assert (table["ci_lo"] <= table["value"]).all()
        assert (table["value"] <= table["ci_hi"]).all()


class TestACaseIdIsNotAPatientId(unittest.TestCase):
    """The bootstrap was defeated once by a column NAME, not by its maths.

    `run_faithfulness.py` wrote `patient#tooth` into a column called
    `patient_id`. Every group then held one row, so the clustered bootstrap was
    a row bootstrap, and nothing in the output said so -- the tables were
    printed under the heading "patient-clustered" for two releases. Recomputing
    the published intervals from the CSVs is what found it: 30 randomisation
    cases came from 14 patients.
    """

    def _frame(self, ids):
        return pd.DataFrame({
            "patient_id": ids,
            "method": ["gradcam"] * len(ids),
            "value": np.linspace(0.0, 1.0, len(ids)),
        })

    def test_case_ids_are_refused(self):
        frame = self._frame([f"P{i // 3:03d}#{30 + i % 3}" for i in range(12)])
        with self.assertRaises(ValueError) as caught:
            clustered_ci(frame, "value")
        message = str(caught.exception)
        self.assertIn("case ids", message)
        self.assertIn("patients_of", message)

    def test_patients_of_makes_the_same_frame_acceptable(self):
        ids = [f"P{i // 3:03d}#{30 + i % 3}" for i in range(12)]
        frame = self._frame(ids)
        frame["patient_id"] = patients_of(ids)
        point, lo, hi = clustered_ci(frame, "value")
        self.assertTrue(np.isfinite([point, lo, hi]).all())
        self.assertLessEqual(lo, point)
        self.assertLessEqual(point, hi)

    def test_row_resampling_would_have_been_narrower(self):
        """The size of what was understated, measured rather than asserted."""
        ids = [f"P{i // 4:03d}#{30 + i % 4}" for i in range(40)]
        rng = np.random.default_rng(0)
        # Sites within a patient agree with each other -- that is the whole
        # reason the rows are not independent draws.
        per_patient = {p: rng.normal() for p in set(patients_of(ids))}
        value = [per_patient[p] + 0.05 * rng.normal() for p in patients_of(ids)]

        by_patient = pd.DataFrame({"patient_id": patients_of(ids), "value": value})
        # What the broken column produced: one group per row, under a name that
        # said otherwise.
        by_row = pd.DataFrame({"patient_id": [f"row{i}" for i in range(len(ids))],
                               "value": value})

        _, lo_p, hi_p = clustered_ci(by_patient, "value")
        _, lo_r, hi_r = clustered_ci(by_row, "value")
        self.assertGreater(hi_p - lo_p, hi_r - lo_r,
                           "resampling rows must understate the interval")
