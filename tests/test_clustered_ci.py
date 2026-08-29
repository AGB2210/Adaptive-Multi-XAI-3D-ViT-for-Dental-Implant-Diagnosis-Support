"""The bootstrap must cluster by patient, and the difference must be visible.

Resampling rows instead of patients is the kind of fault that produces a
narrower interval and says nothing about it. These pin the behaviour that
distinguishes the two, so a refactor cannot quietly revert to row resampling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.xai.runner import ci_table, clustered_ci


def one_value_per_patient(n_patients=40, sites=14, seed=0):
    """Every site of a patient carries that patient's value exactly.

    The extreme case of within-patient correlation. Row resampling sees
    n_patients * sites independent draws; there are only n_patients.
    """
    rng = np.random.default_rng(seed)
    per_patient = rng.normal(0.0, 1.0, n_patients)
    return pd.DataFrame({
        "patient_id": np.repeat([f"p{i}" for i in range(n_patients)], sites),
        "v": np.repeat(per_patient, sites),
    })


def test_returns_nan_not_zero_when_nothing_is_finite():
    df = pd.DataFrame({"patient_id": ["a", "a"], "v": [np.nan, np.nan]})
    point, lo, hi = clustered_ci(df, "v")
    assert all(np.isnan(x) for x in (point, lo, hi))


def test_empty_frame_does_not_raise():
    point, lo, hi = clustered_ci(pd.DataFrame({"patient_id": [], "v": []}), "v")
    assert np.isnan(point)


def test_point_estimate_matches_the_plain_statistic():
    df = one_value_per_patient()
    point, _, _ = clustered_ci(df, "v")
    assert abs(point - np.median(df.v)) < 1e-9


def test_interval_brackets_the_point_estimate():
    point, lo, hi = clustered_ci(one_value_per_patient(), "v")
    assert lo <= point <= hi


def test_clustering_gives_a_wider_interval_than_resampling_rows():
    """The whole reason the helper exists.

    With every site of a patient carrying an identical value, a row-wise
    bootstrap treats 560 correlated observations as independent and reports an
    interval far too narrow. Clustering must be visibly wider.
    """
    df = one_value_per_patient()

    _, lo_c, hi_c = clustered_ci(df, "v", seed=1)

    rng = np.random.default_rng(1)
    vals = df.v.to_numpy()
    draws = [np.median(rng.choice(vals, vals.size, replace=True)) for _ in range(2000)]
    lo_r, hi_r = np.percentile(draws, [2.5, 97.5])

    assert (hi_c - lo_c) > 2 * (hi_r - lo_r), (
        f"clustered width {hi_c - lo_c:.4f} is not meaningfully wider than "
        f"row-wise {hi_r - lo_r:.4f} -- the bootstrap is not clustering"
    )


def test_deterministic_for_a_fixed_seed():
    df = one_value_per_patient()
    assert clustered_ci(df, "v", seed=7) == clustered_ci(df, "v", seed=7)


def test_accepts_a_different_statistic():
    df = one_value_per_patient()
    mean_point, _, _ = clustered_ci(df, "v", stat=np.mean)
    assert abs(mean_point - df.v.mean()) < 1e-9


def test_ci_table_reports_one_row_per_method_with_patient_counts():
    df = one_value_per_patient(n_patients=6, sites=3)
    df["method"] = ["a", "b", "c"] * (len(df) // 3)
    tab = ci_table(df, "v")
    assert set(tab.index) == {"a", "b", "c"}
    assert (tab.patients == 6).all()
    assert (tab.ci_lo <= tab["v"]).all() and (tab["v"] <= tab.ci_hi).all()
