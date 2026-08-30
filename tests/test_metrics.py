"""Metric tests against hand-computable cases."""

from __future__ import annotations

import numpy as np
import pytest

from src.train.metrics import (
    auroc,
    average_precision,
    best_f1,
    bootstrap_ci,
    evaluate,
    f1_at_threshold,
)


def test_auroc_perfect_and_inverted():
    y = np.array([0, 0, 1, 1])
    assert auroc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auroc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0


def test_auroc_ties_give_half():
    y = np.array([0, 0, 1, 1])
    assert auroc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5


def test_auroc_undefined_for_single_class():
    assert np.isnan(auroc(np.array([1, 1, 1]), np.array([0.1, 0.5, 0.9])))
    assert np.isnan(auroc(np.array([0, 0, 0]), np.array([0.1, 0.5, 0.9])))


def test_auroc_matches_a_known_value():
    y = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    # Pairs (neg, pos): (0.1,0.4)+ (0.1,0.8)+ (0.35,0.4)+ (0.35,0.8)+ = 4/4
    assert auroc(y, scores) == 1.0


def test_average_precision_perfect():
    y = np.array([0, 0, 1, 1])
    assert average_precision(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0


def test_average_precision_equals_prevalence_for_random_scores():
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.3).astype(int)
    ap = average_precision(y, rng.random(4000))
    assert abs(ap - 0.3) < 0.05


def test_f1_at_threshold():
    y = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.4, 0.6, 0.1])
    # threshold 0.5 -> predict [1,0,1,0]: tp=1 fp=1 fn=1 -> F1 = 2/(2+1+1) = 0.5
    assert f1_at_threshold(y, scores, 0.5) == 0.5


def test_best_f1_finds_a_perfect_split():
    y = np.array([0, 0, 1, 1])
    f1, threshold = best_f1(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert f1 == 1.0 and 0.2 < threshold <= 0.8


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(1)
    y = (rng.random(300) < 0.4).astype(int)
    # Overlapping distributions: AUROC lands well inside (0, 1) so the interval
    # is non-degenerate. (y * 0.6 + noise * 0.4 separates perfectly -> CI [1, 1].)
    scores = y * 0.35 + rng.random(300)
    point, lo, hi = bootstrap_ci(y, scores, auroc, n_boot=300, seed=0)
    assert lo <= point <= hi and 0.0 <= lo < hi <= 1.0
    assert 0.5 < point < 1.0


def test_evaluate_reports_every_label_and_macros():
    rng = np.random.default_rng(2)
    y = (rng.random((200, 6)) < 0.35).astype(int)
    scores = rng.random((200, 6))
    names = [f"l{i}" for i in range(6)]

    out = evaluate(y, scores, names)
    assert set(out["per_label"]) == set(names)
    for key in ("macro_auroc", "macro_ap", "macro_f1"):
        assert np.isfinite(out[key])
    for name in names:
        assert 0.0 <= out["per_label"][name]["auroc"] <= 1.0


def test_evaluate_reuses_supplied_thresholds():
    """Test-set F1 must use the validation threshold, not re-tune on test."""
    rng = np.random.default_rng(3)
    y = (rng.random((120, 2)) < 0.4).astype(int)
    scores = rng.random((120, 2))
    names = ["a", "b"]

    fixed = {"a": 0.5, "b": 0.5}
    out = evaluate(y, scores, names, thresholds=fixed)
    assert out["per_label"]["a"]["threshold"] == 0.5
    assert out["per_label"]["a"]["f1"] == f1_at_threshold(y[:, 0], scores[:, 0], 0.5)


def test_evaluate_with_bootstrap_adds_intervals():
    rng = np.random.default_rng(4)
    y = (rng.random((100, 2)) < 0.4).astype(int)
    scores = rng.random((100, 2))
    out = evaluate(y, scores, ["a", "b"], n_boot=100)
    for name in ("a", "b"):
        lo, hi = out["per_label"][name]["auroc_ci"]
        assert lo <= out["per_label"][name]["auroc"] <= hi
    assert "macro_auroc_ci" in out


class TestNoInformationFloor:
    """A loss means nothing without the floor, and the floor moves with the
    label set. Part B's three-label floor was 1.0652; quoting it against the
    two-label site task would be a category error."""

    def test_balanced_weights_put_the_optimum_at_logit_zero(self):
        from src.train.metrics import no_information_bce

        y = np.zeros((100, 1), dtype=np.float32)
        y[:20] = 1.0                       # prevalence 0.2
        w = np.array([(1 - 0.2) / 0.2])    # the exact balancing weight
        out = no_information_bce(y, w)
        assert out["optimal_logits"][0] == pytest.approx(0.0, abs=1e-9)
        # loss at logit 0 is log(2) per contributing term
        assert out["floor"] == pytest.approx(2 * np.log(2) * 0.8, rel=1e-9)

    def test_unweighted_floor_is_the_entropy_of_the_prevalence(self):
        from src.train.metrics import no_information_bce

        y = np.zeros((100, 1), dtype=np.float32)
        y[:30] = 1.0
        out = no_information_bce(y)
        p = 0.3
        expected = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        assert out["floor"] == pytest.approx(expected, rel=1e-9)

    def test_clamped_weights_move_the_optimum_off_zero(self):
        """pos_weight is clamped in this project, so the optimum is solved for
        rather than assumed to sit at logit 0."""
        from src.train.metrics import no_information_bce

        y = np.zeros((1000, 1), dtype=np.float32)
        y[:10] = 1.0                        # prevalence 0.01, true weight 99
        out = no_information_bce(y, np.array([10.0]))   # clamped
        assert out["optimal_logits"][0] < 0.0

    def test_a_constant_model_cannot_beat_the_floor(self):
        """The definition: no constant prediction scores lower."""
        from src.train.metrics import no_information_bce

        rng = np.random.default_rng(0)
        y = (rng.random((500, 2)) < np.array([0.2, 0.05])).astype(np.float32)
        w = np.array([4.0, 10.0])
        floor = no_information_bce(y, w)["floor"]

        softplus = lambda t: np.logaddexp(0.0, t)  # noqa: E731
        for z in np.linspace(-4, 4, 81):
            loss = np.mean([
                w[j] * y[:, j].mean() * softplus(-z) + (1 - y[:, j].mean()) * softplus(z)
                for j in range(2)
            ])
            assert loss >= floor - 1e-9

    def test_an_empty_label_is_zero_not_nan(self):
        from src.train.metrics import no_information_bce

        y = np.zeros((50, 2), dtype=np.float32)
        y[:10, 0] = 1.0
        out = no_information_bce(y)
        assert np.isfinite(out["floor"])
        assert out["per_label"][1] == 0.0

    def test_reports_prevalence_alongside(self):
        from src.train.metrics import no_information_bce

        y = np.zeros((10, 2), dtype=np.float32)
        y[:5, 0] = 1.0
        assert no_information_bce(y)["prevalence"] == pytest.approx([0.5, 0.0])



# --- The interval is clustered on patients ------------------------------------
#
# `bootstrap_ci` said "over patients" and resampled rows, for three releases. On
# the Part B whole-scan task the claim was true -- one row was one patient. It
# stopped being true when the task moved to sites, and nothing said so, so every
# published AUROC interval was narrower than the data supported. The same fault
# in the XAI bootstrap cost a published claim about Grad-CAM and IG.


def _correlated(n_patients=40, per_patient=14, seed=0):
    """Sites whose label and score are driven by their patient, not by luck."""
    rng = np.random.default_rng(seed)
    groups, y, s = [], [], []
    for p in range(n_patients):
        skill = rng.normal()
        positive = p % 3 == 0
        for _ in range(per_patient):
            groups.append(f"P{p:03d}")
            y.append(1 if positive else 0)
            s.append(skill + (1.2 if positive else 0.0) + 0.1 * rng.normal())
    return np.array(groups), np.array(y), np.array(s)


def test_clustering_widens_the_interval():
    groups, y, s = _correlated()
    _, lo_row, hi_row = bootstrap_ci(y, s, auroc, n_boot=400, seed=1)
    _, lo_pat, hi_pat = bootstrap_ci(y, s, auroc, n_boot=400, seed=1, groups=groups)
    assert hi_pat - lo_pat > hi_row - lo_row, (
        "resampling rows understates the interval when sites share a patient")


def test_clustering_leaves_the_point_estimate_alone():
    """Only the interval changes. A shifted estimate would mean a different bug."""
    groups, y, s = _correlated()
    p_row, _, _ = bootstrap_ci(y, s, auroc, n_boot=200, seed=1)
    p_pat, _, _ = bootstrap_ci(y, s, auroc, n_boot=200, seed=1, groups=groups)
    assert p_row == pytest.approx(p_pat, abs=1e-12)


def test_one_row_per_patient_matches_no_grouping():
    """The whole-volume task must be unaffected -- that is why None stays legal."""
    rng = np.random.default_rng(3)
    y = (rng.random(120) > 0.6).astype(int)
    s = y + 0.4 * rng.normal(size=120)
    groups = np.array([f"P{i:03d}" for i in range(120)])
    _, lo_a, hi_a = bootstrap_ci(y, s, auroc, n_boot=300, seed=5)
    _, lo_b, hi_b = bootstrap_ci(y, s, auroc, n_boot=300, seed=5, groups=groups)
    assert hi_a - lo_a == pytest.approx(hi_b - lo_b, abs=0.05)


def test_mismatched_groups_raise():
    groups, y, s = _correlated(n_patients=4, per_patient=3)
    with pytest.raises(ValueError):
        bootstrap_ci(y, s, auroc, n_boot=10, groups=groups[:-1])


def test_evaluate_threads_groups_through():
    groups, y, s = _correlated()
    yt, ys = y.reshape(-1, 1), s.reshape(-1, 1)
    row = evaluate(yt, ys, ["needs_implant"], n_boot=300, seed=1)
    pat = evaluate(yt, ys, ["needs_implant"], n_boot=300, seed=1, groups=groups)
    lo_r, hi_r = row["per_label"]["needs_implant"]["auroc_ci"]
    lo_p, hi_p = pat["per_label"]["needs_implant"]["auroc_ci"]
    assert hi_p - lo_p > hi_r - lo_r
