"""Split tests. A patient leaking across splits invalidates every reported number."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.splits import (
    check_disjoint,
    check_folds_disjoint,
    fold_assignment,
    load_folds,
    load_splits,
    make_cv_folds,
    make_splits,
    save_folds,
    save_splits,
    split_prevalence,
)


def fake_labels(n=200, n_labels=6, seed=0):
    rng = np.random.default_rng(seed)
    p = np.array([0.44, 0.40, 0.37, 0.34, 0.27, 0.25])[:n_labels]
    y = (rng.random((n, n_labels)) < p).astype(int)
    ids = [f"p{i}" for i in range(n)]
    return ids, y


def test_splits_are_disjoint_and_complete():
    ids, y = fake_labels()
    splits = make_splits(ids, y, (0.7, 0.15, 0.15), seed=1337)
    check_disjoint(splits)

    allocated = [p for fold in splits.values() for p in fold]
    assert sorted(allocated) == sorted(ids), "every patient must land in exactly one split"


def test_split_sizes_match_ratios():
    ids, y = fake_labels(n=403)
    splits = make_splits(ids, y, (0.7, 0.15, 0.15), seed=1337)
    assert abs(len(splits["train"]) - 0.70 * 403) < 12
    assert abs(len(splits["val"]) - 0.15 * 403) < 12
    assert abs(len(splits["test"]) - 0.15 * 403) < 12


def test_stratification_keeps_prevalence_close():
    ids, y = fake_labels(n=403)
    splits = make_splits(ids, y, (0.7, 0.15, 0.15), seed=1337)
    stats = split_prevalence(splits, ids, y)

    overall = y.mean(axis=0)
    for name in ("train", "val", "test"):
        got = np.array(stats[name]["prevalence"])
        # Iterative stratification should hold every label within 10 points.
        assert np.max(np.abs(got - overall)) < 0.10, (name, got, overall)


def test_check_disjoint_catches_a_leak():
    with pytest.raises(ValueError, match="appears in both"):
        check_disjoint({"train": ["a", "b"], "val": ["b"], "test": ["c"]})


def test_check_disjoint_catches_duplicates_within_a_split():
    with pytest.raises(ValueError, match="duplicates"):
        check_disjoint({"train": ["a", "a"], "val": ["b"], "test": []})


def test_splits_are_reproducible_under_the_same_seed():
    ids, y = fake_labels()
    assert make_splits(ids, y, (0.7, 0.15, 0.15), seed=7) == make_splits(ids, y, (0.7, 0.15, 0.15), seed=7)


def test_different_seeds_give_different_splits():
    ids, y = fake_labels()
    a = make_splits(ids, y, (0.7, 0.15, 0.15), seed=1)
    b = make_splits(ids, y, (0.7, 0.15, 0.15), seed=2)
    assert a != b


def test_all_zero_label_patients_are_still_allocated():
    ids = [f"p{i}" for i in range(50)]
    y = np.zeros((50, 6), dtype=int)
    y[:10, 0] = 1  # only a few positives anywhere
    splits = make_splits(ids, y, (0.7, 0.15, 0.15), seed=0)
    check_disjoint(splits)
    assert sum(len(v) for v in splits.values()) == 50


def test_save_and_load_roundtrip(tmp_path):
    ids, y = fake_labels(n=60)
    splits = make_splits(ids, y, (0.7, 0.15, 0.15), seed=3)
    path = tmp_path / "splits.json"
    save_splits(splits, path, meta={"seed": 3})
    assert load_splits(path) == splits


def test_mismatched_lengths_raise():
    ids, y = fake_labels(n=20)
    with pytest.raises(ValueError, match="label rows"):
        make_splits(ids[:10], y, (0.7, 0.15, 0.15))


# --- cross-validation -------------------------------------------------------
# The failure these guard against is a leak: a case that appears in two folds is
# tested by a model that trained on it, and nothing downstream would notice.

def _cv_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random((n, 3)) < np.array([0.14, 0.36, 0.19])).astype(int)
    return [f"c{i:03d}" for i in range(n)], y


def test_cv_folds_partition_every_case_exactly_once():
    ids, y = _cv_data()
    folds = make_cv_folds(ids, y, n_folds=5, seed=1337)
    check_folds_disjoint(folds)
    assert len(folds) == 5
    assert sorted(p for f in folds for p in f) == sorted(ids)


def test_cv_folds_are_roughly_equal_and_stratified():
    ids, y = _cv_data(n=500)
    folds = make_cv_folds(ids, y, n_folds=5, seed=1337)
    sizes = [len(f) for f in folds]
    assert max(sizes) - min(sizes) <= 2, sizes

    index = {p: i for i, p in enumerate(ids)}
    overall = y.mean(axis=0)
    for f in folds:
        prev = y[[index[p] for p in f]].mean(axis=0)
        # Stratification is greedy, not exact; this catches a fold that has lost
        # a label entirely rather than demanding equality.
        assert np.all(np.abs(prev - overall) < 0.12), (prev, overall)


def test_every_case_is_tested_once_and_validated_once():
    ids, y = _cv_data()
    folds = make_cv_folds(ids, y, n_folds=5, seed=1337)
    tested, validated = [], []
    for k in range(len(folds)):
        a = fold_assignment(folds, k)
        check_disjoint(a)                       # no leak within a round
        assert sorted(a["train"] + a["val"] + a["test"]) == sorted(ids)
        tested += a["test"]
        validated += a["val"]
    assert sorted(tested) == sorted(ids)
    assert sorted(validated) == sorted(ids)


def test_model_is_never_selected_on_the_fold_it_is_scored_on():
    ids, y = _cv_data()
    folds = make_cv_folds(ids, y, n_folds=5, seed=1337)
    for k in range(len(folds)):
        a = fold_assignment(folds, k)
        assert not set(a["val"]) & set(a["test"])
        assert not set(a["train"]) & set(a["test"])


def test_fold_index_is_bounds_checked():
    ids, y = _cv_data(n=60)
    folds = make_cv_folds(ids, y, n_folds=3, seed=1)
    with pytest.raises(ValueError, match="out of range"):
        fold_assignment(folds, 3)


def test_folds_round_trip_through_disk(tmp_path):
    ids, y = _cv_data(n=80)
    folds = make_cv_folds(ids, y, n_folds=4, seed=7)
    path = tmp_path / "cv_folds.json"
    save_folds(folds, path, meta={"seed": 7, "n_folds": 4})
    assert load_folds(path) == folds


def test_duplicate_across_folds_is_caught():
    with pytest.raises(ValueError, match="appears in folds"):
        check_folds_disjoint([["a", "b"], ["b", "c"]])
