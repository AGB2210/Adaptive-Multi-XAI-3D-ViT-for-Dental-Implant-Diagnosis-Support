"""Patient-level splits, stratified on the multi-label matrix.

One CBCT per patient, so patient-level disjointness is guaranteed by splitting
the patient list itself. The split is seeded and persisted to artifacts/splits.json
so every later phase (including the XAI phase) reuses exactly the same partition.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def iterative_stratification(
    y: np.ndarray, ratios: tuple[float, ...], seed: int = 0
) -> list[list[int]]:
    """Sechidis/Tsoumakas iterative stratification for multi-label data.

    Greedy: repeatedly take the rarest remaining label and hand its samples to
    whichever fold is furthest below its quota for that label. Keeps per-label
    prevalence close across folds, which naive random splitting does not do
    reliably at a few hundred cases.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    n, n_labels = y.shape
    ratios = np.asarray(ratios, dtype=float)
    ratios = ratios / ratios.sum()

    folds: list[list[int]] = [[] for _ in ratios]
    desired_total = ratios * n
    desired_per_label = np.outer(ratios, y.sum(axis=0))  # (folds, labels)

    remaining = set(range(n))
    label_sets = {j: {i for i in range(n) if y[i, j]} for j in range(n_labels)}

    while remaining:
        # Rarest label with anything left to place.
        counts = {j: len(label_sets[j] & remaining) for j in range(n_labels)}
        active = {j: c for j, c in counts.items() if c > 0}

        if not active:  # samples with no positive labels: fill by remaining quota
            leftovers = sorted(remaining)
            rng.shuffle(leftovers)
            for i in leftovers:
                # desired_total is already the *remaining* quota; subtracting the
                # fold size again would double-count and starve the largest fold.
                chosen = int(np.argmax(desired_total))
                folds[chosen].append(i)
                desired_total[chosen] -= 1
            break

        label = min(active, key=lambda j: (active[j], j))
        members = sorted(label_sets[label] & remaining)
        rng.shuffle(members)

        for i in members:
            # Only consider folds that still have room overall. Without this the
            # per-label quotas alone drive every choice (exact float ties are rare,
            # so the total-size tie-break almost never fires) and the fold sizes
            # drift far from the requested ratios -- measured 255 instead of 282
            # for a 70% train split at n=403 on an earlier cohort.
            candidates = np.flatnonzero(desired_total > 0)
            if candidates.size == 0:
                candidates = np.arange(len(folds))

            # Clamp at zero: once a fold has met its quota for this label, further
            # negative drift must not make it look worse than another exhausted
            # fold. Clamping lets exhausted folds tie so the total-size tie-break
            # below actually decides -- otherwise the fold that absorbed the
            # label-rich patients is never chosen again and ends up undersized.
            want = np.maximum(desired_per_label[candidates, label], 0.0)
            best = candidates[np.flatnonzero(want == want.max())]
            if len(best) > 1:  # tie-break on remaining overall quota, then at random
                gaps = desired_total[best]
                best = best[np.flatnonzero(gaps == gaps.max())]
            chosen = int(best[0]) if len(best) == 1 else int(rng.choice(best))

            folds[chosen].append(i)
            remaining.discard(i)
            desired_per_label[chosen] -= y[i]
            desired_total[chosen] -= 1

    return [sorted(f) for f in folds]


def make_splits(
    patient_ids: list[str],
    y: np.ndarray,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 1337,
) -> dict[str, list[str]]:
    if len(patient_ids) != len(y):
        raise ValueError(f"{len(patient_ids)} ids vs {len(y)} label rows")
    folds = iterative_stratification(y, ratios, seed=seed)
    names = ["train", "val", "test"]
    return {name: [patient_ids[i] for i in fold] for name, fold in zip(names, folds)}


def make_cv_folds(
    patient_ids: list[str],
    y: np.ndarray,
    n_folds: int = 5,
    seed: int = 1337,
) -> list[list[str]]:
    """K stratified folds of patient ids.

    Cross-validation rather than a single split is a response to how few positive
    cases some labels have. With ~74 positives, a 15% test split holds ~11 of
    them and the resulting confidence interval is wide enough to contain chance
    whatever the model does. Rotating the test fold lets every positive case be
    scored exactly once, so the estimate rests on all of them.

    The same iterative stratification is used, so each fold carries roughly the
    cohort's per-label prevalence.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}")
    if len(patient_ids) != len(y):
        raise ValueError(f"{len(patient_ids)} ids vs {len(y)} label rows")
    ratios = tuple([1.0 / n_folds] * n_folds)
    folds = iterative_stratification(y, ratios, seed=seed)
    return [[patient_ids[i] for i in fold] for fold in folds]


def fold_assignment(folds: list[list[str]], k: int) -> dict[str, list[str]]:
    """train/val/test for cross-validation round `k`.

    Fold k is the test set and fold k+1 (wrapping) is validation, so across the
    whole run every case is tested exactly once and validated exactly once, and
    the model is never selected on the fold it is scored on. With 5 folds each
    round is a 60/20/20 split.
    """
    n = len(folds)
    if not 0 <= k < n:
        raise ValueError(f"fold {k} out of range for {n} folds")
    test = folds[k]
    val = folds[(k + 1) % n]
    train = [pid for i, f in enumerate(folds) if i not in (k, (k + 1) % n) for pid in f]
    return {"train": train, "val": val, "test": test}


def save_folds(folds: list[list[str]], path: str | Path, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"folds": folds, "meta": meta or {}}, indent=2), encoding="utf-8")


def load_folds(path: str | Path) -> list[list[str]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["folds"]


def check_folds_disjoint(folds: list[list[str]]) -> None:
    """Every case appears in exactly one fold. A leak here contaminates every round."""
    seen: dict[str, int] = {}
    for k, ids in enumerate(folds):
        if len(ids) != len(set(ids)):
            dupes = [i for i in set(ids) if ids.count(i) > 1]
            raise ValueError(f"fold {k} contains duplicates: {dupes[:5]}")
        for pid in ids:
            if pid in seen:
                raise ValueError(f"case {pid!r} appears in folds {seen[pid]} and {k}")
            seen[pid] = k


def save_splits(splits: dict[str, list[str]], path: str | Path, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"splits": splits, "meta": meta or {}}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_splits(path: str | Path) -> dict[str, list[str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["splits"]


def check_disjoint(splits: dict[str, list[str]]) -> None:
    """A patient's volume must never appear in two splits."""
    seen: dict[str, str] = {}
    for name, ids in splits.items():
        if len(ids) != len(set(ids)):
            dupes = [i for i in set(ids) if ids.count(i) > 1]
            raise ValueError(f"split {name!r} contains duplicates: {dupes[:5]}")
        for pid in ids:
            if pid in seen:
                raise ValueError(f"patient {pid!r} appears in both {seen[pid]!r} and {name!r}")
            seen[pid] = name


def split_prevalence(splits: dict[str, list[str]], patient_ids: list[str], y: np.ndarray) -> dict:
    index = {pid: i for i, pid in enumerate(patient_ids)}
    return {
        name: {
            "n": len(ids),
            "prevalence": y[[index[p] for p in ids]].mean(axis=0).round(4).tolist() if ids else [],
        }
        for name, ids in splits.items()
    }
