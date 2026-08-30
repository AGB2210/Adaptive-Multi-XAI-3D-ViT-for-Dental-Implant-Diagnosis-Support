"""Multi-label metrics with bootstrap confidence intervals.

At the sample sizes this project works with -- hundreds of cases, tens in the
test split -- a point estimate alone is not publishable, so every headline number
carries a 95% percentile-bootstrap interval.

AUROC and average precision are implemented directly rather than pulled from
sklearn so the tie handling is explicit and auditable.
"""

from __future__ import annotations

import numpy as np


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUROC via the Mann-Whitney U statistic, with mid-ranks for ties.

    Returns NaN when a label has only one class present -- undefined, not 0.5.
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=np.float64)
    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):  # mid-rank for tied scores
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1

    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Step-wise AP, the same estimator as sklearn's average_precision_score."""
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    if y_true.sum() == 0:
        return float("nan")

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    scores_sorted = y_score[order]

    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    # Collapse tied scores to their last index: a threshold cannot split a tie.
    keep = np.r_[np.diff(scores_sorted) != 0, True]
    tp, fp = tp[keep], fp[keep]

    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / y_true.sum()
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def f1_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> float:
    pred = (np.asarray(y_score) >= threshold).astype(int)
    y_true = np.asarray(y_true).astype(int)
    tp = int((pred & y_true).sum())
    fp = int((pred & (1 - y_true)).sum())
    fn = int(((1 - pred) & y_true).sum())
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom else 0.0


def best_f1(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Best F1 over candidate thresholds, and the threshold that achieves it.

    Tune this on validation only -- tuning on test leaks and inflates the number.
    """
    scores = np.unique(np.asarray(y_score, dtype=np.float64))
    if len(scores) == 0:
        return 0.0, 0.5
    if len(scores) > 512:  # keep it cheap inside the bootstrap
        scores = np.quantile(scores, np.linspace(0, 1, 512))
    best, best_t = -1.0, 0.5
    for t in scores:
        f = f1_at_threshold(y_true, y_score, t)
        if f > best:
            best, best_t = f, float(t)
    return best, best_t


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    fn,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
    groups: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """(point estimate, lo, hi) by percentile bootstrap, clustered on `groups`.

    `groups` is one label per row saying which INDEPENDENT unit the row belongs
    to -- a patient. Pass it whenever a unit contributes more than one row.

    ON THE SITE TASK A ROW IS NOT AN INDEPENDENT DRAW. One patient supplies up
    to fourteen mandibular sites that share anatomy, field of view, scanner and
    annotator, so resampling rows treats 6,787 correlated observations as 6,787
    independent ones and returns an interval far narrower than the data
    supports.

    This docstring used to say "over patients" while the code resampled rows,
    and it stayed that way through every AUROC interval this project published.
    On the Part B whole-scan task the claim was true -- one row really was one
    patient -- and it silently stopped being true when the task moved to sites.
    `groups=None` therefore means "every row is its own unit", which is correct
    for a whole-volume task and wrong for a per-site one; the caller must say
    which, because this function cannot tell.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    point = fn(y_true, y_score)

    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = np.empty(n_boot, dtype=np.float64)

    if groups is None:
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            stats[b] = fn(y_true[idx], y_score[idx])
    else:
        groups = np.asarray(groups)
        if len(groups) != n:
            raise ValueError(f"groups has {len(groups)} entries for {n} rows")
        _, inverse = np.unique(groups, return_inverse=True)
        members = [np.flatnonzero(inverse == g) for g in range(inverse.max() + 1)]
        k = len(members)
        for b in range(n_boot):
            idx = np.concatenate([members[i] for i in rng.integers(0, k, k)])
            stats[b] = fn(y_true[idx], y_score[idx])

    stats = stats[np.isfinite(stats)]
    if stats.size == 0:
        return point, float("nan"), float("nan")
    alpha = (1.0 - ci) / 2.0
    return point, float(np.quantile(stats, alpha)), float(np.quantile(stats, 1 - alpha))


def evaluate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label_names: list[str],
    thresholds: dict[str, float] | None = None,
    n_boot: int = 0,
    ci: float = 0.95,
    seed: int = 0,
    groups: np.ndarray | None = None,
) -> dict:
    """Per-label AUROC / AP / F1 plus macro averages.

    `thresholds` should come from the validation split; if omitted, F1 is tuned
    on the data passed in (correct for val, leaky for test -- pass them in).
    Set n_boot > 0 for confidence intervals. **Pass `groups` with them on any
    task where one patient contributes several rows**, or every interval below
    will be narrower than the data supports -- see `bootstrap_ci`.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    out: dict = {"per_label": {}, "thresholds": {}}

    for i, name in enumerate(label_names):
        yt, ys = y_true[:, i], y_score[:, i]

        if thresholds and name in thresholds:
            threshold = thresholds[name]
            f1 = f1_at_threshold(yt, ys, threshold)
        else:
            f1, threshold = best_f1(yt, ys)

        entry = {
            "auroc": auroc(yt, ys),
            "ap": average_precision(yt, ys),
            "f1": f1,
            "threshold": threshold,
            "n_pos": int(yt.sum()),
            "prevalence": float(yt.mean()),
        }

        if n_boot:
            _, lo, hi = bootstrap_ci(yt, ys, auroc, n_boot, ci, seed + i, groups)
            entry["auroc_ci"] = [lo, hi]
            _, lo, hi = bootstrap_ci(yt, ys, average_precision, n_boot, ci, seed + i, groups)
            entry["ap_ci"] = [lo, hi]
            _, lo, hi = bootstrap_ci(yt, ys, lambda a, b, t=threshold: f1_at_threshold(a, b, t),
                                     n_boot, ci, seed + i, groups)
            entry["f1_ci"] = [lo, hi]

        out["per_label"][name] = entry
        out["thresholds"][name] = threshold

    for key in ("auroc", "ap", "f1"):
        values = [out["per_label"][n][key] for n in label_names]
        finite = int(np.isfinite(values).sum())
        # `nanmean` over a partly-undefined set is a macro over FEWER labels than
        # the heading says. The all-NaN case was guarded; the partial one was not.
        out[f"macro_{key}"] = float(np.nanmean(values)) if finite else float("nan")
        if finite and finite < len(label_names):
            out[f"macro_{key}_n_labels"] = finite

    if n_boot:  # macro AUROC CI, resampled once for all labels
        rng = np.random.default_rng(seed + 999)
        if groups is None:
            def _draw():
                return rng.integers(0, len(y_true), size=len(y_true))
        else:
            _, inverse = np.unique(np.asarray(groups), return_inverse=True)
            members = [np.flatnonzero(inverse == g) for g in range(inverse.max() + 1)]

            def _draw():
                return np.concatenate(
                    [members[i] for i in rng.integers(0, len(members), len(members))])
        stats = []
        for _ in range(n_boot):
            idx = _draw()
            vals = [auroc(y_true[idx, i], y_score[idx, i]) for i in range(len(label_names))]
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                stats.append(float(np.mean(vals)))
        if stats:
            alpha = (1.0 - ci) / 2.0
            out["macro_auroc_ci"] = [float(np.quantile(stats, alpha)), float(np.quantile(stats, 1 - alpha))]

    return out


def format_metrics(metrics: dict, label_names: list[str], title: str = "") -> str:
    has_ci = any("auroc_ci" in metrics["per_label"][n] for n in label_names)
    lines = []
    if title:
        lines += [title, "=" * len(title)]

    header = f"{'label':<18}{'n+':>5}{'prev':>7}{'AUROC':>8}"
    if has_ci:
        header += f"{'95% CI':>16}"
    header += f"{'AP':>8}{'F1':>8}{'thr':>7}"
    lines += [header, "-" * len(header)]

    for name in label_names:
        e = metrics["per_label"][name]
        row = f"{name:<18}{e['n_pos']:>5}{e['prevalence']:>7.2f}{e['auroc']:>8.3f}"
        if has_ci:
            lo, hi = e.get("auroc_ci", [float("nan")] * 2)
            row += f"  [{lo:>5.3f},{hi:>6.3f}]"
        row += f"{e['ap']:>8.3f}{e['f1']:>8.3f}{e['threshold']:>7.3f}"
        lines.append(row)

    macro = f"{'MACRO':<18}{'':>5}{'':>7}{metrics['macro_auroc']:>8.3f}"
    if has_ci and "macro_auroc_ci" in metrics:
        lo, hi = metrics["macro_auroc_ci"]
        macro += f"  [{lo:>5.3f},{hi:>6.3f}]"
    elif has_ci:
        macro += " " * 16
    macro += f"{metrics['macro_ap']:>8.3f}{metrics['macro_f1']:>8.3f}"
    lines += ["-" * len(header), macro]
    return "\n".join(lines)


def no_information_bce(y: np.ndarray, pos_weight=None) -> dict:
    """Loss of a model that has learned nothing. Report it beside every loss.

    A training loss is uninterpretable on its own: whether 2.70 is progress or
    noise depends entirely on where chance sits, and chance moves with the label
    set. Part B's three-label floor was 1.0652; quoting it against a two-label
    task would be a category error.

    The floor is the loss of the best CONSTANT prediction -- a model that ignores
    the image entirely. For label j with prevalence p and pos_weight w, the
    optimal constant logit is log(w*p / (1-p)); with the exact balancing weight
    w = (1-p)/p that is logit 0, but pos_weight is clamped in this project, so
    the optimum is solved for rather than assumed.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    n_labels = y.shape[1]

    if pos_weight is None:
        weights = np.ones(n_labels)
    else:
        weights = np.asarray(
            pos_weight.detach().cpu().numpy() if hasattr(pos_weight, "detach") else pos_weight,
            dtype=np.float64,
        ).reshape(-1)
        if weights.size == 1:
            weights = np.repeat(weights, n_labels)

    softplus = lambda t: np.logaddexp(0.0, t)  # noqa: E731
    per_label, logits = [], []
    for j in range(n_labels):
        p, w = float(y[:, j].mean()), float(weights[j])
        if p <= 0.0 or p >= 1.0:
            # A label with no positives (or no negatives) has a degenerate
            # optimum: the constant model is perfect and the floor is zero.
            per_label.append(0.0)
            logits.append(float("-inf") if p <= 0.0 else float("inf"))
            continue
        z = float(np.log(w * p / (1.0 - p)))
        per_label.append(float(w * p * softplus(-z) + (1.0 - p) * softplus(z)))
        logits.append(z)

    return {
        "floor": float(np.mean(per_label)),
        "per_label": per_label,
        "optimal_logits": logits,
        "prevalence": y.mean(axis=0).tolist(),
    }
