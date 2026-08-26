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
) -> tuple[float, float, float]:
    """(point estimate, lo, hi) by percentile bootstrap over patients."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    point = fn(y_true, y_score)

    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
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
) -> dict:
    """Per-label AUROC / AP / F1 plus macro averages.

    `thresholds` should come from the validation split; if omitted, F1 is tuned
    on the data passed in (correct for val, leaky for test -- pass them in).
    Set n_boot > 0 for confidence intervals.
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
            _, lo, hi = bootstrap_ci(yt, ys, auroc, n_boot, ci, seed + i)
            entry["auroc_ci"] = [lo, hi]
            _, lo, hi = bootstrap_ci(yt, ys, average_precision, n_boot, ci, seed + i)
            entry["ap_ci"] = [lo, hi]
            _, lo, hi = bootstrap_ci(yt, ys, lambda a, b, t=threshold: f1_at_threshold(a, b, t), n_boot, ci, seed + i)
            entry["f1_ci"] = [lo, hi]

        out["per_label"][name] = entry
        out["thresholds"][name] = threshold

    for key in ("auroc", "ap", "f1"):
        values = [out["per_label"][n][key] for n in label_names]
        out[f"macro_{key}"] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")

    if n_boot:  # macro AUROC CI, resampling patients once for all labels
        rng = np.random.default_rng(seed + 999)
        stats = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(y_true), size=len(y_true))
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
