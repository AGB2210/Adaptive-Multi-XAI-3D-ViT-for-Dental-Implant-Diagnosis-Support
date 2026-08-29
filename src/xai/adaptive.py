"""The adaptive layer: a confidence gate plus agreement-weighted fusion.

Two mechanisms:

1. Confidence gate -- confident cases get Attention Rollout alone (cheapest);
   uncertain / boundary cases get the full ensemble and fusion. Thresholds are
   tuned on validation only.

2. Agreement-weighted fusion -- each method is scored per input, scores are
   softmax-normalised into weights, and the maps are fused as sum(w_i * S_i).
   The weights are PER-INPUT; that is the entire "adaptive" claim, and a fixed
   uniform ensemble is benchmarked against it rather than assumed inferior.

CIRCULARITY GUARD
-----------------
Weighting methods by a metric and then declaring victory on that same metric is
circular. This module therefore separates the two by construction:

    WEIGHT_METRIC = insertion_auc      (higher is better)
    EVAL_METRIC   = deletion_auc       (lower is better)

`fuse()` refuses to run if asked to weight and evaluate on the same metric.
"""

from __future__ import annotations

import numpy as np
import torch

from src.xai.base import normalize01
from src.xai.faithfulness import deletion_insertion

WEIGHT_METRIC = "insertion_auc"
EVAL_METRIC = "deletion_auc"
# Deletion is better when LOW, insertion when HIGH.
LOWER_IS_BETTER = {"deletion_auc": True, "insertion_auc": False}


def softmax_weights(scores: dict[str, float], metric: str, temperature: float = 1.0) -> dict[str, float]:
    """Turn per-method faithfulness scores into fusion weights."""
    names = list(scores)
    values = np.array([scores[n] for n in names], dtype=np.float64)

    if np.all(~np.isfinite(values)):
        return {n: 1.0 / len(names) for n in names}
    values = np.nan_to_num(values, nan=float(np.nanmean(values)))

    if LOWER_IS_BETTER.get(metric, False):
        values = -values

    # Standardise before softmax so the temperature means the same thing
    # regardless of the metric's raw scale.
    std = values.std()
    if std > 1e-12:
        values = (values - values.mean()) / std
    exp = np.exp(values / max(temperature, 1e-6))
    weights = exp / exp.sum()
    return {n: float(w) for n, w in zip(names, weights)}


def fuse_maps(maps: dict[str, torch.Tensor], weights: dict[str, float]) -> torch.Tensor:
    """S_fused = sum(w_i * S_i), each S_i normalised to [0, 1] first."""
    fused = None
    for name, saliency in maps.items():
        contribution = weights.get(name, 0.0) * normalize01(saliency)
        fused = contribution if fused is None else fused + contribution
    if fused is None:
        raise ValueError("no maps to fuse")
    return normalize01(fused)


def uniform_weights(names) -> dict[str, float]:
    names = list(names)
    return {n: 1.0 / len(names) for n in names}


def score_methods(
    model,
    volume: torch.Tensor,
    maps: dict[str, torch.Tensor],
    target_label: int,
    metric: str = WEIGHT_METRIC,
    steps: int = 50,
    baseline: torch.Tensor | None = None,
    target_is_probability: bool = True,
) -> dict[str, float]:
    """Faithfulness score per method on one input, using `metric` only."""
    scores = {}
    for name, saliency in maps.items():
        result = deletion_insertion(model, volume, saliency, target_label,
                                    baseline=baseline, steps=steps,
                                    target_is_probability=target_is_probability)
        scores[name] = result[metric]
    return scores


def fuse(
    model,
    volume: torch.Tensor,
    maps: dict[str, torch.Tensor],
    target_label: int,
    weight_metric: str = WEIGHT_METRIC,
    eval_metric: str = EVAL_METRIC,
    temperature: float = 1.0,
    steps: int = 50,
    baseline: torch.Tensor | None = None,
    target_is_probability: bool = True,
) -> dict:
    """Agreement-weighted fusion with the circularity guard enforced.

    `target_is_probability=False` when the explained head is a millimetre head
    -- see `deletion_insertion`. Both the weighting and the held-out evaluation
    have to read the same units, so it is one flag for the whole call.
    """
    if weight_metric == eval_metric:
        raise ValueError(
            f"circular evaluation: weighting and evaluating both on {weight_metric!r}. "
            "Weight on insertion_auc and evaluate on deletion_auc (or vice versa)."
        )

    scores = score_methods(model, volume, maps, target_label, weight_metric, steps, baseline,
                           target_is_probability=target_is_probability)
    weights = softmax_weights(scores, weight_metric, temperature)
    fused = fuse_maps(maps, weights)

    uniform = fuse_maps(maps, uniform_weights(maps))

    # Evaluate everything on the held-out metric.
    di = dict(baseline=baseline, steps=steps, target_is_probability=target_is_probability)
    fused_eval = deletion_insertion(model, volume, fused, target_label, **di)
    uniform_eval = deletion_insertion(model, volume, uniform, target_label, **di)
    per_method_eval = {
        name: deletion_insertion(model, volume, s, target_label, **di)[eval_metric]
        for name, s in maps.items()
    }

    better = min if LOWER_IS_BETTER.get(eval_metric, False) else max
    best_individual = better(per_method_eval.values())

    return {
        "fused_map": fused,
        "uniform_map": uniform,
        "weights": weights,
        "weight_metric": weight_metric,
        "eval_metric": eval_metric,
        "weight_scores": scores,
        "fused_eval": fused_eval[eval_metric],
        "uniform_eval": uniform_eval[eval_metric],
        "per_method_eval": per_method_eval,
        "beats_best_individual": bool(
            fused_eval[eval_metric] < best_individual
            if LOWER_IS_BETTER.get(eval_metric, False)
            else fused_eval[eval_metric] > best_individual
        ),
        "beats_uniform": bool(
            fused_eval[eval_metric] < uniform_eval[eval_metric]
            if LOWER_IS_BETTER.get(eval_metric, False)
            else fused_eval[eval_metric] > uniform_eval[eval_metric]
        ),
    }


class ConfidenceGate:
    """Route confident cases to the cheap method, uncertain ones to the ensemble.

    `threshold` is an uncertainty quantile fitted on VALIDATION. Cases above it
    are treated as boundary cases and get the full ensemble.
    """

    def __init__(self, threshold: float, cheap_method: str = "attention_rollout"):
        self.threshold = threshold
        self.cheap_method = cheap_method

    @classmethod
    def fit(cls, val_uncertainty: np.ndarray, ensemble_fraction: float = 0.3,
            cheap_method: str = "attention_rollout") -> "ConfidenceGate":
        """Pick the threshold so `ensemble_fraction` of validation cases escalate."""
        if not 0.0 <= ensemble_fraction <= 1.0:
            raise ValueError(f"ensemble_fraction must be in [0, 1], got {ensemble_fraction}")
        threshold = float(np.quantile(np.asarray(val_uncertainty), 1.0 - ensemble_fraction))
        return cls(threshold, cheap_method)

    def use_ensemble(self, case_uncertainty: float) -> bool:
        return bool(case_uncertainty >= self.threshold)

    def route(self, case_uncertainty: float, all_methods: list[str]) -> list[str]:
        return list(all_methods) if self.use_ensemble(case_uncertainty) else [self.cheap_method]


def pareto_sweep(
    per_case: list[dict],
    runtimes: dict[str, float],
    fractions=(0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0),
    cheap_method: str = "attention_rollout",
) -> list[dict]:
    """Sweep the gate threshold: mean measured compute cost vs mean faithfulness.

    `per_case` needs, per case: 'uncertainty', 'cheap_eval' (cheap method's score)
    and 'fused_eval' (ensemble score), both on the EVAL metric.
    `runtimes` is measured seconds per method -- the adaptive layer's whole
    justification is compute, so these must be measured, never assumed.
    `cheap_method` must match the gate's, and must appear in `runtimes`.
    """
    if not per_case:
        return []

    uncertainties = np.array([c["uncertainty"] for c in per_case])
    if cheap_method not in runtimes:
        # Silently costing the cheap path at zero would make the adaptive policy
        # look free, which is the one number this figure exists to establish.
        raise KeyError(
            f"no measured runtime for cheap method {cheap_method!r}; "
            f"have {sorted(runtimes)}. Run scripts/run_xai.py first."
        )
    cheap_cost = runtimes[cheap_method]
    full_cost = sum(runtimes.values())

    rows = []
    for fraction in fractions:
        threshold = float(np.quantile(uncertainties, 1.0 - fraction)) if 0 < fraction < 1 else (
            np.inf if fraction == 0 else -np.inf
        )
        scores, costs = [], []
        for case in per_case:
            escalate = case["uncertainty"] >= threshold
            scores.append(case["fused_eval"] if escalate else case["cheap_eval"])
            costs.append(full_cost if escalate else cheap_cost)

        rows.append({
            "ensemble_fraction_target": fraction,
            "ensemble_fraction_actual": float(np.mean(uncertainties >= threshold)),
            "mean_cost_seconds": float(np.mean(costs)),
            "mean_faithfulness": float(np.mean(scores)),
            "n_cases": len(per_case),
        })
    return rows
