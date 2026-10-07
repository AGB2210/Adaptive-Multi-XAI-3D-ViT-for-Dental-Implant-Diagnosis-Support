"""Explain one site prediction the way the adaptive layer does.

The gate and the fusion are `src.xai.adaptive` -- `ConfidenceGate` and `fuse`
-- unchanged. Confident cases get attention rollout alone; uncertain ones get
all four methods and the agreement-weighted fusion, with the circularity guard
intact (weighted on insertion AUC, evaluated on deletion AUC).

Three differences from the research scripts, each stated on the result rather
than hidden in it:

  * GradientSHAP baselines. The scripts draw them from TRAINING patches of the
    checkpoint's own fold. The app has no training cache, so it uses blurred
    copies of the input -- GradientSHAP's own documented fallback.
  * The gate threshold comes from `calibration.json` when the checkpoint has
    one. Without it a fixed default is used and labelled as unfitted.
  * On a millimetre head the fusion integrates `score="deviation"`, which is the
    reading `deletion_insertion` documents as founded for a length; the
    research default `"response"` assumes a probability.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from src.inference.checkpoint import ModelBundle
from src.xai import ENSEMBLE_METHODS, build_ensemble, build_method
from src.xai.adaptive import EVAL_METRIC, WEIGHT_METRIC, ConfidenceGate, fuse
from src.xai.base import make_baseline
from src.xai.calibration import uncertainty

CHEAP = "attention_rollout"

# What a progress line calls each method.
LABELS = {"attention_rollout": "Attention rollout", "gradcam": "Grad-CAM",
          "integrated_gradients": "Integrated Gradients", "gradient_shap": "GradientSHAP"}


def gate_for(bundle: ModelBundle, default_threshold: float) -> tuple[ConfidenceGate, str]:
    if bundle.gate:
        return (ConfidenceGate(bundle.gate["threshold"], bundle.gate.get("cheap_method", CHEAP)),
                f"fitted on validation ({bundle.calibration_source})")
    return (ConfidenceGate(float(default_threshold), CHEAP),
            "app default -- NOT fitted on this model's validation split")


def site_uncertainty(bundle: ModelBundle, logits: list[float]) -> float:
    """Margin uncertainty of the calibrated binary block, as the gate was fitted on."""
    n = bundle.n_binary
    if not n:
        return float("nan")
    z = np.asarray(logits[:n], dtype=np.float64)[None, :]
    probs = 1.0 / (1.0 + np.exp(-z / bundle.temperature))
    mode = (bundle.gate or {}).get("uncertainty", "margin")
    return float(uncertainty(probs, mode)[0])


def explain_patch(
    bundle: ModelBundle,
    volume: torch.Tensor,
    target: int,
    logits: list[float],
    settings: dict,
    force_ensemble: bool = False,
    progress=None,
) -> dict:
    """Maps (each (D, H, W) in [0, 1]) and the evidence behind the routing."""
    def step(message, fraction):
        if progress is not None:
            progress(message, fraction)

    names, n_bin = bundle.names, bundle.n_binary
    if not 0 <= target < len(names):
        raise ValueError(f"target {target} outside the model's {len(names)} outputs")

    gate, gate_source = gate_for(bundle, settings.get("default_gate_threshold", -0.25))
    unc = site_uncertainty(bundle, logits)
    # No binary head means no confidence to gate on, so the full ensemble runs.
    escalate = bool(force_ensemble or not np.isfinite(unc) or gate.use_ensemble(unc))
    routing = {
        "uncertainty": None if not np.isfinite(unc) else unc,
        "threshold": gate.threshold,
        "source": gate_source,
        "forced": bool(force_ensemble),
        "decision": "ensemble" if escalate else "cheap",
    }
    result = {"target": names[target],
              "target_unit": "probability" if target < n_bin else "mm",
              "routing": routing, "maps": {}, "timings_s": {}, "notes": []}

    if not escalate:
        step(LABELS.get(gate.cheap_method, gate.cheap_method), 0.3)
        t0 = time.perf_counter()
        method = build_method(gate.cheap_method, bundle.model, bundle.device)
        result["maps"][gate.cheap_method] = method.attribute(volume, target).cpu()
        result["timings_s"][gate.cheap_method] = time.perf_counter() - t0
        if gate.cheap_method == CHEAP:
            result["notes"].append("attention rollout is class-agnostic: it is the same "
                                   "map whichever output is selected")
        return result

    methods = build_ensemble(
        bundle.model, bundle.device, names=ENSEMBLE_METHODS,
        integrated_gradients={"steps": int(settings.get("ig_steps", 256)),
                              "batch_size": int(settings.get("ig_batch", 4))},
        gradient_shap={"n_samples": int(settings.get("shap_samples", 200)),
                       "batch_size": int(settings.get("shap_batch", 4))},
    )
    maps = {}
    for i, (name, method) in enumerate(methods.items()):
        step(LABELS.get(name, name), 0.05 + 0.6 * i / len(methods))
        t0 = time.perf_counter()
        maps[name] = method.attribute(volume, target)
        result["timings_s"][name] = time.perf_counter() - t0

    ig = methods.get("integrated_gradients")
    if ig is not None and ig.last_completeness:
        result["ig_completeness"] = ig.last_completeness
    shap = methods.get("gradient_shap")
    if shap is not None and np.isfinite(shap.last_variance):
        result["shap_relative_se"] = float(shap.last_variance)

    step("Agreement-weighted fusion", 0.7)
    is_prob = target < n_bin
    score = "response" if is_prob else "deviation"
    t0 = time.perf_counter()
    fused = fuse(bundle.model, volume, maps, target,
                 weight_metric=WEIGHT_METRIC, eval_metric=EVAL_METRIC,
                 steps=int(settings.get("fusion_steps", 50)),
                 baseline=make_baseline(volume, "blur"),
                 target_is_probability=is_prob, score=score)
    result["timings_s"]["fusion"] = time.perf_counter() - t0

    result["maps"] = {k: v.cpu() for k, v in maps.items()}
    result["maps"]["fused"] = fused["fused_map"].cpu()
    result["fusion"] = {
        "weights": fused["weights"],
        "weight_metric": fused["weight_metric"],
        "eval_metric": fused["eval_metric"],
        "weight_scores": fused["weight_scores"],
        "per_method_eval": fused["per_method_eval"],
        "fused_eval": fused["fused_eval"],
        "uniform_eval": fused["uniform_eval"],
        "beats_best_individual": fused["beats_best_individual"],
        "beats_uniform": fused["beats_uniform"],
        "score": score,
    }
    result["notes"].append("GradientSHAP baselines are blurred copies of this patch: the app "
                           "has no training patches to draw from, as the research runs did")
    if not is_prob:
        result["notes"].append("deletion/insertion on a millimetre head is read as deviation "
                               "from the full-input prediction (score='deviation')")
    step("Done", 1.0)
    return result
