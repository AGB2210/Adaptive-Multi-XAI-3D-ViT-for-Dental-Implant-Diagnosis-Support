"""Calibration, the confidence gate, and fusion.

The most important test here is the circularity guard: weighting methods by a
metric and then declaring victory on that same metric is scientifically empty,
so `fuse()` must refuse to do it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.vit3d import ViT3D
from src.xai import build_ensemble
from src.xai.adaptive import (
    EVAL_METRIC,
    WEIGHT_METRIC,
    ConfidenceGate,
    fuse,
    fuse_maps,
    pareto_sweep,
    softmax_weights,
    uniform_weights,
)
from src.xai.calibration import (
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    uncertainty,
)

IMG = 32


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = ViT3D(img_size=IMG, stem_channels=8, embed_dim=32, patch_size=4, depth=2,
              num_heads=4, drop_path=0.0, num_classes=6)
    m.eval()
    return m


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------
def test_temperature_recovers_a_known_overconfidence():
    """Logits inflated by 3x should be corrected by a temperature near 3."""
    rng = np.random.default_rng(0)
    true_logits = rng.normal(0, 1.5, size=(3000, 4))
    probs = 1 / (1 + np.exp(-true_logits))
    targets = (rng.random(probs.shape) < probs).astype(float)

    temperature = fit_temperature(true_logits * 3.0, targets)
    assert 2.0 < temperature < 4.5, temperature


def test_temperature_scaling_reduces_ece_on_overconfident_scores():
    rng = np.random.default_rng(1)
    true_logits = rng.normal(0, 1.5, size=(3000, 4))
    probs = 1 / (1 + np.exp(-true_logits))
    targets = (rng.random(probs.shape) < probs).astype(float)
    inflated = true_logits * 3.0

    before, _ = expected_calibration_error(1 / (1 + np.exp(-inflated)), targets)
    temperature = fit_temperature(inflated, targets)
    after, _ = expected_calibration_error(apply_temperature(inflated, temperature), targets)
    assert after < before


def test_calibrating_on_millimetres_raises_instead_of_returning_nan():
    """The fault that killed the whole adaptive layer, as a test.

    `run_adaptive.py` passed the FULL hybrid label matrix to both functions:
    `needs_implant` alongside two columns of millimetres.
    `binary_cross_entropy_with_logits` does not range-check its target, so a
    target of 22.5 mm gives a loss that falls monotonically in T with no
    minimum. LBFGS walked log T to infinity, `fit_temperature` returned NaN,
    every calibrated probability and uncertainty became NaN, and the gate
    threshold -- a quantile of NaN -- compared False against every case. The
    gate never escalated anything and the Pareto sweep collapsed to one point,
    in silence, after the full compute had been spent.

    ECE showed the same input as 9.097, which is the giveaway: ECE is a
    weighted mean of |accuracy - confidence| and cannot exceed 1 when both are
    probabilities.
    """
    rng = np.random.default_rng(7)
    logits = rng.normal(0, 1.5, size=(200, 3))
    hybrid = np.column_stack([
        (rng.random(200) < 0.3).astype(float),   # needs_implant
        rng.normal(14.0, 3.0, 200),              # available_height_mm
        rng.normal(8.0, 1.5, 200),               # ridge_width_mm
    ])

    with pytest.raises(ValueError, match=r"binary targets"):
        fit_temperature(logits, hybrid)
    with pytest.raises(ValueError, match=r"binary targets"):
        expected_calibration_error(1 / (1 + np.exp(-logits)), hybrid)

    # Sliced to the binary block, both work and ECE lands back inside [0, 1].
    ece, _ = expected_calibration_error(1 / (1 + np.exp(-logits[:, :1])), hybrid[:, :1])
    assert 0.0 <= ece <= 1.0
    assert np.isfinite(fit_temperature(logits[:, :1], hybrid[:, :1]))


def test_ece_rejects_scores_that_are_not_probabilities():
    """Raw millimetre predictions reaching the ECE, from the other direction."""
    targets = np.array([[0.0], [1.0], [1.0], [0.0]])
    with pytest.raises(ValueError, match=r"probs must lie"):
        expected_calibration_error(np.array([[13.4], [18.1], [21.0], [9.8]]), targets)


def test_ece_is_zero_for_perfectly_calibrated_predictions():
    rng = np.random.default_rng(2)
    probs = rng.random((20000, 1))
    targets = (rng.random(probs.shape) < probs).astype(float)
    ece, bins = expected_calibration_error(probs, targets, n_bins=10)
    assert ece < 0.02
    assert sum(b["n"] for b in bins) == probs.size


def test_uncertainty_margin_peaks_at_the_boundary():
    probs = np.array([[0.99, 0.01], [0.50, 0.99], [0.90, 0.80]])
    u = uncertainty(probs, "margin")
    assert np.argmax(u) == 1  # the row containing p = 0.5


def test_uncertainty_entropy_peaks_at_the_boundary():
    probs = np.array([[0.99, 0.99], [0.5, 0.5]])
    u = uncertainty(probs, "entropy")
    assert u[1] > u[0]


def test_unknown_uncertainty_mode_raises():
    with pytest.raises(ValueError, match="unknown uncertainty mode"):
        uncertainty(np.array([[0.5]]), "bogus")


# --------------------------------------------------------------------------
# weighting and fusion
# --------------------------------------------------------------------------
def test_softmax_weights_sum_to_one():
    w = softmax_weights({"a": 0.2, "b": 0.5, "c": 0.9}, "insertion_auc")
    assert sum(w.values()) == pytest.approx(1.0)
    assert all(v >= 0 for v in w.values())


def test_higher_insertion_auc_earns_more_weight():
    w = softmax_weights({"good": 0.9, "bad": 0.1}, "insertion_auc")
    assert w["good"] > w["bad"]


def test_lower_deletion_auc_earns_more_weight():
    """Deletion is better when LOW, so the ordering must invert."""
    w = softmax_weights({"good": 0.1, "bad": 0.9}, "deletion_auc")
    assert w["good"] > w["bad"]


def test_softmax_weights_survive_all_nan_scores():
    w = softmax_weights({"a": float("nan"), "b": float("nan")}, "insertion_auc")
    assert sum(w.values()) == pytest.approx(1.0)


def test_fuse_maps_is_a_weighted_combination():
    a = torch.zeros(4, 4, 4)
    b = torch.ones(4, 4, 4)
    a[0, 0, 0] = 1.0
    fused = fuse_maps({"a": a, "b": b}, {"a": 0.5, "b": 0.5})
    assert fused.shape == (4, 4, 4)
    assert float(fused.min()) >= 0.0 and float(fused.max()) <= 1.0


def test_uniform_weights_are_equal():
    w = uniform_weights(["a", "b", "c", "d"])
    assert all(v == pytest.approx(0.25) for v in w.values())


def test_fuse_rejects_circular_metric_choice(model):
    """The guard that keeps the headline claim meaningful."""
    volume = torch.randn(1, 1, IMG, IMG, IMG)
    maps = {"a": torch.rand(IMG, IMG, IMG), "b": torch.rand(IMG, IMG, IMG)}
    with pytest.raises(ValueError, match="circular"):
        fuse(model, volume, maps, 0, weight_metric="deletion_auc", eval_metric="deletion_auc")


def test_default_metrics_are_not_the_same():
    assert WEIGHT_METRIC != EVAL_METRIC


def test_fuse_reports_both_comparisons(model):
    torch.manual_seed(0)
    volume = torch.randn(1, 1, IMG, IMG, IMG)
    maps = {n: m.attribute(volume, 0) for n, m in
            build_ensemble(model, torch.device("cpu"), names=("gradcam", "attention_rollout")).items()}

    result = fuse(model, volume, maps, 0, steps=6)
    assert set(result["weights"]) == set(maps)
    assert sum(result["weights"].values()) == pytest.approx(1.0)
    assert isinstance(result["beats_best_individual"], bool)
    assert isinstance(result["beats_uniform"], bool)
    assert result["weight_metric"] == WEIGHT_METRIC and result["eval_metric"] == EVAL_METRIC


# --------------------------------------------------------------------------
# confidence gate
# --------------------------------------------------------------------------
def test_gate_escalates_the_requested_fraction():
    rng = np.random.default_rng(0)
    val = rng.random(1000)
    gate = ConfidenceGate.fit(val, ensemble_fraction=0.3)
    assert np.mean(val >= gate.threshold) == pytest.approx(0.3, abs=0.03)


def test_gate_routes_confident_cases_to_the_cheap_method():
    gate = ConfidenceGate(threshold=0.5)
    assert gate.route(0.1, ["a", "b", "c"]) == ["attention_rollout"]
    assert gate.route(0.9, ["a", "b", "c"]) == ["a", "b", "c"]


def test_gate_rejects_an_invalid_fraction():
    with pytest.raises(ValueError, match="ensemble_fraction"):
        ConfidenceGate.fit(np.random.random(10), ensemble_fraction=1.5)


# --------------------------------------------------------------------------
# Pareto sweep
# --------------------------------------------------------------------------
def test_pareto_sweep_spans_both_fixed_policies():
    rng = np.random.default_rng(0)
    per_case = [
        {"uncertainty": float(u), "cheap_eval": 0.6, "fused_eval": 0.4}
        for u in rng.random(50)
    ]
    runtimes = {"attention_rollout": 0.1, "gradcam": 0.2, "integrated_gradients": 1.0, "gradient_shap": 0.8}
    rows = pareto_sweep(per_case, runtimes)

    assert rows[0]["ensemble_fraction_actual"] == pytest.approx(0.0)
    assert rows[-1]["ensemble_fraction_actual"] == pytest.approx(1.0)
    # Cost must rise monotonically with the escalated fraction.
    costs = [r["mean_cost_seconds"] for r in rows]
    assert costs == sorted(costs)
    # Always-cheap costs exactly the rollout; always-full costs the whole ensemble.
    assert costs[0] == pytest.approx(0.1)
    assert costs[-1] == pytest.approx(sum(runtimes.values()))


def test_pareto_sweep_handles_no_cases():
    assert pareto_sweep([], {"attention_rollout": 0.1}) == []


def test_pareto_sweep_refuses_an_unmeasured_cheap_method():
    """Costing the cheap path at zero would make the adaptive policy look free.

    The Pareto figure exists to establish exactly that number, so a missing
    runtime must fail rather than default.
    """
    per_case = [{"uncertainty": 0.1, "cheap_eval": 0.6, "fused_eval": 0.4}]
    with pytest.raises(KeyError, match="no measured runtime"):
        pareto_sweep(per_case, {"gradcam": 0.2}, cheap_method="attention_rollout")


def test_pareto_sweep_honours_a_non_default_cheap_method():
    per_case = [{"uncertainty": float(u), "cheap_eval": 0.6, "fused_eval": 0.4}
                for u in (0.1, 0.5, 0.9)]
    runtimes = {"attention_rollout": 0.1, "gradcam": 0.2}
    rows = pareto_sweep(per_case, runtimes, fractions=(0.0,), cheap_method="gradcam")
    assert rows[0]["mean_cost_seconds"] == pytest.approx(0.2)
