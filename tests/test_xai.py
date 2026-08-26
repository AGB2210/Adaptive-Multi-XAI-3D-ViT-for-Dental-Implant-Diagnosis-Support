"""The four saliency methods.

These run against a small randomly-initialised ViT. That is enough to verify the
*mathematics* (completeness, normalisation, shapes, determinism, class
sensitivity). Whether a method finds real anatomy is a separate question, tested
by the synthetic planted-signal check in scripts/run_xai.py against a trained
checkpoint.
"""

from __future__ import annotations

import pytest
import torch

from src.models.vit3d import ViT3D
from src.xai import build_ensemble, build_method
from src.xai.base import gaussian_blur3d, make_baseline, normalize01
from src.xai.rollout import residual_normalize, rollout_from_attentions

IMG = 32
METHODS = ["attention_rollout", "grad_rollout", "gradcam", "integrated_gradients", "gradient_shap"]


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = ViT3D(img_size=IMG, stem_channels=8, embed_dim=32, patch_size=4, depth=2,
              num_heads=4, drop_path=0.0, num_classes=6)
    m.eval()
    return m


@pytest.fixture(scope="module")
def volume():
    torch.manual_seed(1)
    return torch.randn(1, 1, IMG, IMG, IMG)


# --------------------------------------------------------------------------
# base-class contract
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", METHODS)
def test_output_shape_range_and_finiteness(model, volume, name):
    method = build_method(name, model, torch.device("cpu"))
    saliency = method.attribute(volume, target_label=2)

    assert saliency.shape == (IMG, IMG, IMG), name
    assert torch.isfinite(saliency).all(), f"{name} produced NaN/Inf"
    assert float(saliency.min()) >= 0.0 and float(saliency.max()) <= 1.0, name
    assert saliency.dtype == torch.float32


@pytest.mark.parametrize("name", METHODS)
def test_deterministic_under_fixed_seed(model, volume, name):
    a = build_method(name, model, torch.device("cpu")).attribute(volume, 1)
    b = build_method(name, model, torch.device("cpu")).attribute(volume, 1)
    assert torch.allclose(a, b, atol=1e-5), f"{name} is not deterministic"


@pytest.mark.parametrize("name", METHODS)
def test_raw_attribution_is_kept(model, volume, name):
    method = build_method(name, model, torch.device("cpu"))
    method.attribute(volume, 0)
    assert method.last_raw is not None, f"{name} discarded its pre-normalisation attribution"


def test_rejects_wrong_input_shape(model):
    method = build_method("gradcam", model, torch.device("cpu"))
    with pytest.raises(ValueError, match="expected"):
        method.attribute(torch.randn(1, IMG, IMG, IMG), 0)
    with pytest.raises(ValueError, match="expected"):
        method.attribute(torch.randn(2, 1, IMG, IMG, IMG), 0)


def test_normalize01_handles_constant_and_nan():
    assert torch.equal(normalize01(torch.full((4, 4), 3.0)), torch.zeros(4, 4))
    out = normalize01(torch.tensor([float("nan"), 1.0, 2.0]))
    assert torch.isfinite(out).all()


# --------------------------------------------------------------------------
# rollout
# --------------------------------------------------------------------------
def test_residual_normalize_rows_sum_to_one():
    torch.manual_seed(0)
    a = torch.softmax(torch.randn(6, 6), dim=-1)
    out = residual_normalize(a)
    assert torch.allclose(out.sum(-1), torch.ones(6), atol=1e-5)
    # Residual term must raise the diagonal.
    assert (out.diagonal() > a.diagonal()).all()


def test_rollout_needs_captured_attention():
    with pytest.raises(RuntimeError, match="set_store_attention"):
        rollout_from_attentions([])


def test_plain_rollout_is_class_agnostic(model, volume):
    """Documented limitation, asserted so it cannot silently change."""
    method = build_method("attention_rollout", model, torch.device("cpu"))
    a = method.attribute(volume, 0)
    b = method.attribute(volume, 4)
    assert torch.allclose(a, b, atol=1e-6), "plain rollout should NOT depend on the target label"


def test_gradient_weighted_rollout_is_class_specific(model, volume):
    """This is exactly why the gradient-weighted variant exists."""
    method = build_method("grad_rollout", model, torch.device("cpu"))
    a = method.attribute(volume, 0)
    b = method.attribute(volume, 4)
    assert not torch.allclose(a, b, atol=1e-4), "grad_rollout must vary with the target label"


def test_attention_state_is_restored_after_use(model, volume):
    build_method("attention_rollout", model, torch.device("cpu")).attribute(volume, 0)
    assert all(not b.attn.store_attention for b in model.blocks), "storage flag left on"


# --------------------------------------------------------------------------
# Grad-CAM
# --------------------------------------------------------------------------
def test_gradcam_is_class_specific(model, volume):
    method = build_method("gradcam", model, torch.device("cpu"))
    assert not torch.allclose(method.attribute(volume, 0), method.attribute(volume, 3), atol=1e-4)


def test_gradcam_raw_is_token_space_without_cls(model, volume):
    method = build_method("gradcam", model, torch.device("cpu"))
    method.attribute(volume, 0)
    grid = model.grid_size
    assert method.last_raw.numel() == grid[0] * grid[1] * grid[2]


# --------------------------------------------------------------------------
# Integrated Gradients — the completeness axiom
# --------------------------------------------------------------------------
def test_ig_completeness_axiom_holds(model, volume):
    """sum(IG) == F(x) - F(x'). IG's core guarantee: if this fails the
    implementation is wrong, so the test fails rather than warning.

    Uses 256 steps: this fixture is a RANDOMLY INITIALISED net whose
    F(x) - F(x') is ~0.04, so the relative-error denominator is tiny and the
    path is more erratic than a trained model's. Convergence with step count is
    asserted separately below -- that is what actually proves the estimator.
    """
    method = build_method("integrated_gradients", model, torch.device("cpu"), steps=256)
    method.attribute(volume, target_label=1)
    error = method.last_completeness_error
    assert error < 0.05, f"completeness violated: relative error {error:.4f} exceeds 5%"


def test_ig_completeness_converges_with_step_count(model, volume):
    """The real proof of correctness: error must fall monotonically as the
    Riemann sum is refined. A biased implementation would plateau instead."""
    errors = []
    for steps in (16, 64, 256):
        method = build_method("integrated_gradients", model, torch.device("cpu"), steps=steps)
        method.attribute(volume, 1)
        errors.append(method.last_completeness_error)

    assert errors == sorted(errors, reverse=True), f"not converging: {errors}"
    assert errors[-1] < errors[0] / 5, f"convergence too slow: {errors}"


def test_ig_flags_a_completeness_violation(model, volume, caplog):
    """Too few steps must WARN, never pass silently -- a wrong attribution that
    looks fine is the failure mode this whole check exists to prevent."""
    method = build_method("integrated_gradients", model, torch.device("cpu"), steps=4)
    method.attribute(volume, 1)
    assert method.last_completeness_error > 0.05
    assert method.completeness_ok is False


def test_ig_baseline_is_not_zero_by_default(model, volume):
    """Zero is a real tissue intensity on z-scored CBCT, not 'absence'."""
    baseline = make_baseline(volume, "blur")
    assert not torch.allclose(baseline, torch.zeros_like(baseline))
    # Blurring must preserve the overall intensity level, not null it out.
    assert abs(float(baseline.mean()) - float(volume.mean())) < 0.5


def test_ig_mean_baseline_requires_a_mean_volume(volume):
    with pytest.raises(ValueError, match="mean_volume"):
        make_baseline(volume, "mean")


def test_gaussian_blur_preserves_shape_and_reduces_variance(volume):
    blurred = gaussian_blur3d(volume, sigma=3.0)
    assert blurred.shape == volume.shape
    assert float(blurred.std()) < float(volume.std())


# --------------------------------------------------------------------------
# GradientSHAP
# --------------------------------------------------------------------------
def test_gradient_shap_reports_its_own_variance(model, volume):
    method = build_method("gradient_shap", model, torch.device("cpu"), n_samples=8)
    method.attribute(volume, 0)
    assert method.last_variance == method.last_variance  # not NaN
    assert method.last_variance >= 0


def test_gradient_shap_variance_falls_with_more_samples(model, volume):
    few = build_method("gradient_shap", model, torch.device("cpu"), n_samples=4, seed=0)
    many = build_method("gradient_shap", model, torch.device("cpu"), n_samples=32, seed=0)
    few.attribute(volume, 0)
    many.attribute(volume, 0)
    assert many.last_variance < few.last_variance


# --------------------------------------------------------------------------
# ensemble wiring
# --------------------------------------------------------------------------
def test_build_ensemble_returns_every_method(model):
    ensemble = build_ensemble(model, torch.device("cpu"))
    assert set(ensemble) == {"attention_rollout", "gradcam", "integrated_gradients", "gradient_shap"}


def test_unknown_method_raises(model):
    with pytest.raises(ValueError, match="unknown saliency method"):
        build_method("nope", model, torch.device("cpu"))


def test_token_geometry_is_derived_from_the_model(model):
    method = build_method("gradcam", model, torch.device("cpu"))
    assert method.grid_size() == model.grid_size


def test_lime_runs_as_an_ablation(model, volume):
    method = build_method("lime3d", model, torch.device("cpu"), grid=(4, 4, 4), n_samples=16)
    saliency = method.attribute(volume, 0)
    assert saliency.shape == (IMG, IMG, IMG) and torch.isfinite(saliency).all()


# ---------------------------------------------------------------------------
# Regression tests for defects found in the pre-push audit.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("integrated_gradients", {"steps": 8}),
        ("gradient_shap", {"n_samples": 4}),
        ("gradcam", {}),
        ("attention_rollout", {}),
    ],
)
def test_attribution_leaves_no_gradients_on_model_parameters(model, volume, name, kwargs):
    """Attribution must not leave gradients in the model's parameter buffers.

    IG and GradientSHAP backpropagate once per integration step, and those passes
    accumulate into every parameter's .grad. Left behind they waste memory and
    silently contaminate anything that later reads .grad -- a resumed optimiser,
    or the next attribution method to run.
    """
    model.zero_grad(set_to_none=True)
    build_method(name, model, **kwargs).attribute(volume, 0)

    dirty = [n for n, p in model.named_parameters()
             if p.grad is not None and float(p.grad.abs().sum()) > 0]
    model.zero_grad(set_to_none=True)
    assert dirty == [], f"{name} left gradients on {len(dirty)} parameters, e.g. {dirty[:3]}"


def test_lime_grid_is_independent_of_the_model_patch_grid(model, volume):
    """LIME's supervoxel partition is its own; it must not have to match the ViT.

    Returning a flat token vector made every grid except the model's patch grid
    raise inside tokens_to_volume.
    """
    coarse = tuple(g // 2 for g in model.grid_size)
    assert coarse != tuple(model.grid_size)

    saliency = build_method("lime3d", model, grid=coarse, n_samples=8).attribute(volume, 0)
    assert saliency.shape == tuple(volume.shape[2:])
    assert torch.isfinite(saliency).all()


# ---- gradients live on the input, not on the weights -----------------------

def test_frozen_parameters_still_allow_every_method(model, volume):
    """The autograd graph must hang off the input.

    Attribution needs gradients to reach the input, which requires the INPUT to
    have requires_grad -- not the 9.15M parameters. Grad-CAM and gradient-rollout
    call .backward() on the volume directly, so if they silently depended on the
    weights requiring grad, freezing them would raise "does not require grad".
    """
    for param in model.parameters():
        param.requires_grad_(False)

    for name in ("attention_rollout", "grad_rollout", "gradcam",
                 "integrated_gradients", "gradient_shap"):
        method = build_method(name, model, torch.device("cpu"),
                              **({"steps": 4} if name == "integrated_gradients" else {}),
                              **({"n_samples": 2} if name == "gradient_shap" else {}))
        out = method.attribute(volume, 0)
        assert out.shape == volume.shape[2:], name
        assert torch.isfinite(out).all(), name


def test_no_weight_gradients_are_computed_when_parameters_are_frozen(model, volume):
    """The point of freezing: no .grad buffers on the weights at all."""
    for param in model.parameters():
        param.requires_grad_(False)

    build_method("integrated_gradients", model, torch.device("cpu"), steps=4).attribute(volume, 0)

    assert all(p.grad is None for p in model.parameters())


def test_check_input_supplies_the_graph(model, volume):
    method = build_method("gradcam", model, torch.device("cpu"))
    checked = method._check_input(volume.detach())
    assert checked.requires_grad
