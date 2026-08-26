"""Multi-XAI ensemble, adaptive fusion, and faithfulness evaluation."""

from __future__ import annotations

import torch

from src.xai.base import SaliencyMethod, gaussian_blur3d, make_baseline, normalize01
from src.xai.gradcam import GradCAM3D
from src.xai.gradient_shap import GradientSHAP
from src.xai.integrated_gradients import IntegratedGradients
from src.xai.rollout import AttentionRollout, GradientWeightedRollout

# The four methods of the main ensemble. LIME is deliberately excluded -- it is
# available via build_method("lime3d") for the ablation only.
ENSEMBLE_METHODS = ("attention_rollout", "gradcam", "integrated_gradients", "gradient_shap")


def build_method(name: str, model, device=None, **kwargs) -> SaliencyMethod:
    if name == "attention_rollout":
        return AttentionRollout(model, device, **kwargs)
    if name == "grad_rollout":
        return GradientWeightedRollout(model, device, **kwargs)
    if name == "gradcam":
        return GradCAM3D(model, device, **kwargs)
    if name == "integrated_gradients":
        return IntegratedGradients(model, device, **kwargs)
    if name == "gradient_shap":
        return GradientSHAP(model, device, **kwargs)
    if name == "lime3d":
        from src.xai.lime3d import Lime3D

        return Lime3D(model, device, **kwargs)
    raise ValueError(f"unknown saliency method {name!r}")


def build_ensemble(
    model,
    device=None,
    names: tuple[str, ...] = ENSEMBLE_METHODS,
    mean_volume: torch.Tensor | None = None,
    baselines: torch.Tensor | None = None,
    **kwargs,
) -> dict[str, SaliencyMethod]:
    """Instantiate the ensemble, passing each method only the options it accepts."""
    methods = {}
    for name in names:
        opts = dict(kwargs.get(name, {}))
        if name == "integrated_gradients" and mean_volume is not None:
            opts.setdefault("mean_volume", mean_volume)
        if name == "gradient_shap" and baselines is not None:
            opts.setdefault("baselines", baselines)
        methods[name] = build_method(name, model, device, **opts)
    return methods


__all__ = [
    "SaliencyMethod",
    "AttentionRollout",
    "GradientWeightedRollout",
    "GradCAM3D",
    "IntegratedGradients",
    "GradientSHAP",
    "ENSEMBLE_METHODS",
    "build_method",
    "build_ensemble",
    "normalize01",
    "make_baseline",
    "gaussian_blur3d",
]
