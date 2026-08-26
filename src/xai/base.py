"""Shared interface for saliency methods.

Normalisation, token->grid reshaping, upsampling and [0, 1] scaling all live here
so the four methods stay comparable: any difference between their maps must come
from the attribution itself, not from post-processing.

Raw pre-normalisation attributions are kept on `.last_raw` -- Integrated Gradients
needs them for the completeness assertion, which normalised maps would destroy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class SaliencyMethod(ABC):
    """volume (1, 1, D, H, W) -> saliency (D, H, W), float32, normalised to [0, 1]."""

    name: str = "base"

    def __init__(self, model: nn.Module, device: torch.device | None = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.last_raw: torch.Tensor | None = None

    # ---- subclass contract ------------------------------------------------
    @abstractmethod
    def _attribute_raw(self, volume: torch.Tensor, target_label: int) -> torch.Tensor:
        """Return an un-normalised attribution.

        Either voxel-space (D, H, W) or token-space (T,) / (T-1,). Token-space
        maps are reshaped to the model's patch grid and upsampled by the base class.
        """

    # ---- public API -------------------------------------------------------
    def attribute(self, volume: torch.Tensor, target_label: int) -> torch.Tensor:
        volume = self._check_input(volume)
        # Detach at the boundary: the input now carries requires_grad, so an
        # attribution derived from it would otherwise keep the whole graph alive
        # and every returned map would pin activation memory it never uses.
        raw = self._attribute_raw(volume, target_label).detach()
        self.last_raw = raw

        spatial = tuple(volume.shape[2:])
        if raw.ndim == 1:  # token space
            raw = self.tokens_to_volume(raw, spatial)
        elif raw.ndim == 3:
            if tuple(raw.shape) != spatial:
                raw = self._upsample(raw, spatial)
        else:
            raise ValueError(f"{self.name}: unexpected attribution shape {tuple(raw.shape)}")

        return normalize01(raw.float())

    def _check_input(self, volume: torch.Tensor) -> torch.Tensor:
        if volume.ndim != 5 or volume.shape[0] != 1:
            raise ValueError(f"{self.name}: expected (1, C, D, H, W), got {tuple(volume.shape)}")
        volume = volume.to(self.device)
        # The autograd graph must hang off the INPUT, not off the parameters.
        # Grad-CAM and gradient-rollout call .backward() on this tensor directly;
        # without requires_grad here they would depend on the model's weights
        # requiring grad, which makes every backward pass also compute 9.15M
        # weight gradients that no method ever reads.
        if not volume.requires_grad:
            volume = volume.detach().requires_grad_(True)
        return volume

    # ---- token geometry ---------------------------------------------------
    def grid_size(self) -> tuple[int, int, int]:
        """Patch grid taken from the model, never assumed.

        The default config gives 8x8x8, but a different patch_size or img_size
        changes it, so it is always derived.
        """
        grid = getattr(self.model, "grid_size", None)
        if grid is None:
            raise AttributeError(f"{type(self.model).__name__} exposes no grid_size; cannot map tokens to voxels")
        return tuple(grid)

    def tokens_to_volume(self, tokens: torch.Tensor, spatial: tuple[int, int, int]) -> torch.Tensor:
        grid = self.grid_size()
        expected = grid[0] * grid[1] * grid[2]
        if tokens.numel() == expected + 1:  # drop CLS
            tokens = tokens[1:]
        if tokens.numel() != expected:
            raise ValueError(f"{self.name}: {tokens.numel()} tokens, expected {expected} (+1 CLS)")
        return self._upsample(tokens.reshape(*grid), spatial)

    @staticmethod
    def _upsample(vol: torch.Tensor, spatial: tuple[int, int, int]) -> torch.Tensor:
        out = F.interpolate(
            vol[None, None].float(), size=spatial, mode="trilinear", align_corners=False
        )
        return out[0, 0]


def normalize01(x: torch.Tensor) -> torch.Tensor:
    """Scale to [0, 1]. A constant map becomes all-zeros rather than NaN."""
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return torch.zeros_like(x)
    return (x - lo) / (hi - lo)


def gaussian_blur3d(volume: torch.Tensor, sigma: float = 4.0, truncate: float = 3.0) -> torch.Tensor:
    """Separable 3D Gaussian blur -- the default Integrated Gradients baseline.

    A zero volume is the wrong baseline for z-scored CBCT: zero is a real tissue
    intensity, not "absence of signal". Blurring removes structure while keeping
    the intensity distribution plausible.
    """
    if sigma <= 0:
        return volume.clone()
    radius = max(1, int(truncate * sigma + 0.5))
    coords = torch.arange(-radius, radius + 1, device=volume.device, dtype=torch.float32)
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    out = volume
    for axis in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + axis] = kernel.numel()
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (2 - axis)] = radius
        pad[2 * (2 - axis) + 1] = radius
        out = F.conv3d(F.pad(out, pad, mode="replicate"), kernel.view(shape).expand(out.shape[1], 1, *shape[2:]),
                       groups=out.shape[1])
    return out


def make_baseline(volume: torch.Tensor, kind: str = "blur", mean_volume: torch.Tensor | None = None,
                  sigma: float = 4.0) -> torch.Tensor:
    """Baseline for IG / GradientSHAP / deletion-insertion. Never a zero volume."""
    if kind == "blur":
        return gaussian_blur3d(volume, sigma)
    if kind == "mean":
        if mean_volume is None:
            raise ValueError("baseline kind 'mean' requires mean_volume")
        return mean_volume.to(volume.device).expand_as(volume).clone()
    if kind == "zero":  # available for ablation only; documented as inappropriate here
        return torch.zeros_like(volume)
    raise ValueError(f"unknown baseline kind {kind!r} (expected 'blur', 'mean' or 'zero')")


def target_logit(model: nn.Module, volume: torch.Tensor, target_label: int) -> torch.Tensor:
    logits = model(volume)
    if target_label < 0 or target_label >= logits.shape[1]:
        raise IndexError(f"target_label {target_label} outside 0..{logits.shape[1] - 1}")
    return logits[:, target_label]
