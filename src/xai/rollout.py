"""Attention Rollout (Abnar & Zuidema, 2020) and a gradient-weighted variant.

Rollout composes per-layer attention while accounting for residual connections:

    A_hat = 0.5 * A + 0.5 * I      (row-normalised)
    R     = A_hat_L @ ... @ A_hat_1

then reads the CLS row.

KNOWN LIMITATION -- plain rollout is CLASS-AGNOSTIC. It is computed purely from
attention weights, so it returns the same map for every target_label. That is a
property of the method, not a bug here, and it is why the gradient-weighted
variant exists: it scales each layer's attention by ReLU(d y_target / d A),
making the map class-specific. Both are reported.
"""

from __future__ import annotations

import torch

from src.xai.base import SaliencyMethod


def fuse_heads(attn: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """(B, heads, T, T) -> (T, T) for a single-item batch."""
    if attn.ndim != 4:
        raise ValueError(f"expected (B, heads, T, T), got {tuple(attn.shape)}")
    a = attn[0]
    if mode == "mean":
        return a.mean(0)
    if mode == "max":
        return a.max(0).values
    if mode == "min":
        return a.min(0).values
    raise ValueError(f"unknown head fusion {mode!r} (expected 'mean', 'max' or 'min')")


def residual_normalize(a: torch.Tensor) -> torch.Tensor:
    """A_hat = 0.5 A + 0.5 I, rows renormalised to sum to 1."""
    eye = torch.eye(a.shape[-1], device=a.device, dtype=a.dtype)
    a = 0.5 * a + 0.5 * eye
    return a / a.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def rollout_from_attentions(attentions: list[torch.Tensor], head_fusion: str = "mean") -> torch.Tensor:
    """Compose layer attentions into the rollout matrix R (T, T)."""
    if not attentions:
        raise RuntimeError("no attention maps captured -- call model.set_store_attention(True) first")
    result = None
    for attn in attentions:  # layer 1 -> L, left-multiplying each time
        a = residual_normalize(fuse_heads(attn, head_fusion))
        result = a if result is None else a @ result
    return result


class AttentionRollout(SaliencyMethod):
    name = "attention_rollout"

    def __init__(self, model, device=None, head_fusion: str = "mean"):
        super().__init__(model, device)
        self.head_fusion = head_fusion

    def _attribute_raw(self, volume: torch.Tensor, target_label: int) -> torch.Tensor:
        self.model.eval()
        self.model.set_store_attention(True)
        try:
            with torch.no_grad():
                self.model(volume)
            rollout = rollout_from_attentions(self.model.get_attention_maps(), self.head_fusion)
        finally:
            self.model.set_store_attention(False)

        # CLS row, minus its own self-attention entry.
        return rollout[0, 1:].detach()


class GradientWeightedRollout(SaliencyMethod):
    """Class-specific rollout: each layer's attention is weighted by ReLU of the
    gradient of the target logit w.r.t. that attention matrix (cf. Chefer et al.)."""

    name = "grad_rollout"

    def __init__(self, model, device=None, head_fusion: str = "mean"):
        super().__init__(model, device)
        self.head_fusion = head_fusion

    def _attribute_raw(self, volume: torch.Tensor, target_label: int) -> torch.Tensor:
        self.model.eval()
        self.model.set_store_attention(True)
        try:
            self.model.zero_grad(set_to_none=True)
            logits = self.model(volume)
            logits[0, target_label].backward()

            weighted = []
            for block in self.model.blocks:
                attn = block.attn.attn_weights
                if attn is None:
                    continue
                grad = attn.grad
                if grad is None:  # no gradient reached this layer
                    weighted.append(attn.detach())
                    continue
                weighted.append((attn * grad).clamp_min(0).detach())

            rollout = rollout_from_attentions(weighted, self.head_fusion)
        finally:
            self.model.set_store_attention(False)
            self.model.zero_grad(set_to_none=True)

        return rollout[0, 1:].detach()
