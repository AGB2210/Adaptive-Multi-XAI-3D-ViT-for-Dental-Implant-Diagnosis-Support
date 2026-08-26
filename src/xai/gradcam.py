"""Grad-CAM adapted to a 3D ViT.

Classic Grad-CAM pools gradients over spatial positions of a conv feature map.
The ViT analogue: hook the last transformer block's output tokens (B, T, D),
average the gradients over the token dimension to get one weight per channel,
take the weighted channel sum, ReLU, and reshape the patch tokens to the grid.

The CLS token is excluded from the spatial reshape -- it has no location.
"""

from __future__ import annotations

import torch

from src.xai.base import SaliencyMethod


class GradCAM3D(SaliencyMethod):
    name = "gradcam"

    def __init__(self, model, device=None, layer_index: int = -1):
        super().__init__(model, device)
        self.layer_index = layer_index

    def _attribute_raw(self, volume: torch.Tensor, target_label: int) -> torch.Tensor:
        self.model.eval()

        if hasattr(self.model, "blocks"):  # ViT path
            return self._vit_cam(volume, target_label)
        return self._conv_cam(volume, target_label)

    def _vit_cam(self, volume: torch.Tensor, target_label: int) -> torch.Tensor:
        block = self.model.blocks[self.layer_index]
        activations, grads = {}, {}

        h_fwd = block.register_forward_hook(lambda m, i, o: activations.__setitem__("v", o))
        h_bwd = block.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("v", go[0]))
        try:
            self.model.zero_grad(set_to_none=True)
            logits = self.model(volume)
            logits[0, target_label].backward()
        finally:
            h_fwd.remove()
            h_bwd.remove()
            self.model.zero_grad(set_to_none=True)

        if "v" not in activations or "v" not in grads:
            raise RuntimeError("Grad-CAM hooks captured nothing -- model structure changed?")

        acts = activations["v"].detach()[0]  # (T, D)
        grad = grads["v"].detach()[0]  # (T, D)

        # One weight per channel: gradients pooled over tokens.
        weights = grad.mean(dim=0)  # (D,)
        cam = (acts * weights).sum(dim=-1).clamp_min(0)  # (T,)
        return cam[1:]  # drop CLS

    def _conv_cam(self, volume: torch.Tensor, target_label: int) -> torch.Tensor:
        """Fallback for the CNN baseline: standard Grad-CAM on its feature map."""
        self.model.cache_activations = True
        try:
            self.model.zero_grad(set_to_none=True)
            logits = self.model(volume)
            logits[0, target_label].backward()
            feats = self.model.features
            if feats is None or feats.grad is None:
                raise RuntimeError("CNN Grad-CAM: no cached features or gradients")
            weights = feats.grad.detach()[0].mean(dim=(1, 2, 3))  # (C,)
            cam = (feats.detach()[0] * weights[:, None, None, None]).sum(0).clamp_min(0)
        finally:
            self.model.cache_activations = False
            self.model.zero_grad(set_to_none=True)
        return cam
