"""Grad-CAM adapted to a 3D ViT.

Classic Grad-CAM pools gradients over spatial positions of a conv feature map.
The ViT analogue: take one transformer block's tokens (B, T, D), pool their
gradients to get one weight per channel, take the weighted channel sum, ReLU,
and reshape the patch tokens to the grid. The CLS token is excluded from the
spatial reshape -- it has no location.

WHICH TOKENS, AND WHY IT MATTERS MORE THAN IT LOOKS. This model classifies from
the CLS token alone: `head(norm(x)[:, 0])`. So at the OUTPUT of the last block
the gradient of any output is exactly zero on every patch token -- measured, not
assumed: |grad| sums to 0.0 over the patch tokens and the output does not move
when all of them are replaced by noise. Nothing downstream reads them.

`tokens="output"`, the default, hooks exactly there. Its channel weights are
therefore the CLS token's gradient divided by the token count and nothing else,
and the map is

    ReLU( patch-token activation . CLS gradient )

over activations the prediction does not depend on. That is a similarity map --
which patches look like what the readout wants -- and not gradient-weighted
class activation in Selvaraju et al.'s sense, where the pooled gradients belong
to the spatial features themselves. EVERY `gradcam` FIGURE THIS PROJECT HAS
RECORDED WAS MEASURED ON THIS MAP, which is why it remains the default: changing
it changes what those numbers mean, and that is a decision for a release that
re-runs them, not for a bug fix.

`tokens="input"` is the usual remedy for a CLS-pooled ViT: the tokens ENTERING
the block, which the block's attention still mixes into CLS, so their gradients
are real. Weights are pooled over the patch tokens only. It is registered as
its own method, `gradcam_input`, so `--methods ... gradcam gradcam_input` scores
the two side by side and neither overwrites the other's column.

MEASURED WHERE THE ANSWER IS KNOWN, by `scripts/compare_gradcam_tokens.py`.
Eight ViTs trained on the planted-signal task until validation macro AUROC
passed 0.97, each scored on 60 unseen cases. Enrichment on the planted region,
where 1.0 is chance, as the median per model; deletion AUC on the class
probability, where lower is better, as the mean per model:

                         median enrichment      deletion AUC    empty maps
    gradcam              0.20 - 1.39            0.44 - 0.71      0 - 22%
    gradcam_input        0.94 - 5.00            0.25 - 0.50      0 - 32%
    attention_rollout    3.94 - 8.19            0.17 - 0.38      none
    integrated_gradients 40.1 - 58.3            0.12 - 0.34      none

The default sits on chance, on a signal the model demonstrably reads: under
1.05 in six of the eight models and never above 1.4, with the highest deletion
AUC of the four methods in every model. The block-input map is above chance in
seven of eight and has the lower deletion AUC of the two in all eight. It is
not a rescue: it beats rollout in none of the eight, and because Grad-CAM ends
in a ReLU it returns no map at all -- zero everywhere -- in up to a third of cases.

One earlier run disagrees and is not hidden: fold 0's `xai_sanity.csv` has the
default at a median of 3.45 over 8 cases, on one model. Sixty times the cases,
over eight models, is the stronger evidence, and the mechanism above does not
depend on either.
"""

from __future__ import annotations

import torch

from src.xai.base import SaliencyMethod

TOKENS = ("output", "input")


class GradCAM3D(SaliencyMethod):
    name = "gradcam"

    def __init__(self, model, device=None, layer_index: int = -1, tokens: str = "output"):
        super().__init__(model, device)
        if tokens not in TOKENS:
            raise ValueError(f"tokens must be one of {TOKENS}, got {tokens!r}")
        self.layer_index = layer_index
        self.tokens = tokens
        # Two maps must not share a column name in a results file.
        self.name = "gradcam" if tokens == "output" else "gradcam_input"

    def _attribute_raw(self, volume: torch.Tensor, target_label: int) -> torch.Tensor:
        self.model.eval()

        if hasattr(self.model, "blocks"):  # ViT path
            if self.tokens == "input":
                return self._vit_cam_block_input(volume, target_label)
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

        # One weight per channel: gradients pooled over tokens. On the last
        # block of a CLS-pooled model only the CLS row of `grad` is non-zero --
        # see the module docstring.
        weights = grad.mean(dim=0)  # (D,)
        cam = (acts * weights).sum(dim=-1).clamp_min(0)  # (T,)
        return cam[1:]  # drop CLS

    def _vit_cam_block_input(self, volume: torch.Tensor, target_label: int) -> torch.Tensor:
        """Grad-CAM on the tokens entering the block, pooled over patch tokens."""
        block = self.model.blocks[self.layer_index]
        captured = {}

        def keep(_module, args):
            args[0].retain_grad()
            captured["v"] = args[0]

        handle = block.register_forward_pre_hook(keep)
        try:
            self.model.zero_grad(set_to_none=True)
            logits = self.model(volume)
            logits[0, target_label].backward()
        finally:
            handle.remove()
            self.model.zero_grad(set_to_none=True)

        tokens = captured.get("v")
        if tokens is None or tokens.grad is None:
            raise RuntimeError("Grad-CAM captured no tokens or no gradient -- model structure changed?")

        acts = tokens.detach()[0, 1:]  # (T - 1, D), CLS dropped
        weights = tokens.grad.detach()[0, 1:].mean(dim=0)  # (D,)
        return (acts * weights).sum(dim=-1).clamp_min(0)

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
