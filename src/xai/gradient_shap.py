"""GradientSHAP: expected gradients over random baselines and interpolation points.

For each of n samples: draw a baseline, jitter the input with Gaussian noise,
pick a random interpolation point between them, take the gradient there, and
multiply by (x_noisy - baseline). Average over samples.

The variance across samples is reported, not hidden: a high variance means the
estimate has not converged and needs more samples. A saliency map quoted without
knowing its own stability is not evidence.
"""

from __future__ import annotations

import torch

from src.xai.base import SaliencyMethod, gaussian_blur3d


class GradientSHAP(SaliencyMethod):
    name = "gradient_shap"

    def __init__(
        self,
        model,
        device=None,
        n_samples: int = 24,
        noise_std: float = 0.1,
        baselines: torch.Tensor | None = None,
        blur_sigma: float = 4.0,
        batch_size: int = 8,
        absolute: bool = True,
        seed: int | None = 0,
    ):
        super().__init__(model, device)
        self.n_samples = n_samples
        self.noise_std = noise_std
        self.baselines = baselines  # (N, C, D, H, W) drawn from the training set
        self.blur_sigma = blur_sigma
        self.batch_size = batch_size
        self.absolute = absolute
        self.seed = seed
        self.last_variance: float = float("nan")

    def _draw_baselines(self, volume: torch.Tensor, n: int, generator) -> torch.Tensor:
        if self.baselines is not None and len(self.baselines) > 0:
            idx = torch.randint(0, len(self.baselines), (n,), generator=generator, device="cpu")
            return self.baselines[idx].to(volume.device).float()
        # Fall back to blurred copies of the input.
        return gaussian_blur3d(volume, self.blur_sigma).expand(n, -1, -1, -1, -1).clone()

    def _attribute_raw(self, volume: torch.Tensor, target_label: int) -> torch.Tensor:
        self.model.eval()
        # See IntegratedGradients: backprop accumulates into the model's own
        # parameter grads, which must not outlive the call.
        self.model.zero_grad(set_to_none=True)
        generator = None
        if self.seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(self.seed)

        per_sample = []
        for start in range(0, self.n_samples, self.batch_size):
            n = min(self.batch_size, self.n_samples - start)

            baselines = self._draw_baselines(volume, n, generator)
            noise = torch.randn(baselines.shape, generator=generator, device="cpu").to(volume.device)
            x_noisy = volume + self.noise_std * noise

            alpha = torch.rand((n, 1, 1, 1, 1), generator=generator, device="cpu").to(volume.device)
            points = (baselines + alpha * (x_noisy - baselines)).detach().requires_grad_(True)

            logits = self.model(points)
            logits[:, target_label].sum().backward()

            per_sample.append((points.grad * (x_noisy - baselines)).detach())

        self.model.zero_grad(set_to_none=True)

        stacked = torch.cat(per_sample, dim=0)  # (n_samples, C, D, H, W)
        attribution = stacked.mean(dim=0)

        # Stability of the estimate: mean per-voxel standard error, relative to signal.
        if stacked.shape[0] > 1:
            sem = stacked.std(dim=0, unbiased=True) / (stacked.shape[0] ** 0.5)
            denom = attribution.abs().mean().clamp_min(1e-12)
            self.last_variance = float((sem.mean() / denom))

        attribution = attribution.sum(dim=0)  # collapse channels
        return attribution.abs() if self.absolute else attribution
