"""3D LIME with supervoxel occlusion -- ABLATION ONLY, off by default.

Included to justify its exclusion from the main ensemble, not to compete in it.
Expect it to be slow (hundreds of forward passes per map) and artefact-prone: the
explanation is a linear surrogate over a supervoxel partition, so its resolution
is capped by the partition and occlusion introduces intensities the model never
saw in training.

Supervoxels here are a regular grid, not SLIC: no scikit-image dependency, and a
regular partition keeps the surrogate's design matrix balanced.
"""

from __future__ import annotations

import torch

from src.xai.base import SaliencyMethod, make_baseline


class Lime3D(SaliencyMethod):
    name = "lime3d"

    def __init__(
        self,
        model,
        device=None,
        grid: tuple[int, int, int] = (8, 8, 8),
        n_samples: int = 256,
        keep_prob: float = 0.5,
        baseline: str = "blur",
        blur_sigma: float = 4.0,
        batch_size: int = 8,
        ridge_alpha: float = 1.0,
        seed: int | None = 0,
    ):
        super().__init__(model, device)
        self.grid = grid
        self.n_samples = n_samples
        self.keep_prob = keep_prob
        self.baseline_kind = baseline
        self.blur_sigma = blur_sigma
        self.batch_size = batch_size
        self.ridge_alpha = ridge_alpha
        self.seed = seed

    @torch.no_grad()
    def _attribute_raw(self, volume: torch.Tensor, target_label: int) -> torch.Tensor:
        self.model.eval()
        baseline = make_baseline(volume, self.baseline_kind, sigma=self.blur_sigma)

        gz, gy, gx = self.grid
        n_super = gz * gy * gx
        generator = torch.Generator(device="cpu").manual_seed(self.seed) if self.seed is not None else None

        masks = (torch.rand((self.n_samples, n_super), generator=generator) < self.keep_prob).float()
        masks[0] = 1.0  # always include the unperturbed input

        responses = []
        for start in range(0, self.n_samples, self.batch_size):
            chunk = masks[start : start + self.batch_size].to(volume.device)
            n = chunk.shape[0]

            # Supervoxel mask -> voxel mask by nearest-neighbour upsampling.
            m = chunk.reshape(n, 1, gz, gy, gx)
            m = torch.nn.functional.interpolate(m, size=tuple(volume.shape[2:]), mode="nearest")

            perturbed = volume * m + baseline * (1 - m)
            responses.append(torch.sigmoid(self.model(perturbed)[:, target_label]).cpu())

        y = torch.cat(responses).double()
        X = masks.double()

        # Ridge regression: w = (X'X + aI)^-1 X'y, on centred data.
        Xc = X - X.mean(0, keepdim=True)
        yc = y - y.mean()
        gram = Xc.T @ Xc + self.ridge_alpha * torch.eye(n_super, dtype=torch.float64)
        weights = torch.linalg.solve(gram, Xc.T @ yc).float()

        # Reshaped to LIME's own supervoxel grid, not returned as token space:
        # the partition is independent of the model's patch grid, and returning
        # a flat vector would make any grid except the model's raise.
        return weights.to(volume.device).clamp_min(0).reshape(*self.grid)
