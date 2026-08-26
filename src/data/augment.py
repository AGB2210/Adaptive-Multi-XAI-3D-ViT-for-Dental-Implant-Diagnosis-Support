"""Train-time augmentation for 3D CBCT volumes.

Arrays are RAS+ with axes (x=L->R, y=P->A, z=I->S), so only the sagittal mirror
(axis 0) is anatomically sane. Flipping anterior-posterior or superior-inferior
would produce anatomy that cannot occur in a patient, and would teach the model
that "upside-down mandible" is a normal presentation.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


class Augment3D:
    """Deterministic given a seeded RNG, so runs stay reproducible."""

    def __init__(self, cfg):
        self.translate_voxels = int(getattr(cfg, "translate_voxels", 0))
        self.translate_prob = float(getattr(cfg, "translate_prob", 0.0))
        self.flip_lr_prob = cfg.flip_lr_prob
        self.rotate_deg = cfg.rotate_deg
        self.rotate_prob = cfg.rotate_prob
        self.intensity_shift = cfg.intensity_shift
        self.intensity_scale = cfg.intensity_scale
        self.gamma_range = tuple(cfg.gamma_range)
        self.gamma_prob = cfg.gamma_prob
        self.noise_std = cfg.noise_std
        self.noise_prob = cfg.noise_prob

    def __call__(self, vol: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        vol = np.asarray(vol, dtype=np.float32)

        # --- geometry ---
        if self.translate_voxels > 0 and rng.random() < self.translate_prob:
            # The two cohorts differ sharply in how much of the box the anatomy
            # fills (measured on the earlier MMDental/ToothFairy4 pair: 41% vs 76%
            # consequence of their different fields of view). Without random
            # translation the model can key on absolute position, which would not
            # survive a zero-shot transfer to another cohort.
            vol = self._translate(vol, rng)

        if rng.random() < self.flip_lr_prob:
            vol = np.ascontiguousarray(vol[::-1])

        if self.rotate_deg > 0 and rng.random() < self.rotate_prob:
            # Small rotation in one randomly chosen plane.
            axes = [(0, 1), (0, 2), (1, 2)][rng.integers(3)]
            angle = float(rng.uniform(-self.rotate_deg, self.rotate_deg))
            vol = ndimage.rotate(
                vol, angle, axes=axes, reshape=False, order=1, mode="nearest", prefilter=False
            )

        # --- intensity ---
        if self.intensity_scale > 0 or self.intensity_shift > 0:
            scale = 1.0 + float(rng.uniform(-self.intensity_scale, self.intensity_scale))
            shift = float(rng.uniform(-self.intensity_shift, self.intensity_shift))
            vol = vol * scale + shift

        if rng.random() < self.gamma_prob:
            gamma = float(rng.uniform(*self.gamma_range))
            lo, hi = float(vol.min()), float(vol.max())
            if hi > lo:  # gamma needs a [0, 1] domain; map back afterwards
                unit = (vol - lo) / (hi - lo)
                vol = np.power(unit, gamma) * (hi - lo) + lo

        if self.noise_std > 0 and rng.random() < self.noise_prob:
            vol = vol + rng.normal(0.0, self.noise_std, size=vol.shape).astype(np.float32)

        return vol.astype(np.float32)

    def _translate(self, vol: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Shift the volume by a few voxels, filling the vacated edge with air.

        Shift, not roll: wrapping would teleport the mandible to the other side.
        """
        limit = self.translate_voxels
        shifts = rng.integers(-limit, limit + 1, size=3)
        if not shifts.any():
            return vol

        out = np.full_like(vol, vol.min())
        src, dst = [], []
        for axis, shift in enumerate(shifts):
            n = vol.shape[axis]
            s = int(shift)
            src.append(slice(max(0, -s), n - max(0, s)))
            dst.append(slice(max(0, s), n - max(0, -s)))
        out[tuple(dst)] = vol[tuple(src)]
        return out
