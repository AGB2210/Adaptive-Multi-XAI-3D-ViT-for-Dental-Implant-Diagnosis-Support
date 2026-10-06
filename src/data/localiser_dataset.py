"""Training samples for the site localiser: one low-resolution scan, fourteen targets.

Targets are the label builder's own site positions (`sites_toothfairy3.csv`),
converted into the localiser's fixed grid with the offset `localiser_input`
recorded when the cache was built. A sample is a PATIENT -- the split, as
everywhere in this project, is by patient and reuses `cv_folds.json`.

WHICH SITES COUNT. A site is "in view" when the arch fit placed it inside the
scan (`reason != "outside_volume"`, x and y finite). Its z is the crest height,
which is NaN where no bone could be measured; such a site still trains x and y,
and only its z is masked. All position tiers (teeth, sparse, opposite_jaw) are
used by default, because the patients with few or no teeth are the ones who
most need implants -- but the weaker tiers are extrapolations, and the
evaluation reports error per tier so that cost is visible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.models.localiser import to_input_coords


def site_targets(sites: pd.DataFrame, patient_id: str, teeth, offset, factor: int,
                 methods=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(coords (S, 3) in grid units, coord mask (S, 3), in-view (S,)) for one patient."""
    rows = sites[(sites.patient_id == patient_id) & (sites.jaw == "lower")].set_index("tooth")
    coords = np.full((len(teeth), 3), np.nan)
    mask = np.zeros((len(teeth), 3), dtype=bool)
    valid = np.zeros(len(teeth), dtype=np.float32)
    for i, tooth in enumerate(teeth):
        if tooth not in rows.index:
            continue
        r = rows.loc[tooth]
        if methods is not None and r.get("site_method") not in methods:
            continue
        xy_ok = np.isfinite(r["site_x"]) and np.isfinite(r["site_y"]) and r.get("reason") != "outside_volume"
        if not xy_ok:
            continue
        valid[i] = 1.0
        full = np.array([r["site_x"], r["site_y"], r["site_z"]], dtype=np.float64)
        coords[i] = to_input_coords(full, offset, factor)
        mask[i, :2] = True
        mask[i, 2] = bool(np.isfinite(full[2]))
    return coords, mask, valid


class LocaliserDataset(Dataset):
    """Cached low-resolution volumes and their site targets, with augmentation.

    Augmentation is geometric only where the targets move with it: a z flip
    (which is ALSO the orientation label -- the network must learn which way up
    a jaw is) and an integer translation. Intensity jitter is the rest.
    """

    def __init__(self, cache_dir, manifest: pd.DataFrame, sites: pd.DataFrame, patients,
                 teeth, factor: int, augment: bool = False, flip_prob: float = 0.5,
                 translate: int = 0, methods=None, seed: int = 0):
        self.cache_dir = Path(cache_dir)
        man = manifest.set_index("patient_id")
        self.patients = [p for p in patients if p in man.index]
        missing = sorted(set(patients) - set(self.patients))
        self.missing = missing
        self.offsets = {p: np.array([man.loc[p, f"offset_{a}"] for a in "xyz"], dtype=np.int64)
                        for p in self.patients}
        self.targets = {p: site_targets(sites, p, teeth, self.offsets[p], factor, methods)
                        for p in self.patients}
        self.augment, self.flip_prob, self.translate = augment, flip_prob, int(translate)
        self.seed = seed

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int):
        pid = self.patients[idx]
        vol = np.load(self.cache_dir / f"{pid}.npy").astype(np.float32)
        coords, mask, valid = (a.copy() for a in self.targets[pid])
        flipped = 0.0
        if self.augment:
            rng = np.random.default_rng((self.seed, idx, int(torch.randint(0, 2**31, (1,)).item())))
            if rng.random() < self.flip_prob:
                vol = vol[:, :, ::-1]
                coords[:, 2] = (vol.shape[2] - 1) - coords[:, 2]
                flipped = 1.0
            if self.translate > 0:
                shift = rng.integers(-self.translate, self.translate + 1, size=3)
                vol = _shift(vol, shift)
                coords = coords + shift
                for a in range(3):
                    # A comparison with NaN is False, so a masked axis stays masked.
                    inside = (coords[:, a] >= 0) & (coords[:, a] <= vol.shape[a] - 1)
                    mask[:, a] &= inside
            vol = vol * (1.0 + rng.uniform(-0.1, 0.1)) + rng.uniform(-0.1, 0.1)
        x = torch.from_numpy(np.ascontiguousarray(vol, dtype=np.float32))[None]
        return (x, torch.from_numpy(np.nan_to_num(coords).astype(np.float32)),
                torch.from_numpy(mask), torch.from_numpy(valid), torch.tensor(flipped))


def _shift(vol: np.ndarray, shift) -> np.ndarray:
    """Translate by whole voxels, filling with the volume's minimum (air). No wraparound."""
    out = np.full_like(vol, vol.min())
    src, dst = [], []
    for axis, s in enumerate(int(v) for v in shift):
        n = vol.shape[axis]
        src.append(slice(max(0, -s), n - max(0, s)))
        dst.append(slice(max(0, s), n - max(0, -s)))
    out[tuple(dst)] = vol[tuple(src)]
    return out
