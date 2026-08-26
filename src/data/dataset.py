"""Dataset over the preprocessed cache. Training never touches the raw NIfTI files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CachedVolumeDataset(Dataset):
    """Reads the float16 .npy volumes written by scripts/build_cache.py.

    `preload=True` holds the whole cache in RAM and removes disk I/O from the
    training loop. Budget for it: one 128^3 float16 volume is 4.2 MB, one 256^3
    volume is 33.6 MB.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        patient_ids: list[str],
        labels: np.ndarray,
        augment=None,
        preload: bool = False,
        seed: int = 0,
    ):
        if len(patient_ids) != len(labels):
            raise ValueError(f"{len(patient_ids)} ids vs {len(labels)} label rows")
        self.cache_dir = Path(cache_dir)
        self.patient_ids = list(patient_ids)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.augment = augment
        self.seed = seed

        missing = [p for p in self.patient_ids if not (self.cache_dir / f"{p}.npy").exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} cached volumes missing under {self.cache_dir} "
                f"(first few: {missing[:5]}). Run scripts/build_cache.py."
            )

        self.volumes: list[np.ndarray] | None = None
        if preload:
            self.volumes = [np.load(self.cache_dir / f"{p}.npy") for p in self.patient_ids]

    def __len__(self) -> int:
        return len(self.patient_ids)

    def _load(self, idx: int) -> np.ndarray:
        if self.volumes is not None:
            return self.volumes[idx]
        return np.load(self.cache_dir / f"{self.patient_ids[idx]}.npy")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        vol = self._load(idx).astype(np.float32)

        if self.augment is not None:
            # Seed per (epoch-agnostic) sample draw so workers stay reproducible
            # yet do not repeat the same augmentation every epoch.
            rng = np.random.default_rng((self.seed, idx, torch.randint(0, 2**31, (1,)).item()))
            vol = self.augment(vol, rng)

        x = torch.from_numpy(np.ascontiguousarray(vol)).unsqueeze(0)  # (1, D, H, W)
        y = torch.from_numpy(self.labels[idx])
        return x, y


def load_label_matrix(
    labels_csv: str | Path, label_names: list[str]
) -> tuple[list[str], np.ndarray]:
    """Read a labels_*.csv written by scripts/build_labels.py."""
    if not label_names:
        raise ValueError("label_names is required: the label set belongs to the dataset, "
                         "not to this module, so it is always passed in explicitly")
    names = list(label_names)
    df = pd.read_csv(labels_csv, dtype={"patient_id": str})
    missing = [c for c in names if c not in df.columns]
    if missing:
        raise ValueError(f"{labels_csv} lacks label columns {missing}")
    return df["patient_id"].astype(str).tolist(), df[names].to_numpy(dtype=np.float32)


def restrict_to_cache(
    patient_ids: list[str], y: np.ndarray, cache_dir: str | Path
) -> tuple[list[str], np.ndarray]:
    """Keep only patients whose volume actually made it through preprocessing."""
    cache_dir = Path(cache_dir)
    keep = [i for i, p in enumerate(patient_ids) if (cache_dir / f"{p}.npy").exists()]
    return [patient_ids[i] for i in keep], y[keep]


def pos_weight_from_labels(y: np.ndarray, clamp: float = 10.0) -> torch.Tensor:
    """BCEWithLogitsLoss pos_weight = (#neg / #pos) per label, clamped.

    Clamping stops a near-empty label from dominating the gradient.
    """
    y = np.asarray(y, dtype=np.float64)
    pos = y.sum(axis=0)
    neg = len(y) - pos
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(pos > 0, neg / np.maximum(pos, 1.0), 1.0)
    return torch.tensor(np.clip(w, 1.0 / clamp, clamp), dtype=torch.float32)
