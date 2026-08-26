"""Shared setup for the XAI scripts: verify the training outputs, load the model
and the cached cases.

The XAI stage reuses the training partition exactly and never re-splits, so every
number stays comparable to the training results. That partition is either a
single train/val/test split (splits.json) or a cross-validation fold assignment
(cv_folds.json); pass fold=k for the latter, and the held-out cases of round k
are the ones its checkpoint never saw.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch

from src.data.dataset import load_label_matrix, restrict_to_cache
from src.data.splits import check_disjoint, fold_assignment, load_folds, load_splits
from src.data.taskdef import external_dataset, label_names_for, primary_dataset
from src.models import build_model
from src.train.loop import load_checkpoint_file
from src.utils.config import artifacts_dir
from src.utils.config import cache_dir as cache_dir_for


def fold_from_checkpoint(checkpoint: str | Path | None) -> int | None:
    """Recover the CV round from a path like artifacts/runs/cv_fold3/best.pt.

    Explaining a model on cases it was trained on is meaningless -- the map would
    describe memorisation. The fold is therefore taken from the checkpoint itself
    rather than trusted to a flag, and a contradicting --fold is an error.
    """
    if checkpoint is None:
        return None
    match = re.search(r"cv_fold(\d+)", Path(checkpoint).as_posix())
    return int(match.group(1)) if match else None


def resolve_fold(checkpoint: str | Path | None, fold: int | None) -> int | None:
    inferred = fold_from_checkpoint(checkpoint)
    if inferred is not None and fold is not None and inferred != fold:
        raise ValueError(
            f"--fold {fold} contradicts the checkpoint path, which is round {inferred}. "
            f"Round {inferred}'s held-out cases are the only ones that checkpoint never "
            f"saw; explaining it on fold {fold} would describe training data."
        )
    return inferred if inferred is not None else fold


def require_prerequisites(cfg, checkpoint: str | Path | None = None, need_external: bool = False,
                          fold: int | None = None) -> None:
    """Fail loudly, naming what is missing, before any XAI work starts."""
    art = artifacts_dir(cfg)
    problems = []
    primary = primary_dataset(cfg)

    partition = "cv_folds.json" if fold is not None else "splits.json"
    needed = [f"labels_{primary}.csv", partition]
    external = external_dataset(cfg) if need_external else None
    if external:
        needed.append(f"labels_{external}.csv")
    for name in needed:
        if (art / name).is_file():
            continue
        if name.endswith(".json"):
            other = "cv_folds.json" if name == "splits.json" else "splits.json"
            hint = (f" (found {other} instead -- pass --fold K to use the "
                    f"cross-validation partition)") if (art / other).is_file() else ""
            problems.append(f"missing {art / name} -- run scripts/train.py{hint}")
        else:
            problems.append(f"missing {art / name} -- run scripts/build_labels.py")

    for name in [primary] + ([external] if external else []):
        cache = cache_dir_for(cfg, name)
        if not cache.is_dir() or not any(cache.glob("*.npy")):
            problems.append(f"empty cache for {name} at {cache} "
                            f"-- run scripts/build_cache.py --dataset {name}")

    if checkpoint is not None and not Path(checkpoint).is_file():
        problems.append(f"no trained checkpoint at {checkpoint} -- run scripts/train.py")

    if problems:
        raise FileNotFoundError("prerequisites not met:\n  - " + "\n  - ".join(problems))


def load_model(cfg, checkpoint: str | Path, device: torch.device):
    model = build_model(cfg.model, img_size=cfg.preprocess.out_shape[0])
    ckpt = load_checkpoint_file(checkpoint, device)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    # Attribution needs gradients to flow THROUGH the network to the input, which
    # requires the input to have requires_grad -- not the parameters. Leaving the
    # parameters trainable made every backward pass compute and accumulate weight
    # gradients for all 9.15M of them, work no method uses: IG and GradientSHAP
    # read the input's .grad, Grad-CAM and gradient-rollout read hooked
    # activations. SaliencyMethod._check_input supplies the graph instead.
    for param in model.parameters():
        param.requires_grad_(False)
    return model, ckpt


def describe_token_geometry(model) -> dict:
    """Derive the token grid from the real model rather than assuming 8x8x8."""
    grid = tuple(getattr(model, "grid_size", ()))
    n_patches = int(np.prod(grid)) if grid else 0
    return {
        "grid_size": grid,
        "n_patch_tokens": n_patches,
        "sequence_length": n_patches + 1 if n_patches else 0,
        "depth": len(model.blocks) if hasattr(model, "blocks") else 0,
    }


def load_cases(cfg, split: str = "test", dataset: str | None = None, fold: int | None = None):
    """Return (patient_ids, labels, cache_dir, label_names) for a split.

    The primary cohort is split; an external cohort is used whole and never
    split, because it is scored once and nothing is selected on it.
    """
    art = artifacts_dir(cfg)
    dataset = dataset or primary_dataset(cfg)
    cache = cache_dir_for(cfg, dataset)
    labels = label_names_for(cfg)

    ids, y = load_label_matrix(art / f"labels_{dataset}.csv", labels)
    ids, y = restrict_to_cache(ids, y, cache)

    if dataset != primary_dataset(cfg):
        return ids, y, cache, labels

    if fold is not None:
        folds = load_folds(art / "cv_folds.json")
        splits = fold_assignment(folds, fold)
    else:
        splits = load_splits(art / "splits.json")
    check_disjoint(splits)
    index = {p: i for i, p in enumerate(ids)}
    chosen = [p for p in splits[split] if p in index]
    return chosen, y[[index[p] for p in chosen]], cache, labels


def load_volume(cache_dir: Path, patient_id: str, device: torch.device) -> torch.Tensor:
    """One cached volume as (1, 1, D, H, W) float32."""
    arr = np.load(Path(cache_dir) / f"{patient_id}.npy").astype(np.float32)
    return torch.from_numpy(arr)[None, None].to(device)


def training_baselines(cfg, n: int = 16, device: torch.device | None = None, seed: int = 0,
                       fold: int | None = None):
    """A sample of training volumes for GradientSHAP, plus the training mean volume.

    Drawn from TRAIN only -- using val or test volumes as baselines would leak.
    Under cross-validation "train" means round `fold`'s three training folds, so
    the fold must be threaded through here too: baselines built from the round's
    own held-out cases would put test data inside the explanation of test data.
    """
    ids, _, cache, _ = load_cases(cfg, "train", fold=fold)
    if not ids:
        return None, None

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(ids), size=min(n, len(ids)), replace=False)
    volumes = np.stack([np.load(Path(cache) / f"{ids[i]}.npy").astype(np.float32) for i in chosen])

    baselines = torch.from_numpy(volumes)[:, None]
    mean_volume = torch.from_numpy(volumes.mean(axis=0))[None, None]
    if device is not None:
        baselines, mean_volume = baselines.to(device), mean_volume.to(device)
    return baselines, mean_volume


@torch.no_grad()
def predict_logits(model, cache_dir, patient_ids, device, batch_size: int = 8) -> np.ndarray:
    out = []
    for start in range(0, len(patient_ids), batch_size):
        batch = torch.cat([
            load_volume(cache_dir, pid, device) for pid in patient_ids[start : start + batch_size]
        ])
        out.append(model(batch).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 0))
