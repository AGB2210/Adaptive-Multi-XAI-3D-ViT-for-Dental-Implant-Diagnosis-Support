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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.dataset import load_label_matrix, restrict_to_cache
from src.data.site_dataset import (
    cut_patch,
    load_sites,
    patch_centre,
    restrict_sites_to_cache,
    target_matrix,
)
from src.data.splits import check_disjoint, fold_assignment, load_folds, load_splits
from src.data.taskdef import (
    all_target_names,
    external_dataset,
    label_names_for,
    primary_dataset,
)
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
    sites_csv = getattr(cfg.task, "sites_csv", None)
    label_file = sites_csv if sites_csv else f"labels_{primary}.csv"
    builder = ("scripts/build_implant_labels.py" if sites_csv
               else "scripts/build_labels.py")

    needed = [label_file, partition]
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
            problems.append(f"missing {art / name} -- run {builder}")

    for name in [primary] + ([external] if external else []):
        cache = cache_dir_for(cfg, name)
        if not cache.is_dir() or not any(cache.glob("*.npy")):
            cache_builder = ("scripts/build_site_cache.py" if sites_csv
                             else f"scripts/build_cache.py --dataset {name}")
            problems.append(f"empty cache for {name} at {cache} -- run {cache_builder}")

    if checkpoint is not None and not Path(checkpoint).is_file():
        problems.append(f"no trained checkpoint at {checkpoint} -- run scripts/train.py")

    if problems:
        raise FileNotFoundError("prerequisites not met:\n  - " + "\n  - ".join(problems))


def model_img_size(cfg) -> int:
    """Input edge length, from the model config first.

    The site task sets model.img_size and leaves preprocess.out_shape null,
    because nothing is resampled onto a fixed grid -- patches are cut at the
    scan's own resolution. Reading out_shape[0] unconditionally raises a
    TypeError on None, which is a confusing way to be told the config is for a
    different pipeline.
    """
    size = getattr(cfg.model, "img_size", None)
    if size is not None:
        return int(size)
    shape = getattr(cfg.preprocess, "out_shape", None)
    if not shape:
        raise SystemExit(
            "cannot determine the model input size: set model.img_size, or "
            "preprocess.out_shape for the fixed-grid pipeline."
        )
    return int(shape[0])


def load_model(cfg, checkpoint: str | Path, device: torch.device):
    model = build_model(cfg.model, img_size=model_img_size(cfg))
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


SITE_SEP = "#"


@dataclass
class CaseSet:
    """The cases an XAI script explains, whichever task produced them.

    The XAI stage was written when a case meant a whole scan. The site task makes
    a case a (patient, tooth position) pair whose input is a small patch cut from
    that patient's volume. Rather than teach five scripts the difference, both
    pipelines produce one of these and the scripts ask it for an input.

    `ids` are display identifiers: a patient id for the whole-volume task, and
    "<patient>#<tooth>" for the site task, so a figure or a CSV row still names
    exactly what was explained.
    """

    ids: list
    y: np.ndarray
    cache: Path
    labels: list
    sites: pd.DataFrame | None = None
    patch_size: int | None = None

    def __post_init__(self):
        self._volumes: dict = {}
        self._rows = {}
        if self.sites is not None:
            self._rows = {i: row for i, row in zip(self.ids, self.sites.to_dict("records"))}

    @property
    def is_sites(self) -> bool:
        return self.sites is not None

    def patient_of(self, case_id: str) -> str:
        return case_id.split(SITE_SEP)[0]

    def row(self, case_id: str):
        """The site row behind a case, or None on the whole-volume task."""
        return self._rows.get(case_id)

    def _volume(self, patient_id: str) -> np.ndarray:
        cached = self._volumes.get(patient_id)
        if cached is None:
            # Memory-mapped: a native-resolution cache is ~50 GB and a patch
            # read should touch only the pages it needs.
            cached = np.load(Path(self.cache) / f"{patient_id}.npy", mmap_mode="r")
            self._volumes[patient_id] = cached
        return cached

    def load(self, case_id: str, device: torch.device) -> torch.Tensor:
        """The model input for one case, as (1, 1, D, H, W) float32."""
        if not self.is_sites:
            return load_volume(self.cache, case_id, device)

        row = self._rows.get(case_id)
        if row is None:
            raise KeyError(f"{case_id!r} is not in this case set")
        volume = self._volume(row["patient_id"])
        centre = patch_centre(row, volume.shape, int(self.patch_size))
        patch = cut_patch(volume, centre, int(self.patch_size))
        arr = np.asarray(patch, dtype=np.float32)
        return torch.from_numpy(np.ascontiguousarray(arr))[None, None].to(device)


def site_case_ids(sites: pd.DataFrame) -> list:
    return [f"{r.patient_id}{SITE_SEP}{int(r.tooth)}" for r in sites.itertuples()]


def load_case_set(cfg, split: str = "test", dataset: str | None = None,
                  fold: int | None = None) -> CaseSet:
    """Cases for a split, from whichever task the config describes.

    The partition is reused exactly as training made it and is never re-split.
    For the site task the split file holds PATIENT ids -- 28 sites from one scan
    share anatomy and annotator, so they move between folds together -- and the
    site rows are selected by patient membership.
    """
    art = artifacts_dir(cfg)
    dataset = dataset or primary_dataset(cfg)
    cache = cache_dir_for(cfg, dataset)
    # EVERY target, binary and millimetre. `cases.labels` indexes the model's
    # output columns, so a binary-only list makes label_names[1] an IndexError
    # the moment an explanation targets a millimetre head -- and the silent
    # version, on a task where the lengths happen to match, labels one head's
    # attribution with another head's name.
    labels = all_target_names(cfg)
    sites_csv = getattr(cfg.task, "sites_csv", None)

    if not sites_csv:
        ids, y, cache, labels = load_cases(cfg, split, dataset, fold=fold)
        return CaseSet(ids=list(ids), y=y, cache=cache, labels=labels)

    methods = tuple(getattr(cfg.task, "site_methods", ("teeth",)))
    jaws = tuple(getattr(cfg.task, "site_jaws", ("lower",)))
    sites = load_sites(art / sites_csv, targets=labels, methods=methods, jaws=jaws)
    sites = restrict_sites_to_cache(sites, cache)

    if dataset == primary_dataset(cfg):
        if fold is not None:
            splits = fold_assignment(load_folds(art / "cv_folds.json"), fold)
        else:
            splits = load_splits(art / "splits.json")
        check_disjoint(splits)
        sites = sites[sites.patient_id.isin(set(splits[split]))].reset_index(drop=True)

    return CaseSet(
        ids=site_case_ids(sites),
        y=target_matrix(sites, labels),
        cache=cache,
        labels=labels,
        sites=sites,
        patch_size=model_img_size(cfg),
    )


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
    train = load_case_set(cfg, "train", fold=fold)
    if not train.ids:
        return None, None

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(train.ids), size=min(n, len(train.ids)), replace=False)
    # On the site task a baseline is a training PATCH, not a whole scan: it has
    # to be the same shape as the input being explained, and it has to come from
    # a site the checkpoint was trained on.
    cpu = torch.device("cpu")
    stack = torch.cat([train.load(train.ids[i], cpu) for i in chosen])  # (n, 1, D, H, W)

    baselines = stack
    mean_volume = stack.mean(dim=0, keepdim=True)
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


@torch.no_grad()
def predict_case_logits(model, cases: CaseSet, device, batch_size: int = 8) -> np.ndarray:
    """Logits for every case, on either task."""
    out = []
    for start in range(0, len(cases.ids), batch_size):
        batch = torch.cat([cases.load(i, device)
                           for i in cases.ids[start : start + batch_size]])
        out.append(model(batch).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 0))


def xai_setting(cfg, key: str, cli_value, fallback):
    """Sample size for an XAI run: CLI flag, else `xai:` in the config, else a
    small fallback that finishes on a laptop.

    The config block existed for a whole release and NOTHING READ IT. Every
    script used its own argparse default, so `configs/sites.yaml` asking for 200
    GradientSHAP samples and 30 randomisation cases silently delivered 24 and 3.
    24 is the setting whose relative standard error was measured at 0.5689 --
    the estimate had not converged and its numbers were not quotable -- and the
    config comment says exactly that, three lines above the value being ignored.

    A config value that is never read is worse than no config value: it reads as
    a decision that was made and honoured.
    """
    if cli_value is not None:
        return cli_value
    block = getattr(cfg, "xai", None)
    if block is not None:
        value = getattr(block, key, None)
        if value is not None:
            return value
    return fallback


def select_cases(ids, y, n: int, seed: int, log=None):
    """Take `n` cases at random, with a fixed seed, and say how many patients.

    Head-truncation (`ids[:n]`) is not a sample. The case list comes out of the
    CSV in (patient, tooth) order and a patient contributes ~14 mandibular
    sites, so the default n=20 was TWO PATIENTS -- one jaw, one field of view,
    one scanner session, 20 times over. Any interval computed across them
    measures within-patient repeatability and reports it as between-patient
    uncertainty.

    Sampling fixes which cases are drawn. It does not make them independent:
    several sites still come from one jaw, so anything bootstrapped over the
    result must resample PATIENTS, not rows. `patients_of` is what the callers
    pass to the bootstrap for that.
    """
    n = min(int(n), len(ids))
    idx = np.random.default_rng(seed).choice(len(ids), size=n, replace=False)
    idx.sort()                        # keep CSV order for readable output
    ids = [ids[i] for i in idx]
    y = y[idx]
    if log is not None:
        log.info("%d cases across %d patients", len(ids), len(set(patients_of(ids))))
    return ids, y


def patients_of(ids) -> list[str]:
    """Patient for each case id, for clustering a bootstrap by patient."""
    return [str(i).split(SITE_SEP)[0] for i in ids]


def clustered_ci(frame, value_col: str, patient_col: str = "patient_id",
                 stat=np.median, n_boot: int = 4000, ci: float = 0.95,
                 seed: int = 1337):
    """(point estimate, lo, hi) resampling PATIENTS, not rows.

    RESAMPLING ROWS IS WRONG HERE AND THE ERROR IS INVISIBLE. A patient
    contributes up to fourteen mandibular sites that share anatomy, field of
    view, scanner and annotator. Treating them as independent draws narrows the
    interval by roughly the square root of the sites-per-patient ratio, so a
    result reported that way looks more certain than the data supports and
    nothing in the output says so.

    This is the same fault as `ids[:n]` head-truncation, one level up: there the
    sample was not a sample, here the resample is not a resample.

    Returns nan when nothing is finite -- callers must not turn that into a zero.
    """
    groups = [g[value_col].to_numpy(dtype=float) for _, g in frame.groupby(patient_col)]
    groups = [g[np.isfinite(g)] for g in groups]
    groups = [g for g in groups if g.size]
    if not groups:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    k = len(groups)
    point = float(stat(np.concatenate(groups)))
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        pick = rng.integers(0, k, k)
        draws[b] = stat(np.concatenate([groups[i] for i in pick]))
    lo, hi = np.percentile(draws, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
    return point, float(lo), float(hi)


def ci_table(frame, value_col: str, group_col: str = "method", **kw):
    """`clustered_ci` per method, as a frame ready to print beside the medians."""
    rows = []
    for name, g in frame.groupby(group_col):
        point, lo, hi = clustered_ci(g, value_col, **kw)
        rows.append({group_col: name, value_col: point, "ci_lo": lo, "ci_hi": hi,
                     "patients": int(g["patient_id"].nunique())})
    return pd.DataFrame(rows).set_index(group_col).sort_values(value_col)


def explanation_target(cfg, spec=None, outputs=None) -> int:
    """Which output column to explain.

    For a pure-classification task, the most confident label -- which is what
    every script did, and remains right there.

    For the hybrid site task, the FIRST MILLIMETRE TARGET, by default
    `available_height_mm`. Two reasons, and the second is the important one.

    Mechanically, `argmax(sigmoid(out))` is meaningless once some columns are
    millimetres: a predicted 18 mm sigmoids to 1.0 and wins every comparison, so
    the "most confident label" is whichever regression head happens to have the
    largest raw value.

    Scientifically, this is the target worth explaining. Available height in the
    mandible is crest-to-canal distance, so a model predicting it correctly MUST
    have located both the crest and the inferior alveolar canal -- and the canal
    is annotated in every scan. That makes "did the explanation point at the
    right anatomy?" answerable against ground truth rather than merely plausible,
    which is the whole reason this task is a usable XAI testbed. Explaining a
    binary feasibility head instead only ever asked whether the map looked
    reasonable.

    `task.explain_target` overrides by name.
    """
    names = all_target_names(cfg)
    wanted = getattr(getattr(cfg, "task", None), "explain_target", None)
    if wanted:
        if wanted not in names:
            raise ValueError(f"task.explain_target={wanted!r} is not one of {names}")
        return names.index(wanted)

    if spec is not None and spec.is_hybrid:
        return len(spec.binary)                     # first millimetre column

    if outputs is not None:
        return int(np.argmax(np.asarray(outputs).ravel()))
    return 0
