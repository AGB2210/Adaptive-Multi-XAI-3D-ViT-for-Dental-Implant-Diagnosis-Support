"""Measuring the site localiser: one function, used for selection and for the report.

Position error is reported in MILLIMETRES of the full-resolution scan, split
three ways, because the three matter differently downstream:

  in-plane (x, y)  moves the patch along the arch -- toward a neighbouring
                   tooth, whose anatomy the model will then describe
  vertical (z)     moves the crest within the box -- `patch_centre` puts the
                   crest a quarter from the top, and an error here shifts how
                   much of the canal the box reaches
  3D               both, for the headline

Only sites whose three coordinates are all labelled count toward the 3D error;
a site with an unmeasured crest still counts in-plane.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.localiser_dataset import LocaliserDataset
from src.data.splits import check_disjoint, fold_assignment, load_folds
from src.models.localiser import LocaliserConfig
from src.utils.config import artifacts_dir
from src.utils.log import get_logger

_log = get_logger("localiser")


@torch.no_grad()
def predict_dataset(model, dataset, device, flip: bool = False) -> dict:
    """Run the localiser over every patient in an un-augmented dataset.

    `flip=True` turns each input upside down first, which is how the orientation
    head is tested: a correct head says "flipped" on every one of those.
    """
    model.eval()
    out = {"patients": [], "coords": [], "spread": [], "valid_prob": [], "flip_prob": [],
           "target": [], "mask": [], "valid": []}
    for i in range(len(dataset)):
        x, coords, mask, valid, _ = dataset[i]
        if flip:
            x = torch.flip(x, dims=[3])
            coords = coords.clone()
            coords[:, 2] = (x.shape[3] - 1) - coords[:, 2]
        pred = model(x[None].to(device))
        out["patients"].append(dataset.patients[i])
        out["coords"].append(pred["coords"][0].float().cpu().numpy())
        out["spread"].append(pred["spread"][0].float().cpu().numpy())
        out["valid_prob"].append(torch.sigmoid(pred["valid_logit"][0]).float().cpu().numpy())
        out["flip_prob"].append(float(torch.sigmoid(pred["flip_logit"][0])))
        out["target"].append(coords.numpy())
        out["mask"].append(mask.numpy())
        out["valid"].append(valid.numpy())
    for k in ("coords", "spread", "valid_prob", "target", "mask", "valid"):
        out[k] = np.stack(out[k]) if out[k] else np.zeros((0,))
    out["flip_prob"] = np.asarray(out["flip_prob"])
    return out


def site_errors(pred: dict, factor: int, spacing_mm: float) -> dict:
    """Per (patient, site) errors in mm; NaN where that error is not defined."""
    mm = float(factor) * float(spacing_mm)
    diff = (pred["coords"] - pred["target"]) * mm                 # (P, S, 3)
    mask = pred["mask"].astype(bool)
    xy_ok = mask[..., 0] & mask[..., 1]
    all_ok = xy_ok & mask[..., 2]
    xy = np.where(xy_ok, np.hypot(diff[..., 0], diff[..., 1]), np.nan)
    z = np.where(mask[..., 2], np.abs(diff[..., 2]), np.nan)
    d3 = np.where(all_ok, np.linalg.norm(diff, axis=-1), np.nan)
    return {"xy_mm": xy, "z_mm": z, "error_mm": d3}


def summarise(pred: dict, factor: int, spacing_mm: float) -> dict:
    """The numbers a checkpoint is selected on and reported with."""
    err = site_errors(pred, factor, spacing_mm)

    def stat(a, fn):
        a = a[np.isfinite(a)]
        return float(fn(a)) if a.size else float("nan")

    valid = pred["valid"].astype(bool)
    return {
        "n_patients": int(len(pred["patients"])),
        "n_sites_3d": int(np.isfinite(err["error_mm"]).sum()),
        "median_error_mm": stat(err["error_mm"], np.median),
        "p90_error_mm": stat(err["error_mm"], lambda a: np.percentile(a, 90)),
        "median_xy_mm": stat(err["xy_mm"], np.median),
        "median_spread_mm": stat(pred["spread"] * float(factor) * float(spacing_mm), np.median),
        "median_z_mm": stat(err["z_mm"], np.median),
        "within_3mm": stat(err["error_mm"], lambda a: (a <= 3.0).mean()),
        "in_view_accuracy": float(((pred["valid_prob"] >= 0.5) == valid).mean())
        if valid.size else float("nan"),
    }


def localiser_datasets(cfg, fold: int, limit: int = 0):
    """(LocaliserConfig, {train, val, test} datasets) for one CV round of `cv_folds.json`."""
    loc = cfg.localiser
    lcfg = LocaliserConfig(factor=loc.factor, input_shape=tuple(loc.input_shape),
                           channels=tuple(loc.channels))
    manifest = pd.read_csv(Path(loc.cache_dir) / "manifest.csv", dtype={"patient_id": str})
    manifest = manifest[manifest.status == "ok"]
    sites = pd.read_csv(artifacts_dir(cfg) / cfg.task.sites_csv, dtype={"patient_id": str})
    split = fold_assignment(load_folds(artifacts_dir(cfg) / "cv_folds.json"), fold)
    check_disjoint(split)
    methods = tuple(getattr(loc, "site_methods", ("teeth",)))
    common = dict(cache_dir=loc.cache_dir, manifest=manifest, sites=sites,
                  teeth=lcfg.sites, factor=lcfg.factor, methods=methods)
    out = {}
    for name in ("train", "val", "test"):
        patients = split[name][:limit] if limit else split[name]
        out[name] = LocaliserDataset(patients=patients, augment=(name == "train"),
                                     flip_prob=float(loc.flip_prob),
                                     translate=int(loc.translate_voxels), seed=cfg.seed, **common)
        if out[name].missing:
            _log.warning("%s: %d patients have no cached volume (e.g. %s)", name,
                        len(out[name].missing), out[name].missing[:3])
    return lcfg, out
