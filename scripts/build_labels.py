"""Derive binary case-level labels from voxel-level segmentation masks.

    python scripts/build_labels.py --config configs/default.yaml --dataset toothfairy3

Writes artifacts/labels_<dataset>.csv: one row per case, one column per label in
task.labels, plus a vox_<label> column recording the voxel count that produced it.

Why the counts are kept: a positive here means "at least min_voxels of this class
are present", and a case sitting one voxel either side of that line is a labelling
decision, not a fact. Keeping the count makes the threshold auditable after the
fact instead of baking an unrecoverable judgement into a 0/1.

The class indices are NOT guessed. They come from task.class_indices in the
config, because an index that is wrong by one silently trains the model against a
different anatomical structure and nothing downstream would notice.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.taskdef import label_names_for, primary_dataset  # noqa: E402
from src.utils.config import (  # noqa: E402
    artifacts_dir,
    dataset_root,
    label_path,
    list_cases,
    load_config,
)
from src.utils.log import get_logger  # noqa: E402

log = get_logger("build_labels")


def class_indices(cfg, labels: list[str]) -> dict[str, int]:
    """label name -> mask class index, from the config. Never inferred."""
    raw = getattr(cfg.task, "class_indices", None)
    mapping = dict(vars(raw)) if raw is not None and not isinstance(raw, dict) else dict(raw or {})
    missing = [name for name in labels if name not in mapping]
    if missing:
        raise SystemExit(
            f"task.class_indices has no entry for {missing}.\n"
            "Fill these in from the dataset's own class list before running this. "
            "A guessed index trains the model against the wrong structure and the "
            "metrics will look entirely reasonable while doing it."
        )
    return {name: int(mapping[name]) for name in labels}


def count_classes(mask_path: Path, indices: dict[str, int]) -> dict[str, int]:
    """Voxels of each requested class in one mask."""
    mask = np.asanyarray(nib.load(str(mask_path)).dataobj)
    # One pass over the volume; masks are large and reading them is the cost here.
    present, counts = np.unique(mask, return_counts=True)
    histogram = dict(zip(present.tolist(), counts.tolist()))
    return {name: int(histogram.get(idx, 0)) for name, idx in indices.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset", default=None, help="defaults to data.primary")
    ap.add_argument("--min-voxels", dest="min_voxels", type=int, default=None,
                    help="overrides task.min_voxels")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    dataset = args.dataset or primary_dataset(cfg)
    labels = label_names_for(cfg)
    indices = class_indices(cfg, labels)
    min_voxels = args.min_voxels or int(getattr(cfg.task, "min_voxels", 0) or 0)
    if min_voxels <= 0:
        raise SystemExit("task.min_voxels must be a positive integer: a single stray "
                         "voxel of a class is annotation noise, not a finding")

    root = dataset_root(cfg, dataset)
    if not root.is_dir():
        raise SystemExit(f"dataset root not found: {root.resolve()}")

    cases = list_cases(cfg, dataset)
    if args.limit:
        cases = cases[: args.limit]
    log.info("%s: %d cases, classes %s, min_voxels=%d", dataset, len(cases), indices, min_voxels)

    rows, missing = [], 0
    for k, pid in enumerate(cases, 1):
        path = label_path(cfg, dataset, pid)
        if path is None:
            raise SystemExit(f"dataset {dataset!r} declares no `labels` path pattern; "
                             "this script needs voxel masks")
        if not path.is_file():
            missing += 1
            log.warning("no mask for %s -- skipping", pid)
            continue

        counts = count_classes(path, indices)
        row = {"patient_id": pid}
        row.update({name: int(counts[name] >= min_voxels) for name in labels})
        row.update({f"vox_{name}": counts[name] for name in labels})
        rows.append(row)
        if k % 50 == 0:
            log.info("  %d/%d", k, len(cases))

    if not rows:
        raise SystemExit("no masks were read -- check the dataset's `labels` path pattern")

    df = pd.DataFrame(rows)
    out = artifacts_dir(cfg) / f"labels_{dataset}.csv"
    df.to_csv(out, index=False, encoding="utf-8")

    log.info("wrote %s (%d cases, %d without a mask)", out, len(df), missing)
    print("\nprevalence")
    for name in labels:
        n = int(df[name].sum())
        print(f"  {name:<20} {n:>5}  ({100 * n / len(df):5.1f}%)")
        if n == 0:
            log.warning("%s has no positive cases -- check its class index", name)


if __name__ == "__main__":
    main()
