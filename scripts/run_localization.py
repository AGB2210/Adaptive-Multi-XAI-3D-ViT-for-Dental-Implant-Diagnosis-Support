"""Score explanations against ToothFairy3's ground-truth masks.

    python scripts/run_localization.py --checkpoint artifacts/runs/cv_fold1/best.pt

Writes artifacts/results_localization.csv and a summary table.

Why this exists. Deletion/insertion ask whether the model's own score moves when
the highlighted voxels are removed -- self-consistency, not correctness. A method
can be perfectly self-consistent and point at the wrong anatomy. ToothFairy3
ships voxel masks, so here the question "is the explanation pointing at the
implant?" has a ground-truth answer.

And it settles the project's open question. 68 of the 74 implant cases also
carry a crown or bridge, so the classifier could be scoring `implant` by
detecting the restoration on top of it. Every case is therefore scored twice:
once against the implant mask, once against the crown/bridge mask. If the
implant explanation prefers the neighbour, the model's implant performance must
not be described as implant detection.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocess import preprocess_mask  # noqa: E402
from src.data.taskdef import label_names_for, primary_dataset  # noqa: E402
from src.utils.config import artifacts_dir, label_path, load_config, volume_path  # noqa: E402
from src.utils.log import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402
from src.xai import ENSEMBLE_METHODS, build_ensemble  # noqa: E402
from src.xai.localization import competing_structure_ratio, localization_scores  # noqa: E402
from src.xai.runner import (  # noqa: E402
    load_cases,
    load_model,
    load_volume,
    require_prerequisites,
    resolve_fold,
    training_baselines,
)

log = get_logger("localization")


def class_index_map(cfg, labels: list[str]) -> dict[str, int]:
    raw = getattr(cfg.task, "class_indices", None)
    if raw is None:
        raise SystemExit("task.class_indices missing -- cannot map labels to mask values")
    table = dict(vars(raw)) if hasattr(raw, "__dict__") else dict(raw)
    missing = [n for n in labels if n not in table]
    if missing:
        raise SystemExit(f"task.class_indices has no entry for {missing}")
    return {n: int(table[n]) for n in labels}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--deterministic", action="store_true",
                    help="pin cuDNN to deterministic kernels: ~100x slower for attribution")
    ap.add_argument("--fold", type=int, default=None,
                    help="cross-validation round; inferred from a cv_foldK checkpoint path")
    ap.add_argument("--label", default="implant", help="which label's explanation to score")
    ap.add_argument("--n-cases", dest="n_cases", type=int, default=20)
    ap.add_argument("--methods", nargs="*", default=list(ENSEMBLE_METHODS))
    ap.add_argument("--ig-steps", dest="ig_steps", type=int, default=256)
    ap.add_argument("--ig-batch", dest="ig_batch", type=int, default=4)
    ap.add_argument("--topk", type=float, default=0.001, help="fraction of voxels kept for IoU/Dice")
    args = ap.parse_args()

    cfg = load_config(args.config)
    labels = label_names_for(cfg)
    cfg.model.num_classes = len(labels)
    if args.label not in labels:
        raise SystemExit(f"--label {args.label!r} is not one of {labels}")
    target = labels.index(args.label)

    fold = resolve_fold(args.checkpoint, args.fold)
    require_prerequisites(cfg, args.checkpoint, fold=fold)
    set_seed(cfg.seed, deterministic=args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = primary_dataset(cfg)
    indices = class_index_map(cfg, labels)
    model, ckpt = load_model(cfg, args.checkpoint, device)
    log.info("checkpoint epoch %s | scoring '%s' explanations against masks",
             ckpt.get("epoch"), args.label)

    baselines, mean_volume = training_baselines(cfg, n=16, device=device, seed=cfg.seed, fold=fold)
    methods = build_ensemble(
        model, device, names=tuple(args.methods),
        mean_volume=mean_volume, baselines=baselines,
        integrated_gradients={"steps": args.ig_steps, "batch_size": args.ig_batch},
        gradient_shap={"n_samples": 24, "batch_size": 4},
    )

    ids, y, cache, _ = load_cases(cfg, "test", dataset, fold=fold)
    positive = [(pid, row) for pid, row in zip(ids, y) if row[target] == 1]
    if not positive:
        raise SystemExit(f"no {args.label}-positive cases in the fold-{fold} test split")
    positive = positive[: args.n_cases]
    log.info("%d %s-positive cases in fold-%s test (scoring %d)",
             sum(int(r[target]) for r in y), args.label, fold, len(positive))

    others = [n for n in labels if n != args.label]
    rows = []
    for i, (pid, row) in enumerate(positive, 1):
        try:
            masks = preprocess_mask(
                volume_path(cfg, dataset, pid), label_path(cfg, dataset, pid), indices,
                out_shape=tuple(cfg.preprocess.out_shape), margin=cfg.preprocess.fg_margin,
                fit_mode=cfg.preprocess.fit_mode, target_spacing=cfg.preprocess.target_spacing,
            )
        except Exception as exc:  # noqa: BLE001 - one bad mask must not lose the run
            log.error("%s: mask failed (%s) -- skipped", pid, exc)
            continue

        if not masks[args.label].any():
            # Labelled positive but nothing survives the crop and 1 mm resample.
            log.warning("%s: %s mask is empty after preprocessing -- skipped", pid, args.label)
            continue

        volume = load_volume(cache, pid, device)
        for name, method in methods.items():
            saliency = method.attribute(volume, target)
            scores = localization_scores(saliency, masks[args.label], k_fraction=args.topk)
            record = {"patient_id": pid, "method": name, "label": args.label, **scores}
            for other in others:
                record[f"vs_{other}"] = competing_structure_ratio(
                    saliency, masks[args.label], masks[other]
                )
                record[f"present_{other}"] = bool(masks[other].any())
            rows.append(record)
        log.info("%d/%d %s", i, len(positive), pid)

    if not rows:
        raise SystemExit("no case produced a usable mask")

    df = pd.DataFrame(rows)
    art = artifacts_dir(cfg)
    out = art / "results_localization.csv"
    df.to_csv(out, index=False)

    agg = df.groupby("method").agg(
        cases=("patient_id", "nunique"),
        pointing_rate=("pointing_hit", "mean"),
        enrichment_median=("enrichment", "median"),
        mass_inside_median=("mass_inside", "median"),
        iou_median=("iou", "median"),
        dice_median=("dice", "median"),
    )
    chance = float(df["mask_fraction"].median())

    print("\n" + "=" * 78)
    print(f"LOCALISATION vs GROUND-TRUTH MASK  --  '{args.label}', fold {fold}")
    print(f"enrichment 1.0 = chance;  the {args.label} mask is {chance:.5%} of the volume")
    print("=" * 78)
    print(agg.round(4).to_string())

    for other in others:
        col, present = f"vs_{other}", df[f"present_{other}"]
        sub = df[present & np.isfinite(df[col])]
        if sub.empty:
            continue
        print(f"\n{args.label} enrichment / {other} enrichment "
              f"(median over {sub['patient_id'].nunique()} cases with both present)")
        print("  > 1 means the explanation prefers the " + args.label)
        print(sub.groupby("method")[col].median().round(3).to_string())

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
