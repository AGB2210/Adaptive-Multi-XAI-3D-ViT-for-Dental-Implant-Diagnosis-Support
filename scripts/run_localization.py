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
    load_case_set,
    load_model,
    require_prerequisites,
    resolve_fold,
    training_baselines,
)
from src.xai.site_masks import describe_coverage, patch_masks  # noqa: E402

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
    ap.add_argument("--label", default=None,
                    help="which label's explanation to score "
                         "(site task defaults to 'feasible', otherwise 'implant')")
    ap.add_argument("--target-value", dest="target_value", type=int, default=None,
                    help="score cases where the label equals this "
                         "(site task defaults to 0: sites that are NOT feasible, "
                         "because that is the verdict the nerve explains)")
    ap.add_argument("--n-cases", dest="n_cases", type=int, default=20)
    ap.add_argument("--methods", nargs="*", default=list(ENSEMBLE_METHODS))
    ap.add_argument("--ig-steps", dest="ig_steps", type=int, default=256)
    ap.add_argument("--ig-batch", dest="ig_batch", type=int, default=4)
    ap.add_argument("--topk", type=float, default=0.001, help="fraction of voxels kept for IoU/Dice")
    args = ap.parse_args()

    cfg = load_config(args.config)
    labels = label_names_for(cfg)
    cfg.model.num_classes = len(labels)
    is_sites = bool(getattr(cfg.task, "sites_csv", None))

    label = args.label or ("feasible" if is_sites else "implant")
    if label not in labels:
        raise SystemExit(f"--label {label!r} is not one of {labels}")
    target = labels.index(label)
    # On the site task the interesting verdict is the NEGATIVE one: an
    # explanation for "not feasible" should land on whatever blocks the implant,
    # and in the mandible that is the nerve canal 289 times out of 361.
    want = args.target_value if args.target_value is not None else (0 if is_sites else 1)

    fold = resolve_fold(args.checkpoint, args.fold)
    require_prerequisites(cfg, args.checkpoint, fold=fold)
    set_seed(cfg.seed, deterministic=args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = primary_dataset(cfg)
    # Only the detection path maps label names onto mask class indices. The
    # site task's labels are verdicts, not structures, and its anatomy comes
    # from src/xai/site_masks.py instead.
    indices = None if is_sites else class_index_map(cfg, labels)
    model, ckpt = load_model(cfg, args.checkpoint, device)
    log.info("checkpoint epoch %s | scoring '%s'==%d explanations against %s",
             ckpt.get("epoch"), label, want,
             "anatomy (nerve vs jawbone)" if is_sites else "restoration masks")

    baselines, mean_volume = training_baselines(cfg, n=16, device=device, seed=cfg.seed, fold=fold)
    methods = build_ensemble(
        model, device, names=tuple(args.methods),
        mean_volume=mean_volume, baselines=baselines,
        integrated_gradients={"steps": args.ig_steps, "batch_size": args.ig_batch},
        gradient_shap={"n_samples": 24, "batch_size": 4},
    )

    cases = load_case_set(cfg, "test", dataset, fold=fold)
    ids, y = cases.ids, cases.y
    selected = [(pid, row) for pid, row in zip(ids, y) if int(row[target]) == want]
    if not selected:
        raise SystemExit(f"no cases with {label}=={want} in the fold-{fold} test split")
    n_available = len(selected)
    selected = selected[: args.n_cases]
    log.info("%d cases with %s==%d in fold-%s test (scoring %d)",
             n_available, label, want, fold, len(selected))

    # WHAT THE EXPLANATION IS SCORED AGAINST.
    #
    # Site task: the primary structure is the inferior alveolar canal and the
    #   competitor is the surrounding jawbone. The canal is a dark, low-contrast
    #   tube inside bone, so an edge detector cannot find it by accident -- which
    #   is precisely what made the old target useless. Spreading over bone
    #   generally describes where the jaw is, not why the site fails.
    #
    # Detection task: the implant mask, with crown and bridge as competitors,
    #   answering whether the model scores `implant` by detecting the
    #   restoration sitting on top of it.
    primary = "nerve" if is_sites else label
    others = ["jawbone"] if is_sites else [n for n in labels if n != label]

    rows = []
    for i, (case_id, _row) in enumerate(selected, 1):
        pid = cases.patient_of(case_id)
        try:
            if is_sites:
                masks = patch_masks(label_path(cfg, dataset, pid), cases.row(case_id),
                                    cases.patch_size)
            else:
                masks = preprocess_mask(
                    volume_path(cfg, dataset, pid), label_path(cfg, dataset, pid), indices,
                    out_shape=tuple(cfg.preprocess.out_shape), margin=cfg.preprocess.fg_margin,
                    fit_mode=cfg.preprocess.fit_mode,
                    target_spacing=cfg.preprocess.target_spacing,
                )
        except Exception as exc:  # noqa: BLE001 - one bad mask must not lose the run
            log.error("%s: mask failed (%s) -- skipped", case_id, exc)
            continue

        if not masks[primary].any():
            # No canal inside this patch. Anterior mandibular sites genuinely
            # have none, and scoring them against an empty mask would divide by
            # a zero-size target rather than report a miss.
            log.warning("%s: no %s in this patch -- skipped", case_id, primary)
            continue

        volume = cases.load(case_id, device)
        coverage = describe_coverage(masks) if is_sites else {}
        for name, method in methods.items():
            saliency = method.attribute(volume, target)
            scores = localization_scores(saliency, masks[primary], k_fraction=args.topk)
            record = {"case_id": case_id, "patient_id": pid, "method": name,
                      "label": label, "target_value": want,
                      "structure": primary, **scores}
            record.update({f"coverage_{k}": v for k, v in coverage.items()})
            for other in others:
                record[f"vs_{other}"] = competing_structure_ratio(
                    saliency, masks[primary], masks[other]
                )
                record[f"present_{other}"] = bool(masks[other].any())
            rows.append(record)
        log.info("%d/%d %s", i, len(selected), case_id)

    if not rows:
        raise SystemExit("no case produced a usable mask")

    df = pd.DataFrame(rows)
    art = artifacts_dir(cfg)
    out = art / "results_localization.csv"
    df.to_csv(out, index=False)

    agg = df.groupby("method").agg(
        cases=("case_id", "nunique"),
        pointing_rate=("pointing_hit", "mean"),
        enrichment_median=("enrichment", "median"),
        mass_inside_median=("mass_inside", "median"),
        iou_median=("iou", "median"),
        dice_median=("dice", "median"),
    )
    chance = float(df["mask_fraction"].median())

    print("\n" + "=" * 78)
    print(f"LOCALISATION vs GROUND-TRUTH ANATOMY  --  '{label}'=={want}, "
          f"structure '{primary}', fold {fold}")
    print(f"enrichment 1.0 = chance;  the {primary} mask is {chance:.5%} of the patch")
    print("=" * 78)
    print(agg.round(4).to_string())

    for other in others:
        col, present = f"vs_{other}", df[f"present_{other}"]
        sub = df[present & np.isfinite(df[col])]
        if sub.empty:
            continue
        print(f"\n{primary} enrichment / {other} enrichment "
              f"(median over {sub['case_id'].nunique()} cases with both present)")
        print("  > 1 means the explanation prefers the " + primary)
        print(sub.groupby("method")[col].median().round(3).to_string())

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
