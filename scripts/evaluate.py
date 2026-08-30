"""Evaluate a checkpoint on the primary test split and, zero-shot, on the external cohort.

    python scripts/evaluate.py --checkpoint artifacts/runs/vit3d/best.pt
    python scripts/evaluate.py --checkpoint ... --external      # add the external cohort

The external cohort is never trained on:
evaluated on Italian cohorts (0.175-0.3 mm, different scanners). It is never
trained on. impacted_tooth is excluded -- the reports have no reliable equivalent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import CachedVolumeDataset, load_label_matrix, restrict_to_cache  # noqa: E402
from src.data.taskdef import (  # noqa: E402
    external_dataset,
    label_names_for,
    primary_dataset,
    regression_names_for,
)
from src.models import build_model  # noqa: E402
from src.train.loop import load_checkpoint_file, predict  # noqa: E402
from src.train.metrics import evaluate, format_metrics  # noqa: E402
from src.train.targets import (  # noqa: E402
    TargetSpec,
    format_regression,
    regression_metrics,
)
from src.utils.config import artifacts_dir, load_config  # noqa: E402
from src.utils.log import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402
from src.xai.runner import (  # noqa: E402
    load_case_set,
    model_img_size,
    patients_of,
    predict_case_logits,
)

log = get_logger("evaluate")


def build_loader(cache_dir, ids, labels, batch_size, num_workers):
    dataset = CachedVolumeDataset(cache_dir, ids, labels, augment=None)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model", default=None, choices=["vit3d", "cnn3d"])
    ap.add_argument("--external", action="store_true",
                    help="also evaluate zero-shot on the external cohort")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--num-workers", dest="num_workers", type=int, default=0)
    ap.add_argument("--fold", type=int, default=None,
                    help="cross-validation round; omit for a single train/val/test split")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg.model.name = args.model
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # The checkpoint is authoritative about which labels it was trained on and in
    # what order; the config is only a fallback for checkpoints predating the field.
    ckpt = load_checkpoint_file(args.checkpoint, device)
    labels = list(ckpt.get("label_names") or label_names_for(cfg))
    cfg.model.num_classes = len(labels)

    model = build_model(cfg.model, img_size=model_img_size(cfg))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    log.info("loaded %s (epoch %s), %d labels: %s",
             args.checkpoint, ckpt.get("epoch"), len(labels), ", ".join(labels))

    art = artifacts_dir(cfg)
    results: dict = {"checkpoint": str(args.checkpoint), "model": cfg.model.name}

    # ---- internal: the primary cohort -----------------------------------
    primary = primary_dataset(cfg)

    # load_case_set covers both tasks: a case is a whole scan, or a tooth site
    # whose input is a patch cut from one. It also applies the split, which for
    # the site task is by patient.
    cases = load_case_set(cfg, args.split, primary, fold=args.fold)
    ids, targets = cases.ids, cases.y
    if not ids:
        raise SystemExit(f"no cases in the {args.split!r} split -- run scripts/train.py first")

    logits = predict_case_logits(model, cases, device, batch_size=cfg.train.batch_size)
    # Sigmoid the binary block ONLY. On a millimetre head it squashes an 18 mm
    # prediction to 1.0 and every metric downstream becomes a statement about a
    # constant. The millimetre block is un-standardised instead, using the
    # scaler that travelled in the checkpoint.
    spec = TargetSpec.from_state(ckpt.get("target_spec"))
    n_bin = len(label_names_for(cfg))
    probs = np.array(logits, dtype=np.float64, copy=True)
    probs[:, :n_bin] = 1.0 / (1.0 + np.exp(-logits[:, :n_bin]))
    if spec.is_hybrid:
        probs = spec.to_millimetres(probs.astype(np.float32)).astype(np.float64)

    # Thresholds come from validation, never from the split being reported.
    val_metrics_path = Path(args.checkpoint).parent / "best_val_metrics.json"
    thresholds = None
    if val_metrics_path.exists() and args.split != "val":
        thresholds = json.loads(val_metrics_path.read_text(encoding="utf-8"))["thresholds"]
        log.info("using validation-tuned thresholds from %s", val_metrics_path)

    # Binary metrics on the binary block, millimetre metrics on the rest.
    # Scoring every column with `evaluate` reported available_height_mm at
    # "prevalence 13.11" with an AP of 15.315 and a "threshold" of 19.997 mm --
    # numbers that are not wrong so much as meaningless, since 18 mm is not a
    # probability and AP over it is not defined.
    mm_names = regression_names_for(cfg)
    # Cluster the interval on patients. On the site task `ids` are `patient#tooth`
    # and one jaw supplies up to fourteen of them, so a row bootstrap would treat
    # fourteen views of one patient as fourteen independent draws. `groups=None`
    # is correct only when a row already is a patient, which is the whole-volume
    # case below.
    groups = np.asarray(patients_of(ids)) if cases.is_sites else None
    metrics = evaluate(targets[:, :n_bin], probs[:, :n_bin], labels[:n_bin],
                       thresholds=thresholds,
                       n_boot=cfg.eval.bootstrap_n, ci=cfg.eval.bootstrap_ci,
                       groups=groups)
    print("\n" + format_metrics(metrics, labels[:n_bin],
                                f"{primary} {args.split} (n={len(ids)})"))
    results[f"{primary}_{args.split}"] = metrics
    if mm_names:
        mm = regression_metrics(targets[:, n_bin:], probs[:, n_bin:], mm_names)
        print(format_regression(mm, f"{primary} {args.split} -- millimetres"))
        results[f"{primary}_{args.split}_mm"] = mm

    # ---- external cohort, zero-shot -------------------------------------
    external = external_dataset(cfg)
    if args.external and not external:
        log.warning("--external given but the config declares no data.external -- skipping")
    if args.external and external:
        tf_cache = Path(cfg.data.cache_dir) / external
        tf_labels_csv = art / f"labels_{external}.csv"
        if not tf_cache.exists() or not tf_labels_csv.exists():
            log.warning("%s cache or labels missing -- skipping external evaluation", external)
        else:
            # Only labels this checkpoint has a head for AND the external cohort carries.
            available = set(pd.read_csv(tf_labels_csv, nrows=0).columns)
            shared = [n for n in labels if n in available]
            if not shared:
                raise SystemExit(
                    f"checkpoint labels {labels} share no column with {tf_labels_csv}"
                )
            tf_ids, tf_y = load_label_matrix(tf_labels_csv, shared)
            tf_ids, tf_y = restrict_to_cache(tf_ids, tf_y, tf_cache)
            log.info("%s: %d cases with a cached volume", external, len(tf_ids))

            tf_loader = build_loader(tf_cache, tf_ids, tf_y, cfg.train.batch_size, args.num_workers)
            tf_probs, tf_targets = predict(model, tf_loader, device)

            # Select the model's columns for the shared label subset only.
            cols = [labels.index(name) for name in shared]
            tf_probs = tf_probs[:, cols]

            tf_thresholds = {k: thresholds[k] for k in shared} if thresholds else None
            tf_metrics = evaluate(tf_targets, tf_probs, shared, thresholds=tf_thresholds,
                                  n_boot=cfg.eval.bootstrap_n, ci=cfg.eval.bootstrap_ci)
            print("\n" + format_metrics(tf_metrics, shared,
                                        f"{external} zero-shot external (n={len(tf_ids)})"))
            results[f"{external}_external"] = tf_metrics

            # Per-cohort breakdown: sub-cohorts come from different scanners.
            cohorts = pd.read_csv(tf_labels_csv, dtype={"patient_id": str}).set_index("patient_id")["cohort"]
            by_cohort = {}
            for cohort in sorted(set(cohorts.get(p, "?") for p in tf_ids)):
                mask = np.array([cohorts.get(p, "?") == cohort for p in tf_ids])
                if mask.sum() < 10:
                    continue
                m = evaluate(tf_targets[mask], tf_probs[mask], shared, thresholds=tf_thresholds)
                by_cohort[cohort] = {"n": int(mask.sum()), "macro_auroc": m["macro_auroc"]}
            if by_cohort:
                print("\nper-cohort macro AUROC:", json.dumps(by_cohort, indent=2))
                results[f"{external}_by_cohort"] = by_cohort

    out = Path(args.out or (Path(args.checkpoint).parent / "evaluation.json"))
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
