"""Pool the test folds of a finished cross-validation into one result.

    python scripts/pool_cv.py --config configs/default.yaml --folds 5

Each CV round trains on three folds, selects on a fourth, and scores the fifth.
Averaging the five resulting AUROCs is *not* the same as scoring all 532 cases
together: the mean of five estimates on ~107 cases each has no straightforward
confidence interval, and it silently weights a fold with 14 implant positives the
same as one with 15.

So this script rebuilds the per-case predictions instead. For each round it
reloads that round's best checkpoint, runs it over its own held-out test fold --
the only cases that model never saw -- and concatenates. Every case is then
predicted exactly once, by a model blind to it, and the pooled numbers are
ordinary metrics over the whole cohort with an ordinary bootstrap CI.

Predictions are written to artifacts/cv_predictions.csv so the XAI phase and any
later analysis can reuse them without touching a GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import (  # noqa: E402
    CachedVolumeDataset,
    load_label_matrix,
    restrict_to_cache,
)
from src.data.splits import check_folds_disjoint, fold_assignment, load_folds  # noqa: E402
from src.data.taskdef import label_names_for, primary_dataset  # noqa: E402
from src.models import build_model  # noqa: E402
from src.train.loop import load_checkpoint_file, predict  # noqa: E402
from src.train.metrics import evaluate, format_metrics  # noqa: E402
from src.utils.config import artifacts_dir, load_config  # noqa: E402
from src.utils.log import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

log = get_logger("pool_cv")


def f1_from_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else 2 * tp / denom


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--runs", default=None,
                    help="directory holding cv_fold0..N (default: <train.out_dir>)")
    ap.add_argument("--num-workers", dest="num_workers", type=int, default=0)
    ap.add_argument("--out", default=None, help="where to write pooled metrics json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    labels = label_names_for(cfg)
    cfg.model.num_classes = len(labels)

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s labels=%s", device, ", ".join(labels))

    dataset = primary_dataset(cfg)
    cache_dir = Path(cfg.data.cache_dir) / dataset
    adir = artifacts_dir(cfg)
    patient_ids, y = load_label_matrix(adir / f"labels_{dataset}.csv", labels)
    patient_ids, y = restrict_to_cache(patient_ids, y, cache_dir)
    index = {pid: i for i, pid in enumerate(patient_ids)}

    folds = load_folds(adir / "cv_folds.json")
    check_folds_disjoint(folds)
    if len(folds) != args.folds:
        raise SystemExit(f"cv_folds.json holds {len(folds)} folds, --folds says {args.folds}")

    runs_root = Path(args.runs or cfg.train.out_dir)

    pooled_ids: list[str] = []
    pooled_true: list[np.ndarray] = []
    pooled_prob: list[np.ndarray] = []
    pooled_pred: list[np.ndarray] = []
    per_fold: list[dict] = []

    for k in range(len(folds)):
        run_dir = runs_root / f"cv_fold{k}"
        ckpt_path = run_dir / "best.pt"
        if not ckpt_path.exists():
            raise SystemExit(f"{ckpt_path} missing -- round {k} has not finished")

        test_ids = [p for p in fold_assignment(folds, k)["test"] if p in index]
        rows = np.array([index[p] for p in test_ids], dtype=int)
        y_test = y[rows]

        loader = DataLoader(
            CachedVolumeDataset(cache_dir, test_ids, y_test, augment=None, seed=cfg.seed),
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

        model = build_model(cfg.model, img_size=cfg.preprocess.out_shape[0])
        ckpt = load_checkpoint_file(ckpt_path, device)
        if ckpt.get("label_names") and list(ckpt["label_names"]) != list(labels):
            raise SystemExit(
                f"round {k} was trained on {ckpt['label_names']} but the config says {labels}"
            )
        model.load_state_dict(ckpt["model"])
        model.to(device).eval()

        probs, targets = predict(model, loader, device, amp=bool(cfg.train.amp))

        # Thresholds come from that round's own validation fold, never from the
        # pooled set -- picking one threshold on all 532 would tune it on the
        # same cases it is scored against.
        metrics_path = run_dir / "metrics.json"
        thresholds = json.loads(metrics_path.read_text(encoding="utf-8"))["val"]["thresholds"]
        thr = np.array([thresholds[name] for name in labels], dtype=np.float64)
        preds = (probs >= thr).astype(int)

        fold_macro = evaluate(targets, probs, labels)["macro_auroc"]
        per_fold.append({"fold": k, "n": len(test_ids), "macro_auroc": fold_macro})
        log.info("round %d: n=%3d  macro AUROC %.4f  (epoch %s)",
                 k, len(test_ids), fold_macro, ckpt.get("epoch", "?"))

        pooled_ids.extend(test_ids)
        pooled_true.append(targets)
        pooled_prob.append(probs)
        pooled_pred.append(preds)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    y_true = np.concatenate(pooled_true)
    y_prob = np.concatenate(pooled_prob)
    y_pred = np.concatenate(pooled_pred)

    if len(pooled_ids) != len(set(pooled_ids)):
        raise SystemExit("a case appears in more than one test fold -- partition is broken")
    log.info("pooled %d cases, each predicted once by a model that never saw it", len(pooled_ids))

    pred_csv = adir / "cv_predictions.csv"
    with pred_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["patient_id", "fold"]
                   + [f"true_{n}" for n in labels]
                   + [f"prob_{n}" for n in labels]
                   + [f"pred_{n}" for n in labels])
        fold_of = {p: k for k, f in enumerate(folds) for p in f}
        for i, pid in enumerate(pooled_ids):
            w.writerow([pid, fold_of[pid]]
                       + y_true[i].astype(int).tolist()
                       + [f"{v:.6f}" for v in y_prob[i]]
                       + y_pred[i].tolist())
    log.info("wrote %s", pred_csv)

    pooled = evaluate(y_true, y_prob, labels, n_boot=cfg.eval.bootstrap_n, ci=cfg.eval.bootstrap_ci)

    # F1 is recomputed from each fold's own thresholds rather than retuned on the
    # pooled set, so it stays comparable with the per-round numbers.
    for i, name in enumerate(labels):
        pooled["per_label"][name]["f1"] = f1_from_binary(y_true[:, i], y_pred[:, i])
    pooled["macro_f1"] = float(np.mean([pooled["per_label"][n]["f1"] for n in labels]))

    print("\n" + format_metrics(pooled, labels, f"VIT3D -- POOLED across {len(folds)} test folds"))

    macros = [f["macro_auroc"] for f in per_fold]
    print(f"\nper-fold macro AUROC: {', '.join('%.4f' % m for m in macros)}")
    print(f"  mean {np.mean(macros):.4f}   sd {np.std(macros, ddof=1):.4f}"
          f"   range {min(macros):.4f}-{max(macros):.4f}")

    out_path = Path(args.out or adir / "cv_pooled_metrics.json")
    out_path.write_text(json.dumps({
        "pooled": pooled,
        "per_fold": per_fold,
        "n_cases": len(pooled_ids),
        "labels": labels,
        "n_folds": len(folds),
    }, indent=2), encoding="utf-8")
    log.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
