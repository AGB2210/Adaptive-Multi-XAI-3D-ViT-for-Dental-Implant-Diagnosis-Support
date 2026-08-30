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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.splits import check_folds_disjoint, load_folds  # noqa: E402
from src.data.taskdef import (  # noqa: E402
    all_target_names,
    label_names_for,
    primary_dataset,
    regression_names_for,
)
from src.models import build_model  # noqa: E402
from src.train.loop import load_checkpoint_file  # noqa: E402
from src.train.metrics import evaluate, format_metrics  # noqa: E402
from src.train.targets import (  # noqa: E402
    TargetSpec,
    format_regression,
    regression_metrics,
    to_report_units,
)
from src.utils.config import artifacts_dir, load_config  # noqa: E402
from src.utils.log import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402
from src.xai.runner import (  # noqa: E402
    SITE_SEP,
    load_case_set,
    model_img_size,
    patients_of,
    predict_case_logits,
)

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
    # EVERY head, binary and millimetre. Sizing from the binary block alone
    # builds a 1-output model against a 3-output checkpoint, and torch refuses
    # to load it -- which is the good outcome. The silent version of this bug is
    # a head that happens to match and reports one target under another's name.
    labels = all_target_names(cfg)
    n_bin = len(label_names_for(cfg))
    cfg.model.num_classes = len(labels)

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s labels=%s", device, ", ".join(labels))

    dataset = primary_dataset(cfg)
    adir = artifacts_dir(cfg)

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

        # Each round predicts ONLY on its own held-out fold, so no case is ever
        # scored by a model that trained on it. load_case_set applies that
        # fold's partition and works on either task.
        cases = load_case_set(cfg, "test", dataset, fold=k)
        test_ids, targets = cases.ids, cases.y
        if not test_ids:
            raise SystemExit(f"round {k} has no held-out cases")

        model = build_model(cfg.model, img_size=model_img_size(cfg))
        ckpt = load_checkpoint_file(ckpt_path, device)
        if ckpt.get("label_names") and list(ckpt["label_names"]) != list(labels):
            raise SystemExit(
                f"round {k} was trained on {ckpt['label_names']} but the config says {labels}"
            )
        model.load_state_dict(ckpt["model"])
        model.to(device).eval()

        logits = predict_case_logits(model, cases, device, batch_size=cfg.train.batch_size)
        # Sigmoid the BINARY block only, and un-standardise the millimetre block
        # with this round's own scaler. Sigmoiding the whole row -- which this
        # script did until the millimetre metrics below were checked against a
        # hand computation -- turns an 18 mm prediction into 0.9999 and then
        # scores it against a truth of 18, so the pooled MAE reported roughly
        # the mean of the target instead of the model's error. Each fold carries
        # its own scaler, so the conversion has to happen here, inside the loop,
        # before anything is pooled.
        spec = TargetSpec.from_state(ckpt.get("target_spec"))
        if n_bin < len(labels) and not spec.is_hybrid:
            raise SystemExit(
                f"round {k}: the config declares millimetre targets "
                f"{labels[n_bin:]} but the checkpoint carries no target_spec, so "
                f"there is no scaler to undo. Retrain, or pool a classification "
                f"config -- do not report standardised units as millimetres."
            )
        probs = to_report_units(logits, spec)

        # Thresholds come from that round's own validation fold, never from the
        # pooled set -- picking one threshold on all 532 would tune it on the
        # same cases it is scored against.
        metrics_path = run_dir / "metrics.json"
        val = json.loads(metrics_path.read_text(encoding="utf-8"))["val"]
        # metrics.json is nested by unit now -- {"val": {"classification": ...,
        # "regression": ...}} -- because millimetre heads have no thresholds to
        # report. The flat shape is still read so checkpoints from before the
        # hybrid head can be pooled without being retrained.
        thresholds = val.get("classification", val).get("thresholds", {})
        thr = np.array([thresholds[name] for name in labels[:n_bin]], dtype=np.float64)
        preds = (probs[:, :n_bin] >= thr).astype(int)

        # Binary block only: macro AUROC over a millimetre column is not a
        # quantity. The millimetre heads are pooled and scored below, in mm.
        fold_macro = evaluate(targets[:, :n_bin], probs[:, :n_bin],
                              labels[:n_bin])["macro_auroc"]
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
        # Column names carry the unit: `prob_` for the binary block, `mm_` for
        # the millimetre block. A header that calls 18.4 mm a probability is how
        # a reader ends up quoting it as one.
        def _pred_col(i: int, name: str) -> str:
            return f"prob_{name}" if i < n_bin else f"mm_{name}"

        w.writerow(["case_id", "patient_id", "fold"]
                   + [f"true_{n}" for n in labels]
                   + [_pred_col(i, n) for i, n in enumerate(labels)]
                   + [f"pred_{n}" for n in labels[:n_bin]])
        # Folds hold PATIENT ids. On the site task a case id is
        # "<patient>#<tooth>", so the fold is looked up by the patient it
        # belongs to -- 28 sites from one scan always move together.
        fold_of = {p: k for k, f in enumerate(folds) for p in f}
        for i, case_id in enumerate(pooled_ids):
            pid = str(case_id).split(SITE_SEP)[0]
            # `.astype(int)` truncated the truth: a site with 17.4 mm of bone
            # was written to the predictions CSV as 17, and any later analysis
            # reading this file inherited a silently rounded ground truth.
            w.writerow([case_id, pid, fold_of[pid]]
                       + [str(int(v)) if j < n_bin else f"{v:.6f}"
                          for j, v in enumerate(y_true[i])]
                       + [f"{v:.6f}" for v in y_prob[i]]
                       + y_pred[i][:n_bin].tolist())
    log.info("wrote %s", pred_csv)

    binary_names = labels[:n_bin]
    # Cluster the interval on PATIENTS. A pooled row is a site, and 6,787 sites
    # come from 486 patients, so resampling rows would treat fourteen views of
    # one jaw as fourteen independent draws and return an interval far narrower
    # than the data supports. `bootstrap_ci` used to do exactly that while its
    # docstring said otherwise.
    pooled = evaluate(y_true[:, :n_bin], y_prob[:, :n_bin], binary_names,
                      n_boot=cfg.eval.bootstrap_n, ci=cfg.eval.bootstrap_ci,
                      groups=np.asarray(patients_of(pooled_ids)))

    # F1 is recomputed from each fold's own thresholds rather than retuned on the
    # pooled set, so it stays comparable with the per-round numbers.
    for i, name in enumerate(binary_names):
        pooled["per_label"][name]["f1"] = f1_from_binary(y_true[:, i], y_pred[:, i])
    pooled["macro_f1"] = float(np.mean([pooled["per_label"][n]["f1"] for n in binary_names]))

    print("\n" + format_metrics(pooled, binary_names,
                                f"VIT3D -- POOLED across {len(folds)} test folds"))

    mm_names = regression_names_for(cfg)
    if mm_names:
        # Every case predicted once, by a model that never saw it -- the same
        # discipline as the classification pooling, in millimetres.
        pooled_mm = regression_metrics(y_true[:, n_bin:], y_prob[:, n_bin:], mm_names)
        print(format_regression(pooled_mm,
                                f"pooled across {len(folds)} test folds -- millimetres"))

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
