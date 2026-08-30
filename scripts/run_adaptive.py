"""Calibration, confidence gate, and agreement-weighted fusion.

    python scripts/run_adaptive.py --config configs/sites.yaml \n        --checkpoint artifacts_sites/runs/cv_fold0/best.pt

Tests the three claims the project rests on, and reports them whichever way they
come out:

  1. Does fusion beat every individual method on the HELD-OUT metric?
  2. Does agreement-weighted fusion beat a uniform-average ensemble?
  3. The Pareto curve: measured compute cost vs faithfulness.

Weighting uses insertion AUC; evaluation uses deletion AUC. They are never the
same metric -- src.xai.adaptive.fuse() raises if asked to make them so.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.taskdef import primary_dataset  # noqa: E402
from src.train.targets import TargetSpec, to_report_units  # noqa: E402
from src.utils.config import artifacts_dir, load_config  # noqa: E402
from src.utils.log import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402
from src.xai import ENSEMBLE_METHODS, build_ensemble  # noqa: E402
from src.xai.adaptive import (  # noqa: E402
    EVAL_METRIC,
    WEIGHT_METRIC,
    ConfidenceGate,
    fuse,
    pareto_sweep,
)
from src.xai.base import make_baseline  # noqa: E402
from src.xai.calibration import (  # noqa: E402
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    reliability_diagram,
    uncertainty,
)
from src.xai.runner import (
    explanation_target,
    load_case_set,
    load_model,
    patients_of,
    predict_case_logits,
    require_prerequisites,
    resolve_fold,
    select_cases,  # noqa: E402
    training_baselines,
    xai_setting,
)
from src.xai.visualize import pareto_curve  # noqa: E402

log = get_logger("adaptive")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--deterministic", action="store_true",
                    help="pin cuDNN to deterministic kernels: ~100x slower for "
                         "attribution and still non-deterministic in attention")
    ap.add_argument("--fold", type=int, default=None,
                    help="cross-validation round; inferred from a cv_foldK checkpoint path")
    ap.add_argument("--n-cases", dest="n_cases", type=int, default=None,
                    help="default comes from xai.adaptive_cases in the config")
    ap.add_argument("--from-csv", dest="from_csv", action="store_true",
                    help="reuse artifacts/results_ablations.csv and only redo the summary; "
                         "the per-case sweep costs over an hour and must not be lost to a "
                         "reporting bug")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--uncertainty", default="margin", choices=["margin", "entropy"])
    ap.add_argument("--ensemble-fraction", dest="ensemble_fraction", type=float, default=0.3)
    ap.add_argument("--baseline", default="blur", choices=["blur", "mean", "zero"],
                    help="what a deleted voxel is replaced with. The default blur is "
                         "sigma=4 voxels = 1.2 mm at 0.3 mm, which removes TEXTURE and "
                         "leaves geometry -- and the explained head is a crest-to-canal "
                         "DISTANCE, so a deleted region still carries the measurement. "
                         "Use 'mean' to destroy geometry")
    ap.add_argument("--score", default="response", choices=["response", "deviation"],
                    help="what the curve integrates. 'response' is the model output, "
                         "correct for a probability. 'deviation' is minus the distance "
                         "from the full-input prediction, which is the reading a "
                         "millimetre head needs -- see deletion_insertion")
    ap.add_argument("--methods", nargs="*", default=list(ENSEMBLE_METHODS))
    args = ap.parse_args()

    cfg = load_config(args.config)
    fold = resolve_fold(args.checkpoint, args.fold)
    require_prerequisites(cfg, args.checkpoint, fold=fold)
    set_seed(cfg.seed, deterministic=args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, ckpt = load_model(cfg, args.checkpoint, device)
    spec = TargetSpec.from_state(ckpt.get("target_spec"))
    baselines, mean_volume = training_baselines(cfg, n=16, device=device, seed=cfg.seed, fold=fold)
    art = artifacts_dir(cfg)

    # ---- 1. calibration on VALIDATION -----------------------------------
    val_cases = load_case_set(cfg, "val", primary_dataset(cfg), fold=fold)
    val_ids, val_y, label_names = val_cases.ids, val_cases.y, val_cases.labels
    val_logits = predict_case_logits(model, val_cases, device)

    # Calibration is a statement about PROBABILITIES, so it runs on the binary
    # block alone. Passing the whole hybrid matrix -- which this script did --
    # fits a temperature against targets of 13-26 mm: the BCE has no minimum in
    # T, LBFGS walks log T to infinity, and T comes back NaN. Every calibrated
    # probability, every uncertainty score and the gate threshold are then NaN,
    # and `nan >= nan` is False, so the gate never escalates a single case and
    # the Pareto sweep collapses to one point. The tell was ece_before = 9.097,
    # a value ECE cannot reach when both its arguments are probabilities.
    # A checkpoint with no target_spec predates the hybrid head and is
    # all-binary; a spec that declares millimetres but no binary column has
    # nothing to calibrate, and fitting a temperature on an empty slice would
    # return a number that means nothing.
    bin_names = list(spec.binary) if spec.names else list(label_names)
    if not bin_names:
        raise SystemExit(
            "this checkpoint has no binary head, so there is no probability to "
            "calibrate and no confidence gate to fit. The adaptive layer is "
            "defined on the binary block; run it on a config that has one."
        )
    val_bin_logits = spec.slice_binary(val_logits) if spec.names else val_logits
    val_bin_y = spec.slice_binary(val_y) if spec.names else val_y

    probs_before = 1.0 / (1.0 + np.exp(-val_bin_logits))
    ece_before, bins_before = expected_calibration_error(probs_before, val_bin_y)

    temperature = fit_temperature(val_bin_logits, val_bin_y)
    probs_after = apply_temperature(val_bin_logits, temperature)
    ece_after, bins_after = expected_calibration_error(probs_after, val_bin_y)

    # `val_ids` holds CASE ids -- `patient#tooth` on the site task -- so its
    # length is a site count, not a patient count. Reporting it as patients
    # overstated the sample by an order of magnitude (1339 against 97) and is
    # the same fault as the `patient_id` column that defeated the clustered
    # bootstrap. Both counts are written, each under its own name.
    n_cases = len(val_ids)
    n_patients = len(set(patients_of(val_ids)))
    (art / "calibration").mkdir(parents=True, exist_ok=True)
    reliability_diagram(bins_before, bins_after, ece_before, ece_after,
                        art / "calibration" / "reliability.png",
                        title=f"Validation calibration (n={n_cases} cases from "
                              f"{n_patients} patients x {len(bin_names)} binary labels)")
    (art / "calibration" / "calibration.json").write_text(json.dumps({
        "temperature": temperature, "ece_before": ece_before, "ece_after": ece_after,
        "n_val_cases": n_cases, "n_val_patients": n_patients,
        "fitted_on": "validation split only",
        "calibrated_labels": list(bin_names),
    }, indent=2), encoding="utf-8")

    print("\n" + "=" * 74)
    print("CALIBRATION (temperature scaling, fitted on VALIDATION only)")
    print("=" * 74)
    print(f"temperature T = {temperature:.4f}")
    print(f"ECE before    = {ece_before:.4f}")
    print(f"ECE after     = {ece_after:.4f}   ({'improved' if ece_after < ece_before else 'NO improvement'})")
    if ece_after >= ece_before:
        print("Temperature scaling did not help -- report this; the gate rests on calibrated scores.")

    # ---- 2. confidence gate fitted on validation ------------------------
    val_uncertainty = uncertainty(probs_after, args.uncertainty)
    gate = ConfidenceGate.fit(val_uncertainty, args.ensemble_fraction)
    log.info("gate threshold %.4f (escalates ~%.0f%% of validation cases)",
             gate.threshold, 100 * args.ensemble_fraction)

    # ---- 3. fusion on TEST ----------------------------------------------
    cases = load_case_set(cfg, "test", primary_dataset(cfg), fold=fold)
    ids, y = cases.ids, cases.y
    ids, y = select_cases(ids, y, xai_setting(cfg, "adaptive_cases", args.n_cases, 20),
                          cfg.seed, log)
    test_logits = predict_case_logits(model, cases, device)

    # `select_cases` draws a RANDOM subset, but the logits above are predicted
    # over every held-out case, so the two are not positionally aligned: row i
    # of the prediction array is not the case at position i of the sample. The
    # loop below indexed them as if it were, which attached each row's
    # uncertainty, routing decision and confidence to a different site than its
    # own patient_id. Look the row up by id instead.
    row_of = {pid: i for i, pid in enumerate(cases.ids)}

    test_bin_logits = spec.slice_binary(test_logits) if spec.names else test_logits
    test_bin_probs = apply_temperature(test_bin_logits, temperature)
    test_uncertainty = uncertainty(test_bin_probs, args.uncertainty)
    # Probabilities for the binary block, millimetres for the millimetre block,
    # so a row reports its target in the unit that target is measured in.
    test_report = to_report_units(test_logits, spec, temperature)

    methods = build_ensemble(model, device, names=tuple(args.methods),
                             mean_volume=mean_volume, baselines=baselines)

    rows, per_case = [], []
    for i, pid in enumerate([] if args.from_csv else ids):
        row = row_of[pid]
        volume = cases.load(pid, device)
        baseline = make_baseline(volume, args.baseline, mean_volume=mean_volume)
        target = explanation_target(cfg, spec, test_report[row])

        maps = {name: method.attribute(volume, target) for name, method in methods.items()}
        result = fuse(model, volume, maps, target,
                      weight_metric=WEIGHT_METRIC, eval_metric=EVAL_METRIC,
                      steps=args.steps, baseline=baseline,
                      target_is_probability=target < len(bin_names),
                      score=args.score)

        rows.append({
            # `pid` is a case id -- `patient#tooth`. Both go in, each under its
            # own name, so a bootstrap over this file can cluster correctly and
            # `RUNBOOK.md` 6's check passes.
            "case_id": pid,
            "patient_id": cases.patient_of(pid),
            "target_label": label_names[target],
            # In the target's own unit: a calibrated probability for a binary
            # head, millimetres for a millimetre head. The old column was named
            # `calibrated_prob` for both, which invited reading 18.4 mm as a
            # confidence of 18.4.
            "target_value": float(test_report[row, target]),
            "target_unit": "probability" if target < len(bin_names) else "mm",
            "uncertainty": float(test_uncertainty[row]),
            "routed_to_ensemble": gate.use_ensemble(test_uncertainty[row]),
            "weight_metric": WEIGHT_METRIC,
            "eval_metric": EVAL_METRIC,
            "baseline_kind": args.baseline,
            "score": args.score,
            "fused_eval": result["fused_eval"],
            "uniform_eval": result["uniform_eval"],
            "beats_best_individual": result["beats_best_individual"],
            "beats_uniform": result["beats_uniform"],
            **{f"eval_{k}": v for k, v in result["per_method_eval"].items()},
            **{f"weight_{k}": v for k, v in result["weights"].items()},
        })
        if gate.cheap_method not in result["per_method_eval"]:
            raise KeyError(
                f"the gate's cheap method {gate.cheap_method!r} is not among the methods run "
                f"({sorted(result['per_method_eval'])}); the Pareto curve would compare nothing."
            )
        per_case.append({
            "uncertainty": float(test_uncertainty[row]),
            "cheap_eval": result["per_method_eval"][gate.cheap_method],
            "fused_eval": result["fused_eval"],
        })

        if (i + 1) % 5 == 0:
            log.info("%d/%d cases", i + 1, len(ids))

    if args.from_csv:
        existing = art / "results_ablations.csv"
        if not existing.is_file():
            raise SystemExit(f"--from-csv needs {existing} from a previous run")
        ablations = pd.read_csv(existing)
        log.info("reusing %d rows from %s", len(ablations), existing)
        # per_case is built inside the loop the reuse path skips, so the Pareto
        # sweep would silently produce an empty table. Every column it needs is
        # already in the CSV.
        cheap_col = f"eval_{gate.cheap_method}"
        if cheap_col not in ablations.columns:
            raise SystemExit(f"{existing} has no {cheap_col}; rerun without --from-csv")
        per_case = ablations[["uncertainty", cheap_col, "fused_eval"]].rename(
            columns={cheap_col: "cheap_eval"}
        ).to_dict("records")
    else:
        ablations = pd.DataFrame(rows)
        ablations.to_csv(art / "results_ablations.csv", index=False)


    # ---- report the three claims ----------------------------------------
    print("\n" + "=" * 74)
    print("ADAPTIVE LAYER — the three claims")
    print(f"weighted BY: {WEIGHT_METRIC}   evaluated ON: {EVAL_METRIC} (lower is better)")
    print("=" * 74)

    # eval_metric holds the metric's NAME, not a score, and the prefix match
    # swept it in alongside the per-method columns -- the summary then tried to
    # average a column of strings and killed the run after all the compute.
    eval_cols = [c for c in ablations.columns
                 if c.startswith("eval_") and pd.api.types.is_numeric_dtype(ablations[c])]
    print("\nmean held-out score per method:")
    # `.mean()` skips NaN, so an ordering built from it can rank methods scored
    # on different subsets of cases. Print the count alongside: equal counts mean
    # the ordering is like for like, unequal ones mean it is not.
    for col in sorted(eval_cols, key=lambda c: ablations[c].mean()):
        print(f"   {col.replace('eval_', ''):<24}{ablations[col].mean():.4f}"
              f"   n={int(ablations[col].notna().sum())}")
    print(f"   {'FUSED (agreement-weighted)':<24}{ablations['fused_eval'].mean():.4f}")
    print(f"   {'UNIFORM ensemble':<24}{ablations['uniform_eval'].mean():.4f}")

    win1 = float(ablations["beats_best_individual"].mean())
    win2 = float(ablations["beats_uniform"].mean())
    print(f"\nclaim 1 — fusion beats every individual method : {win1:.0%} of cases")
    print(f"claim 2 — agreement-weighted beats uniform     : {win2:.0%} of cases")
    if win1 < 0.5:
        print("   -> claim 1 NOT supported. Report as a negative result; do not tune until it passes.")
    if win2 < 0.5:
        print("   -> claim 2 NOT supported: per-input weighting is not earning its complexity.")

    # ---- Pareto sweep ----------------------------------------------------
    runtime_path = art / "xai_runtime.csv"
    if runtime_path.exists():
        runtimes = pd.read_csv(runtime_path).set_index("method")["seconds_mean"].to_dict()
        sweep = pareto_sweep(per_case, runtimes, cheap_method=gate.cheap_method)
        pd.DataFrame(sweep).to_csv(art / "results_pareto.csv", index=False)
        pareto_curve(sweep, art / "figures" / "pareto_compute_vs_faithfulness.png")
        print("\nclaim 3 — compute vs faithfulness Pareto curve:")
        print(pd.DataFrame(sweep).round(4).to_string(index=False))
    else:
        print(f"\nclaim 3 skipped: {runtime_path} not found — run scripts/run_xai.py first "
              "(the gate's justification is MEASURED compute, so this cannot be assumed).")

    print(f"\nwrote {art / 'results_ablations.csv'}")


if __name__ == "__main__":
    main()
