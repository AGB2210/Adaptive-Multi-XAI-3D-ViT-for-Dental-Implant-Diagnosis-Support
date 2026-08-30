"""Faithfulness metrics that need no annotation.

    python scripts/run_faithfulness.py --config configs/sites.yaml \
        --checkpoint artifacts_sites/runs/cv_fold0/best.pt

Do not pass --n-cases to shrink the run: the sample sizes come from the `xai:`
block and are the sizes that make a claim.

Emits artifacts/results_faithfulness.csv, artifacts/results_agreement.csv,
artifacts/results_randomization.csv and the corresponding figures.

Nothing here compares saliency to anatomical ground truth, by design: these
metrics have to work on a cohort without masks. Mask-based localisation scoring
is a separate, additional check. The bone-mass metric is a coarse
intensity-threshold proxy and is labelled as such wherever it appears.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.taskdef import primary_dataset  # noqa: E402
from src.train.targets import TargetSpec, to_report_units  # noqa: E402
from src.utils.config import artifacts_dir, load_config  # noqa: E402
from src.utils.log import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402
from src.xai import ENSEMBLE_METHODS, build_ensemble, build_method  # noqa: E402
from src.xai.base import make_baseline  # noqa: E402
from src.xai.faithfulness import (  # noqa: E402
    agreement_matrix,
    bone_mask,
    bone_mass_fraction,
    deletion_insertion,
    model_randomization_check,
)
from src.xai.runner import (
    ci_table,
    explanation_target,
    load_case_set,
    load_model,
    require_prerequisites,
    resolve_fold,
    select_cases,  # noqa: E402
    training_baselines,
    xai_setting,
)
from src.xai.visualize import (  # noqa: E402
    agreement_heatmap,
    deletion_insertion_curves,
    randomization_plot,
)

log = get_logger("faithfulness")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--deterministic", action="store_true",
                    help="pin cuDNN to deterministic kernels: ~100x slower for "
                         "attribution and still non-deterministic in attention")
    ap.add_argument("--fold", type=int, default=None,
                    help="cross-validation round; inferred from a cv_foldK checkpoint path")
    ap.add_argument("--dataset", default=None,
                    help="dataset name from the config; defaults to data.primary")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-cases", dest="n_cases", type=int, default=None,
                    help="default comes from xai.faithfulness_cases in the config")
    ap.add_argument("--steps", type=int, default=100, help="deletion/insertion steps (~1%% each)")
    ap.add_argument("--randomization-cases", dest="rand_cases", type=int, default=None,
                    help="default comes from xai.randomization_cases in the config")
    ap.add_argument("--only-randomization", dest="only_randomization", action="store_true",
                    help="skip the deletion/insertion sweep and redo only the sanity check, "
                         "reusing the existing results_faithfulness.csv")
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
    dataset = args.dataset or primary_dataset(cfg)
    fold = resolve_fold(args.checkpoint, args.fold)
    require_prerequisites(cfg, args.checkpoint,
                          need_external=(dataset != primary_dataset(cfg)), fold=fold)
    set_seed(cfg.seed, deterministic=args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, ckpt = load_model(cfg, args.checkpoint, device)
    spec = TargetSpec.from_state(ckpt.get("target_spec"))
    baselines, mean_volume = training_baselines(cfg, n=16, device=device, seed=cfg.seed, fold=fold)
    cases = load_case_set(cfg, args.split, dataset, fold=fold)
    ids, y, label_names = cases.ids, cases.y, cases.labels
    # Where the binary block ends. Deletion/insertion reads a probability on one
    # side of this line and a standardised length on the other. A checkpoint
    # with no target_spec predates the hybrid head and is all-binary.
    n_bin = len(spec.binary) if spec.names else len(label_names)
    n_cases = xai_setting(cfg, "faithfulness_cases", args.n_cases, 20)
    log.info("%s/%s: %d cases available", dataset, args.split, len(ids))
    ids, y = select_cases(ids, y, n_cases, cfg.seed, log)

    # Kept so the randomisation cascade can rebuild each method IDENTICALLY.
    # `build_method(name, m, device)` with no options was rebuilding
    # integrated_gradients at its 256-step default against a config asking for
    # 16, and gradient_shap at 24 samples against 8 -- and, worse, without the
    # training baselines. The cascade then compared an intact attribution built
    # with one baseline against re-initialised attributions built with another,
    # so part of what it measured was the baseline change rather than the weight
    # randomisation. It also made the stage 70x slower than the other methods,
    # which is how it was noticed at all.
    method_opts = {
        "integrated_gradients": {"steps": xai_setting(cfg, "ig_steps", None, 256),
                                 "batch_size": 4},
        "gradient_shap": {"n_samples": xai_setting(cfg, "shap_samples", None, 24),
                          "batch_size": 4},
    }
    methods = build_ensemble(model, device, names=tuple(args.methods),
                             mean_volume=mean_volume, baselines=baselines,
                             **method_opts)

    def rebuild(m, name):
        opts = dict(method_opts.get(name, {}))
        if name == "integrated_gradients" and mean_volume is not None:
            opts.setdefault("mean_volume", mean_volume)
        if name == "gradient_shap" and baselines is not None:
            opts.setdefault("baselines", baselines)
        return build_method(name, m, device, **opts)
    art = artifacts_dir(cfg)
    figures = art / "figures"

    rows, agreement_rows = [], []
    todo = [] if args.only_randomization else ids
    # Progress, because this loop prints nothing for tens of minutes on CPU and
    # a remote operator cannot tell a slow run from a hung one. The RUNBOOK sends
    # someone else to run this on a rented box; silence there costs real money.
    for case_index, pid in enumerate(todo):
        started = time.perf_counter()
        volume = cases.load(pid, device)
        baseline = make_baseline(volume, args.baseline, mean_volume=mean_volume)
        mask = bone_mask(volume)

        with torch.no_grad():
            outputs = to_report_units(model(volume)[0].cpu().numpy(), spec)

        # See runner.explanation_target: on the hybrid task this is
        # available_height_mm, whose evidence is crest-to-canal distance and
        # therefore checkable against an annotated structure.
        target = explanation_target(cfg, spec, outputs)
        maps, results = {}, {}

        for name, method in methods.items():
            saliency = method.attribute(volume, target)
            maps[name] = saliency
            result = deletion_insertion(model, volume, saliency, target,
                                        baseline=baseline, steps=args.steps,
                                        target_is_probability=target < n_bin,
                                        score=args.score)
            results[name] = result
            plausibility = bone_mass_fraction(saliency, mask)

            rows.append({
                "dataset": dataset,
                # `pid` is a case id -- `patient#tooth` on the site task. Writing
                # it into a column named `patient_id` is what defeated the
                # clustered bootstrap: `clustered_ci` resampled distinct values
                # of `patient_id`, found one per row, and produced a row
                # bootstrap while every printed table said "patient-clustered".
                "case_id": pid,
                "patient_id": cases.patient_of(pid),
                "target_label": label_names[target] if target < len(label_names) else target,
                # In the target's own unit -- a probability for the binary
                # block, millimetres for the millimetre block. `int()` on the
                # truth turned 17.4 mm of bone into 17.
                "predicted_value": float(outputs[target]),
                "true_value": (float(y[case_index, target])
                               if target < y.shape[1] else float("nan")),
                "target_unit": "probability" if target < n_bin else "mm",
                # In the row, because a deletion AUC means nothing without them
                # and two runs with different settings are otherwise
                # indistinguishable in the CSV.
                "baseline_kind": args.baseline,
                "score": args.score,
                "method": name,
                "deletion_auc": result["deletion_auc"],
                "insertion_auc": result["insertion_auc"],
                # Coarse intensity-threshold proxy, NOT clinical ground truth.
                "bone_mass_fraction": plausibility["bone_mass_fraction"],
                "bone_mask_volume_fraction": plausibility["mask_volume_fraction"],
                "bone_enrichment": plausibility["enrichment"],
            })

        matrix = agreement_matrix(maps)
        for pair, rho in matrix["spearman"].items():
            a, b = pair.split("|")
            if a == b:
                continue
            agreement_rows.append({
                "dataset": dataset,
                "case_id": pid, "patient_id": cases.patient_of(pid),
                "method_a": a, "method_b": b, "spearman": rho,
                "jaccard_top1pct": matrix["jaccard"]["top1pct"][pair],
                "jaccard_top5pct": matrix["jaccard"]["top5pct"][pair],
            })

        if case_index == 0:
            deletion_insertion_curves(results, figures / f"deletion_insertion_{dataset}_{pid}.png",
                                      title=f"{dataset} / {pid} / target={label_names[target]}",
                                      target_is_probability=(target < n_bin))
            agreement_heatmap(matrix, figures / f"agreement_spearman_{dataset}_{pid}.png",
                              "spearman", f"Inter-method agreement — {pid}")

        # Every case, not every fifth. At `% 5` a 4-case run printed nothing at
        # all between "cases selected" and the randomisation stage -- tens of
        # minutes of silence that a remote operator cannot distinguish from a
        # hang. The per-case time is also the only estimate available for how
        # long the full run will take.
        log.info("%d/%d cases done (%.1fs)", case_index + 1, len(todo),
                 time.perf_counter() - started)

    faithfulness = pd.DataFrame(rows)
    if args.only_randomization:
        existing = art / "results_faithfulness.csv"
        if not existing.is_file():
            raise SystemExit(f"--only-randomization needs {existing} from a full run")
        faithfulness = pd.read_csv(existing)
        log.info("reusing %d deletion/insertion rows from %s", len(faithfulness), existing)
    else:
        faithfulness.to_csv(art / "results_faithfulness.csv", index=False)
        pd.DataFrame(agreement_rows).to_csv(art / "results_agreement.csv", index=False)

    print("\n" + "=" * 86)
    print(f"FAITHFULNESS — {dataset}/{args.split}, n={len(ids)} cases")
    # ONLY FOR A PROBABILITY TARGET. The metric assumes a score that rises
    # with evidence for the class; a millimetre head is not that -- deleting
    # voxels moves a length toward whatever the baseline implies, which may
    # be larger or smaller, so neither direction is founded under
    # score="response". Under score="deviation" the negated absolute change
    # restores the assumption and the usual reading applies again.
    if target < n_bin:
        print("deletion AUC: LOWER is better    insertion AUC: HIGHER is better")
    elif args.score == "deviation":
        print("deletion AUC: LOWER is better    insertion AUC: HIGHER is better"
              "   (millimetre head, restored by score=deviation)")
    else:
        print(f"target {label_names[target]!r} is a MILLIMETRE head at "
              f"score={args.score!r}: neither direction is founded here. "
              f"Compare methods, do not rank them against an absolute.")
    print("=" * 86)
    cols = ["deletion_auc", "insertion_auc", "bone_mass_fraction", "bone_enrichment"]
    # pandas `.mean()` skips NaN, and `bone_mass_fraction` / `bone_enrichment`
    # return NaN when the proxy mask is empty -- so without a count column two
    # methods can be averaged over different subsets of cases and printed side
    # by side as if they were comparable. The randomisation block below already
    # counts its undefined rows and says so; this table did not.
    summary = faithfulness.groupby("method")[cols].agg(["mean", "std", "count"])
    print(summary.round(4).to_string())
    n_cases = faithfulness.groupby("method").size()
    short = {c: summary[(c, "count")][summary[(c, "count")] < n_cases]
             for c in cols}
    for c, rows in short.items():
        if len(rows):
            print(f"   NOTE: {c} is undefined on some cases -- averaged over "
                  f"{rows.to_dict()} of {n_cases.iloc[0]}. Not comparable across "
                  f"methods with different counts.")
    print("\nbone_* columns are a coarse intensity-threshold proxy, not clinical ground truth:")
    print("they cannot tell WHICH anatomy the saliency landed on.")

    # ---- model randomisation sanity check --------------------------------
    rand_cases = xai_setting(cfg, "randomization_cases", args.rand_cases, 3)
    log.info("model-randomisation sanity check on %d cases...", rand_cases)
    randomization = {}
    rand_rows = []
    for case_no, pid in enumerate(ids[:rand_cases], 1):
        volume = cases.load(pid, device)
        with torch.no_grad():
            target = explanation_target(cfg, spec,
                                        to_report_units(model(volume)[0].cpu().numpy(), spec))

        for name in methods:
            # Per method, not per case: the cascade re-initialises the network 12
            # times and re-attributes at every stage, so one case takes minutes
            # and a whole run printed nothing between "sanity check on N cases"
            # and its own results. That silence is indistinguishable from a hang,
            # which is exactly the failure mode the deletion/insertion loop had.
            t0 = time.perf_counter()
            intact = methods[name].attribute(volume, target)
            stages = model_randomization_check(
                model,
                lambda m, _n=name: rebuild(m, _n),
                volume, target, intact, seed=cfg.seed,
            )
            randomization.setdefault(name, stages)
            for stage in stages:
                rand_rows.append({"case_id": pid,
                                  "patient_id": cases.patient_of(pid),
                                  "method": name, **stage})
            log.info("  randomisation %d/%d %s (%.0fs)", case_no, rand_cases, name,
                     time.perf_counter() - t0)

    rand_df = pd.DataFrame(rand_rows)
    rand_df.to_csv(art / "results_randomization.csv", index=False)
    if randomization:
        randomization_plot(randomization, figures / "model_randomization.png")

    print("\n" + "=" * 86)
    print("MODEL-RANDOMISATION SANITY CHECK (Adebayo et al. 2018)")
    print("A faithful map must DECORRELATE as weights are destroyed.")
    print("=" * 86)
    if not rand_df.empty and "spearman_vs_intact" in rand_df:
        # `.agg(["mean", "last"])` SKIPS NaN, and a constant degraded map yields
        # NaN rather than a number. An earlier version therefore reported the
        # FIRST stage of the cascade under the heading "last" and concluded the
        # opposite of what the data said. Select the final stage explicitly, and
        # count the undefined rows instead of letting pandas drop them.
        last_stage = rand_df["stage"].iloc[-1]
        tail = rand_df[rand_df["stage"] == last_stage]
        col = f"at_{last_stage}"
        final = pd.DataFrame({
            "mean": rand_df.groupby("method")["spearman_vs_intact"].mean(),
            col: tail.groupby("method")["spearman_vs_intact"].mean(),
            "n_undefined": rand_df.groupby("method")["spearman_vs_intact"]
                                  .apply(lambda s: int(s.isna().sum())),
            "n_stages": rand_df.groupby("method")["stage"].nunique(),
        })
        print(final.round(4).to_string())
        print()
        print(f"last stage of the cascade: {last_stage!r}")

        # An ordering without intervals is an ordering the data may not support.
        # Adebayo et al. define no pass/fail cutoff, so no verdict is printed
        # here -- the intervals are the claim. They are clustered by PATIENT
        # because several sites come from one jaw; see runner.clustered_ci.
        if "patient_id" in tail:
            table = ci_table(tail, "spearman_vs_intact")
            print()
            print(f"at_{last_stage} with a 95% interval clustered by patient "
                  "(lower = decorrelates = more faithful):")
            print(table.round(4).to_string())
            overlapping = [
                (a, b)
                for i, a in enumerate(table.index) for b in list(table.index)[i + 1:]
                if table.loc[a, "ci_hi"] >= table.loc[b, "ci_lo"]
            ]
            if overlapping:
                print()
                print("   intervals OVERLAP, so these pairs are not ordered by the data:")
                for a, b in overlapping:
                    print(f"      {a} / {b}")
            else:
                print("   no intervals overlap: the ordering above is supported")

        undefined = final[final["n_undefined"] > 0]
        if not undefined.empty:
            print()
            print("!! Some stages produced a CONSTANT map, so Spearman is undefined")
            print("   (NaN, not 0). A constant map means the degraded network lost all")
            print("   input dependence -- confirm randomisation RE-INITIALISED the weights")
            print("   rather than nullifying them before reading this as a pass:")
            print(f"   {undefined.index.tolist()}")

        # Two different failures, and calling both "barely changed" misreports
        # one of them. A map still correlated with the intact one is tracking
        # the image rather than the weights. A map that has INVERTED is not
        # unchanged -- but a stable anti-correlation is still structural
        # dependence on the input, not the decorrelation this check looks for.
        stuck = final[final[col] > 0.5].index.tolist()
        flipped = final[final[col] < -0.5].index.tolist()

        if stuck:
            print()
            print(f"!! {stuck} barely change under FULL randomisation")
            print(f"   (rho > 0.5 at {last_stage!r}). That is the edge-detector")
            print("   signature: the same map from a trained and a random network.")
        if flipped:
            print()
            print(f"!! {flipped} INVERT rather than decorrelate")
            print(f"   (rho < -0.5 at {last_stage!r}). Not unchanged, but a stable")
            print("   anti-correlation is still structure carried from the input.")
        if not stuck and not flipped:
            print()
            print(f"all methods decorrelated (|rho| <= 0.5 at {last_stage!r}).")


    print(f"\nwrote {art / 'results_faithfulness.csv'}, {art / 'results_agreement.csv'}, "
          f"{art / 'results_randomization.csv'}")


if __name__ == "__main__":
    main()
