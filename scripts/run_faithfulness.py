"""Faithfulness metrics that need no annotation.

    python scripts/run_faithfulness.py --checkpoint artifacts/runs/vit3d_vit3d/best.pt --n-cases 20

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
from src.train.targets import TargetSpec  # noqa: E402
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
        baseline = make_baseline(volume, "blur")
        mask = bone_mask(volume)

        with torch.no_grad():
            probs = torch.sigmoid(model(volume))[0].cpu().numpy()

        # See runner.explanation_target: on the hybrid task this is
        # available_height_mm, whose evidence is crest-to-canal distance and
        # therefore checkable against an annotated structure.
        target = explanation_target(cfg, spec, probs)
        maps, results = {}, {}

        for name, method in methods.items():
            saliency = method.attribute(volume, target)
            maps[name] = saliency
            result = deletion_insertion(model, volume, saliency, target,
                                        baseline=baseline, steps=args.steps)
            results[name] = result
            plausibility = bone_mass_fraction(saliency, mask)

            rows.append({
                "dataset": dataset,
                "patient_id": pid,
                "target_label": label_names[target] if target < len(label_names) else target,
                "predicted_prob": float(probs[target]),
                "true_label": int(y[case_index, target]) if target < y.shape[1] else -1,
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
                "dataset": dataset, "patient_id": pid,
                "method_a": a, "method_b": b, "spearman": rho,
                "jaccard_top1pct": matrix["jaccard"]["top1pct"][pair],
                "jaccard_top5pct": matrix["jaccard"]["top5pct"][pair],
            })

        if case_index == 0:
            deletion_insertion_curves(results, figures / f"deletion_insertion_{dataset}_{pid}.png",
                                      title=f"{dataset} / {pid} / target={label_names[target]}")
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
    print("deletion AUC: LOWER is better    insertion AUC: HIGHER is better")
    print("=" * 86)
    summary = faithfulness.groupby("method")[
        ["deletion_auc", "insertion_auc", "bone_mass_fraction", "bone_enrichment"]
    ].agg(["mean", "std"])
    print(summary.round(4).to_string())
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
                                        torch.sigmoid(model(volume))[0].cpu().numpy())

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
                rand_rows.append({"patient_id": pid, "method": name, **stage})
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
