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
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.taskdef import primary_dataset  # noqa: E402
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
from src.xai.runner import (  # noqa: E402
    load_case_set,
    load_cases,
    load_model,
    load_volume,
    require_prerequisites,
    resolve_fold,
    training_baselines,
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
    ap.add_argument("--n-cases", dest="n_cases", type=int, default=20)
    ap.add_argument("--steps", type=int, default=100, help="deletion/insertion steps (~1%% each)")
    ap.add_argument("--randomization-cases", dest="rand_cases", type=int, default=3)
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

    model, _ = load_model(cfg, args.checkpoint, device)
    baselines, mean_volume = training_baselines(cfg, n=16, device=device, seed=cfg.seed, fold=fold)
    cases = load_case_set(cfg, args.split, dataset, fold=fold)
    ids, y, cache, label_names = cases.ids, cases.y, cases.cache, cases.labels
    ids, y = ids[: args.n_cases], y[: args.n_cases]
    log.info("%s/%s: %d cases", dataset, args.split, len(ids))

    methods = build_ensemble(model, device, names=tuple(args.methods),
                             mean_volume=mean_volume, baselines=baselines)
    art = artifacts_dir(cfg)
    figures = art / "figures"

    rows, agreement_rows = [], []
    for case_index, pid in enumerate([] if args.only_randomization else ids):
        volume = cases.load(pid, device)
        baseline = make_baseline(volume, "blur")
        mask = bone_mask(volume)

        with torch.no_grad():
            probs = torch.sigmoid(model(volume))[0].cpu().numpy()

        # Explain the label the model is most confident about for this patient.
        target = int(np.argmax(probs))
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

        if (case_index + 1) % 5 == 0:
            log.info("%d/%d cases done", case_index + 1, len(ids))

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
    log.info("model-randomisation sanity check on %d cases...", args.rand_cases)
    randomization = {}
    rand_rows = []
    for pid in ids[: args.rand_cases]:
        volume = load_volume(cache, pid, device)
        with torch.no_grad():
            target = int(np.argmax(torch.sigmoid(model(volume))[0].cpu().numpy()))

        for name in methods:
            intact = methods[name].attribute(volume, target)
            stages = model_randomization_check(
                model,
                lambda m, _n=name: build_method(_n, m, device),
                volume, target, intact, seed=cfg.seed,
            )
            randomization.setdefault(name, stages)
            for stage in stages:
                rand_rows.append({"patient_id": pid, "method": name, **stage})

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

        suspicious = final[final[col].abs() > 0.5].index.tolist()
        if suspicious:
            print()
            print(f"!! {suspicious} barely change under FULL randomisation")
            print(f"   (|rho| > 0.5 at {last_stage!r}). That is the edge-detector")
            print("   signature. Report it explicitly.")


    print(f"\nwrote {art / 'results_faithfulness.csv'}, {art / 'results_agreement.csv'}, "
          f"{art / 'results_randomization.csv'}")


if __name__ == "__main__":
    main()
