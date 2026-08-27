"""The XAI methods — synthetic sanity, IG completeness, runtime.

    python scripts/run_xai.py --checkpoint artifacts/runs/vit3d_vit3d/best.pt

Emits artifacts/xai_runtime.csv and artifacts/xai_sanity.csv. Stop and read both
before continuing to 2B: a method that fails the synthetic-signal test is broken,
and that is a finding to report, not something to work around.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.taskdef import label_names_for, primary_dataset  # noqa: E402
from src.utils.config import artifacts_dir, load_config  # noqa: E402
from src.utils.log import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402
from src.xai import ENSEMBLE_METHODS, build_ensemble, build_method  # noqa: E402
from src.xai.runner import (  # noqa: E402
    describe_token_geometry,
    load_case_set,
    load_model,
    model_img_size,
    require_prerequisites,
    resolve_fold,
    training_baselines,
    xai_setting,
)

NATIVE_SPACING_MM = 0.3      # ToothFairy3 is isotropic 0.3 mm; nothing is resampled
CANAL_DIAMETER_MM = 3.0      # inferior alveolar canal, 2-3 mm across

log = get_logger("run_xai")


def synthetic_sanity(model, device, methods_spec, label_names: list[str],
                     n_cases: int = 8, seed: int = 0,
                     shape: tuple[int, int, int] = (128, 128, 128)) -> pd.DataFrame:
    """Does each method's saliency mass land on the planted signal?

    The synthetic volumes carry a blob at a known site per label, so 'saliency
    mass inside the planted region / mass expected by chance' is a direct hit
    rate. Enrichment <= 1 means the method is no better than random.

    NOTE: this measures the method against a model that must actually have learned
    the planted signal. Run it with a checkpoint trained on synthetic data
    (scripts/train.py --synthetic); against an untrained model it tests nothing.
    """
    from tests.synthetic import make_case, signal_mask

    rows = []
    for case in range(n_cases):
        labels = np.zeros(len(label_names), dtype=np.int64)
        target = case % len(label_names)
        labels[target] = 1

        volume_np, _ = make_case(1000 + case, shape=shape, labels=labels)
        volume = torch.from_numpy(volume_np)[None, None].to(device)
        mask = torch.from_numpy(signal_mask(labels, shape=shape)).to(device)
        chance = float(mask.float().mean())

        for name, method in methods_spec.items():
            saliency = method.attribute(volume, target)
            total = float(saliency.sum())
            inside = float(saliency[mask].sum()) if total > 0 else 0.0
            fraction = inside / total if total > 0 else float("nan")

            rows.append({
                "method": name,
                "case": case,
                "target_label": label_names[target],
                "mass_in_planted_region": fraction,
                "chance_level": chance,
                "enrichment": fraction / chance if chance > 0 else float("nan"),
            })
    return pd.DataFrame(rows)


def benchmark(model, device, methods_spec, volume, target_label: int, repeats: int = 3) -> pd.DataFrame:
    """Measured wall-clock per method per volume.

    The adaptive layer's entire justification is compute cost, so these numbers
    are measured, never assumed.
    """
    rows = []
    for i, (name, method) in enumerate(methods_spec.items(), 1):
        log.info("benchmarking %d/%d: %s", i, len(methods_spec), name)
        method.attribute(volume, target_label)  # warm up (lazy init, cuDNN autotune)
        if device.type == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            method.attribute(volume, target_label)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        log.info("    %s: %.2fs per volume", name, float(np.mean(times)))
        rows.append({
            "method": name,
            "seconds_mean": float(np.mean(times)),
            "seconds_std": float(np.std(times)),
            "seconds_min": float(np.min(times)),
            "repeats": repeats,
            "device": str(device),
        })
    return pd.DataFrame(rows).sort_values("seconds_mean")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--deterministic", action="store_true",
                    help="pin cuDNN to deterministic kernels: ~100x slower for "
                         "attribution and still non-deterministic in attention")
    ap.add_argument("--fold", type=int, default=None,
                    help="cross-validation round; inferred from a cv_foldK checkpoint path")
    ap.add_argument("--methods", nargs="*", default=list(ENSEMBLE_METHODS) + ["grad_rollout"])
    ap.add_argument("--ig-steps", dest="ig_steps", type=int, default=None,
                    help="default comes from xai.ig_steps in the config")
    ap.add_argument("--shap-samples", dest="shap_samples", type=int, default=None,
                    help="default comes from xai.shap_samples in the config")
    # Reduce THIS, never --ig-steps, when memory is tight: batch size costs time,
    # step count costs correctness (32 steps -> 48% completeness error).
    ap.add_argument("--ig-batch", dest="ig_batch", type=int, default=4)
    ap.add_argument("--shap-batch", dest="shap_batch", type=int, default=4)
    ap.add_argument("--sanity-cases", dest="sanity_cases", type=int, default=8)
    ap.add_argument("--skip-sanity", dest="skip_sanity", action="store_true")
    ap.add_argument("--lime", action="store_true", help="include 3D LIME (ablation only, slow)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    label_names = label_names_for(cfg)
    fold = resolve_fold(args.checkpoint, args.fold)
    require_prerequisites(cfg, args.checkpoint, fold=fold)
    set_seed(cfg.seed, deterministic=args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, ckpt = load_model(cfg, args.checkpoint, device)
    geometry = describe_token_geometry(model)
    log.info("checkpoint epoch %s | token geometry: %s", ckpt.get("epoch"), geometry)
    # Read off the checkpoint, never assumed: the conv stem's stride 2 means a
    # token spans 2 x patch_size input voxels, which is how patch_size 8 came to
    # be documented as 2.4 mm/token while measuring 4.8 mm -- wider than the
    # 2-3 mm canal the explanations exist to resolve.
    mm_per_token = model_img_size(cfg) / geometry["grid_size"][0] * NATIVE_SPACING_MM
    log.info("token grid %s (%d tokens), %.2f mm per token",
             geometry["grid_size"], geometry["n_patch_tokens"], mm_per_token)
    if mm_per_token > CANAL_DIAMETER_MM:
        log.warning("a token spans %.2f mm, wider than the %.1f mm inferior alveolar "
                    "canal. Localisation cannot resolve what it is being scored "
                    "against; lower model.patch_size.", mm_per_token, CANAL_DIAMETER_MM)

    ig_steps = xai_setting(cfg, "ig_steps", args.ig_steps, 256)
    shap_samples = xai_setting(cfg, "shap_samples", args.shap_samples, 24)
    log.info("IG: %d steps at batch %d | GradientSHAP: %d samples at batch %d",
             ig_steps, args.ig_batch, shap_samples, args.shap_batch)
    if shap_samples < 128:
        log.warning("GradientSHAP at %d samples: the 24-sample setting measured a "
                    "relative standard error of 0.5689 and was not quotable. Raise "
                    "xai.shap_samples before reporting anything from this run.",
                    shap_samples)
    baselines, mean_volume = training_baselines(cfg, n=16, device=device, seed=cfg.seed, fold=fold)
    log.info("baselines ready (%s volumes from round %s training folds)",
             "none" if baselines is None else len(baselines), fold)

    names = list(args.methods) + (["lime3d"] if args.lime else [])
    methods = build_ensemble(
        model, device, names=tuple(n for n in names if n != "lime3d"),
        mean_volume=mean_volume, baselines=baselines,
        integrated_gradients={"steps": ig_steps, "batch_size": args.ig_batch},
        gradient_shap={"n_samples": shap_samples, "batch_size": args.shap_batch},
    )
    if args.lime:
        methods["lime3d"] = build_method("lime3d", model, device)

    log.info("ensemble built: %s", ", ".join(methods))
    art = artifacts_dir(cfg)

    # ---- synthetic signal sanity ---------------------------------------
    if not args.skip_sanity:
        log.info("synthetic sanity check over %d cases...", args.sanity_cases)
        # Shape comes from the config, not a constant: a model built for a
        # different out_shape would reject a hardcoded 128^3 volume.
        sanity = synthetic_sanity(model, device, methods, label_names,
                                  args.sanity_cases, cfg.seed,
                                  shape=(model_img_size(cfg),) * 3)
        sanity.to_csv(art / "xai_sanity.csv", index=False)

        summary = sanity.groupby("method")["enrichment"].agg(["mean", "std", "min"])
        print("\n" + "=" * 74)
        print("SYNTHETIC SIGNAL SANITY — saliency mass on the planted region")
        print("enrichment = observed mass / mass expected by chance;  <= 1.0 means no better than random")
        print("=" * 74)
        print(summary.round(3).to_string())
        failed = summary[summary["mean"] <= 1.0].index.tolist()
        if failed:
            print(f"\n!! FAILS SANITY (enrichment <= 1.0): {failed}")
            print("   Report this rather than proceeding as though the method works.")

    # ---- IG completeness ------------------------------------------------
    cases = load_case_set(cfg, "test", primary_dataset(cfg), fold=fold)
    ids = cases.ids
    if not ids:
        raise SystemExit("no cached test cases found")
    volume = cases.load(ids[0], device)
    log.info("explaining case %s (%d in fold-%s test split)", ids[0], len(ids), fold)

    if "integrated_gradients" in methods:
        errors = []
        for label in range(len(label_names)):
            t0 = time.perf_counter()
            methods["integrated_gradients"].attribute(volume, label)
            log.info("IG completeness %d/%d (%s): %.1fs",
                     label + 1, len(label_names), label_names[label], time.perf_counter() - t0)
            errors.append({
                "target_label": label_names[label],
                **methods["integrated_gradients"].last_completeness,
            })
        ig_df = pd.DataFrame(errors)
        ig_df.to_csv(art / "xai_ig_completeness.csv", index=False)
        print("\n" + "=" * 74)
        print("INTEGRATED GRADIENTS — completeness axiom:  sum(IG) == F(x) - F(x')")
        print("=" * 74)
        print(ig_df.round(5).to_string(index=False))
        worst = ig_df["relative_error"].max()
        print(f"\nworst relative error: {worst:.4f} "
              f"({'OK' if worst < 0.05 else 'FAILS the 5% threshold — implementation is wrong'})")

    if "gradient_shap" in methods:
        methods["gradient_shap"].attribute(volume, 0)
        print(f"\nGradientSHAP relative standard error: {methods['gradient_shap'].last_variance:.4f} "
              f"(high values mean the estimate needs more samples)")

    # ---- runtime --------------------------------------------------------
    log.info("benchmarking runtime over %d methods...", len(methods))
    runtime = benchmark(model, device, methods, volume, target_label=0)
    runtime.to_csv(art / "xai_runtime.csv", index=False)
    print("\n" + "=" * 74)
    print("MEASURED WALL-CLOCK PER METHOD PER VOLUME")
    print("=" * 74)
    print(runtime.round(4).to_string(index=False))
    print(f"\nwrote {art / 'xai_runtime.csv'}")


if __name__ == "__main__":
    main()
