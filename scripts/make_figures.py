"""Tri-planar overlay figures and the RESULTS.md scaffold.

    python scripts/make_figures.py --checkpoint artifacts/runs/vit3d_vit3d/best.pt

Generates one figure per selected case showing all four methods plus the fused
map, labelled with predictions, calibrated confidences, and faithfulness scores.

Cases are selected to be the ones a guide or examiner will actually ask about:
highest confidence, lowest confidence, correct vs incorrect, and cases positive
for the sparse auxiliary labels recorded by the label builder (nerve_proximity,
bone_quality) — those are the clinically interesting ones.
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
from src.train.targets import TargetSpec  # noqa: E402
from src.utils.config import artifacts_dir, load_config  # noqa: E402
from src.utils.log import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402
from src.xai import ENSEMBLE_METHODS, build_ensemble  # noqa: E402
from src.xai.adaptive import EVAL_METRIC, WEIGHT_METRIC, fuse  # noqa: E402
from src.xai.base import make_baseline  # noqa: E402
from src.xai.calibration import apply_temperature  # noqa: E402
from src.xai.runner import (
    explanation_target,  # noqa: E402
    load_case_set,
    load_model,
    predict_case_logits,
    require_prerequisites,
    resolve_fold,
    training_baselines,
)
from src.xai.visualize import method_comparison_figure  # noqa: E402

log = get_logger("figures")


def select_cases(ids, y, probs, aux: pd.DataFrame, per_group: int = 2) -> dict[str, list[str]]:
    """Pick the cases worth showing, grouped by why they are interesting."""
    confidence = probs.max(axis=1)
    predicted = (probs >= 0.5).astype(int)
    correct = (predicted == y).all(axis=1)

    order = np.argsort(-confidence)
    groups: dict[str, list[str]] = {
        "highest_confidence": [ids[i] for i in order[:per_group]],
        "lowest_confidence": [ids[i] for i in order[-per_group:]],
        "correct": [ids[i] for i in np.flatnonzero(correct)[:per_group]],
        "incorrect": [ids[i] for i in np.flatnonzero(~correct)[:per_group]],
    }

    # Auxiliary labels: recorded by the label builder, never trained on.
    if aux is not None and not aux.empty:
        indexed = aux.set_index("patient_id")
        for label in ("nerve_proximity", "bone_quality"):
            if label not in indexed.columns:
                continue
            positives = [p for p in ids if int(indexed.loc[p, label]) == 1] if len(indexed) else []
            if positives:
                groups[f"aux_{label}"] = positives[:per_group]
    return groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--fold", type=int, default=None,
                    help="cross-validation round; inferred from a cv_foldK checkpoint path")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--deterministic", action="store_true",
                    help="pin cuDNN to deterministic kernels: ~100x slower for "
                         "attribution and still non-deterministic in attention")
    ap.add_argument("--per-group", dest="per_group", type=int, default=2)
    ap.add_argument("--steps", type=int, default=50)
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
    figures = art / "figures" / "cases"

    primary = primary_dataset(cfg)
    cases = load_case_set(cfg, "test", primary, fold=fold)
    ids, y, label_names = cases.ids, cases.y, cases.labels
    logits = predict_case_logits(model, cases, device)

    calib_path = art / "calibration" / "calibration.json"
    temperature = 1.0
    if calib_path.exists():
        temperature = json.loads(calib_path.read_text(encoding="utf-8"))["temperature"]
        log.info("using calibrated temperature T=%.4f", temperature)
    else:
        log.warning("no calibration.json — showing UNcalibrated confidences; run run_adaptive.py first")
    probs = apply_temperature(logits, temperature)

    # Auxiliary columns are optional context for case selection -- computed by the
    # label builder but never trained on. Whatever extra columns the cohort's CSV
    # carries beyond the trained labels are used; a cohort with none still works.
    aux_csv = art / (getattr(cfg.task, "sites_csv", None) or f"labels_{primary}.csv")
    aux = pd.read_csv(aux_csv, dtype={"patient_id": str})
    aux = aux[[c for c in aux.columns
               if c == "patient_id" or (c not in label_names and not c.startswith("ev_"))]]

    groups = select_cases(ids, y, probs, aux, args.per_group)
    methods = build_ensemble(model, device, names=tuple(args.methods),
                             mean_volume=mean_volume, baselines=baselines)

    index = {p: i for i, p in enumerate(ids)}
    manifest = []

    for group, members in groups.items():
        for pid in members:
            i = index[pid]
            volume = cases.load(pid, device)
            baseline = make_baseline(volume, "blur")
            target = explanation_target(cfg, spec, probs[i])

            maps = {n: m.attribute(volume, target) for n, m in methods.items()}
            result = fuse(model, volume, maps, target, weight_metric=WEIGHT_METRIC,
                          eval_metric=EVAL_METRIC, steps=args.steps, baseline=baseline)
            maps["fused"] = result["fused_map"]

            scores = dict(result["per_method_eval"])
            scores["fused"] = result["fused_eval"]

            predicted = [label_names[j] for j in range(len(label_names)) if probs[i, j] >= 0.5]
            truth = [label_names[j] for j in range(len(label_names)) if y[i, j] == 1]
            # A case on the site task is a tooth position, not a whole patient.
            unit = "site" if cases.is_sites else "patient"
            title = (
                f"{group.replace('_', ' ')} — {unit} {pid} — explaining '{label_names[target]}'\n"
                f"predicted: {', '.join(predicted) or 'none'}  |  true: {', '.join(truth) or 'none'}\n"
                f"calibrated p={probs[i, target]:.3f}   column headers show {EVAL_METRIC} (lower is better)"
            )

            path = figures / f"{group}_{pid}.png"
            method_comparison_figure(volume, maps, path, title=title, scores=scores)
            manifest.append({
                "group": group, "patient_id": pid, "target_label": label_names[target],
                "calibrated_prob": float(probs[i, target]),
                "predicted": "|".join(predicted), "true": "|".join(truth),
                "fused_eval": result["fused_eval"], "figure": str(path),
            })
            log.info("wrote %s", path)

    pd.DataFrame(manifest).to_csv(art / "figures" / "case_manifest.csv", index=False)
    write_results_scaffold(art, temperature)
    print(f"\nwrote {len(manifest)} case figures to {figures}")


def write_results_scaffold(art: Path, temperature: float) -> None:
    """RESULTS.md skeleton wired to the CSVs the code generated.

    Deliberately not pre-filled with numbers: every claim must trace to a value
    in a CSV this pipeline produced, so the narrative is written after reading
    them, never before.
    """
    path = art / "RESULTS.md"
    if path.exists():
        log.info("%s already exists — leaving it alone", path)
        return

    available = sorted(p.name for p in art.glob("*.csv"))
    path.write_text(f"""# Results

> Every claim below must trace to a number in one of the generated CSVs.
> Negative results are findings and stay in.

## Source data
{chr(10).join(f'- `{name}`' for name in available)}
- `calibration/calibration.json` (temperature T = {temperature:.4f})
- `figures/` — tri-planar overlays, curves, agreement matrices, Pareto curve

## 1. Classification (`results_classification.csv`)
Primary test split vs external cohort zero-shot, per label, with bootstrap 95% CIs.
State the mean AUROC shift and discuss it — do not explain it away.

## 2. XAI methods (`xai_sanity.csv`, `xai_ig_completeness.csv`, `xai_runtime.csv`)
- Synthetic planted-signal enrichment per method. Anything <= 1.0 is no better than chance.
- IG completeness relative error (must be < 5%).
- Measured wall-clock per method — the adaptive layer's justification.
- Note that plain attention rollout is class-agnostic; compare against grad_rollout.

## 3. Faithfulness (`results_faithfulness.csv`, `results_randomization.csv`)
- Deletion (lower better) / insertion (higher better) AUC per method.
- Model-randomisation: name any method whose map survives randomisation — it is an
  edge detector, not an explanation.
- Bone-mass proxy: state explicitly that it is intensity-threshold derived, not
  clinical ground truth, and cannot identify *which* anatomy.

## 4. Adaptive layer (`results_ablations.csv`, `results_pareto.csv`)
Weighted BY `{WEIGHT_METRIC}`, evaluated ON `{EVAL_METRIC}` — never the same metric.
- Claim 1: fusion beats every individual method on the held-out metric?
- Claim 2: agreement-weighted beats uniform averaging?
- Claim 3: Pareto curve dominates both fixed policies?
Report each as supported or not supported, with the fraction of cases.

## 5. Limitations
- Labels are rule-derived from free text, not clinical annotation.
- Attribution is scored by deletion/insertion, not by overlap with anatomy.
- n is small (~60 test patients); CIs are wide and single-label claims are weak.
""", encoding="utf-8")
    log.info("wrote %s", path)


if __name__ == "__main__":
    main()
