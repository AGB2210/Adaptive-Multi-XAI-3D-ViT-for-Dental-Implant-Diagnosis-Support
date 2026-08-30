"""Score the hand-crafted geometric estimator on the same split as the model.

    python scripts/run_geometric_baseline.py --config configs/sites.yaml --fold 0

CPU only, no checkpoint, no GPU. It reads the patches through `CaseSet.load` --
the model's own input path, so the same `patch_centre`, the same `cut_patch` and
the same 96^3 box, with no second implementation that could drift from it -- and
measures crest-to-canal height and bucco-lingual width from intensity alone. It
never opens a segmentation mask; see `src/models/geometric.py` for why that
distinction is the whole point.

Reported through the SAME functions as the model: `regression_metrics` against
the same floors, and `threshold_sensitivity` across the same rule sweep. The
two tables are therefore directly comparable, which is the only reason this is
worth running.

Read the result this way:

  * geometric MAE far above the model's -> the transformer is buying something
    a ruler cannot, and the architecture argument stands.
  * geometric MAE near the model's -> the task is geometry and the model is an
    expensive way to do it. Report that. It is a real finding and it is better
    found here than in a viva.
  * geometric agreement near the model's on the threshold sweep -> the same
    conclusion, in the units a clinician reads.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.taskdef import primary_dataset, regression_names_for  # noqa: E402
from src.models.geometric import measure_patch  # noqa: E402
from src.train.targets import (  # noqa: E402
    format_regression,
    no_information_regression,
    regression_metrics,
    threshold_sensitivity,
)
from src.utils.config import artifacts_dir, load_config  # noqa: E402
from src.utils.log import get_logger  # noqa: E402
from src.xai.runner import load_case_set  # noqa: E402

log = get_logger("geometric")

NATIVE_SPACING_MM = 0.3   # ToothFairy3 is isotropic 0.3 mm; nothing is resampled


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sites.yaml")
    ap.add_argument("--fold", type=int, default=None,
                    help="cross-validation round whose split to score on")
    ap.add_argument("--split", default="val",
                    help="val by default. The test split is scored ONCE, after "
                         "every fold is trained -- do not spend it here")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    fold = args.fold
    cases = load_case_set(cfg, args.split, primary_dataset(cfg), fold=fold)
    if not cases.is_sites:
        raise SystemExit("this baseline measures tooth sites; point it at a site config")

    mm_names = regression_names_for(cfg)
    if not mm_names:
        raise SystemExit("the config declares no millimetre targets to measure against")

    # `cfg.preprocess.spacing` does not exist in any config -- this line raised
    # AttributeError on the exact invocation RUNBOOK 4e prescribes, so this
    # baseline had never once run. The nearby `target_spacing` is the wrong
    # substitute: it is null on sites.yaml (meaning "do not resample", i.e.
    # native) and 1.0 on default.yaml, either of which would scale every
    # millimetre this script reports by 3.33x and then compare it against a
    # model whose millimetres are real.
    target = getattr(cfg.preprocess, "target_spacing", None)
    if target is None:
        spacing = (NATIVE_SPACING_MM,) * 3          # nothing is resampled
    else:
        spacing = (float(target),) * 3
    log.info("spacing %s mm/voxel -- every millimetre below depends on this",
             spacing)
    name_index = {n: i for i, n in enumerate(cases.labels)}
    cpu = torch.device("cpu")

    log.info("%d %s sites, fold %s, spacing %s, patch %s",
             len(cases.ids), args.split, fold, spacing, cases.patch_size)

    pred = np.full((len(cases.ids), len(mm_names)), np.nan, dtype=np.float64)
    limiters: dict[str, int] = {}
    for i, case_id in enumerate(cases.ids):
        row = cases.row(case_id)
        # `cases.load` is the model's own input path. Re-deriving the patch here
        # would be a second implementation of `patch_centre` that could drift
        # from the one training used, and the whole claim of this baseline is
        # that it sees exactly what the model sees.
        patch = cases.load(case_id, cpu)[0, 0].numpy()

        got = measure_patch(patch, str(row.get("jaw", "lower")), spacing)
        for j, name in enumerate(mm_names):
            pred[i, j] = got.get(name, np.nan)
        limiters[got["limiter"]] = limiters.get(got["limiter"], 0) + 1
        if (i + 1) % 500 == 0:
            log.info("%d/%d sites", i + 1, len(cases.ids))

    true = np.stack([cases.y[:, name_index[n]] for n in mm_names], axis=1).astype(np.float64)

    measured = regression_metrics(true, pred, mm_names)
    floors = no_information_regression(true, mm_names)
    print("\n" + format_regression(
        measured, f"GEOMETRIC BASELINE -- {args.split} fold {fold} (n={len(cases.ids)})"))

    print("\nAgainst the no-information floor, and against nothing else:")
    for name in mm_names:
        m, f = measured[name], floors[name]
        ratio = m["mae"] / f["mae"] if f["mae"] else float("nan")
        verdict = "beats the floor" if ratio < 1 else "NO BETTER THAN THE MEDIAN"
        print(f"   {name:22} MAE {m['mae']:7.3f}   floor {f['mae']:7.3f}   "
              f"{ratio:5.2f}x   {verdict}")

    print("\nWhat limited each site (this is a measurement, not a guess):")
    for k in sorted(limiters, key=lambda k: -limiters[k]):
        print(f"   {k:14} {limiters[k]:6d}   {limiters[k] / len(cases.ids):6.1%}")
    unmeasurable = int(np.isnan(pred).any(axis=1).sum())
    print(f"   {'unmeasured':14} {unmeasurable:6d}   {unmeasurable / len(cases.ids):6.1%}"
          "   <- reported as NaN rather than as a plausible number")

    rules = dict(getattr(cfg.task, "rules", {}) or {})
    sweep = []
    if rules:
        sweep = threshold_sensitivity(true, pred, mm_names, rules)
        print("\nFeasibility agreement across the rule sweep -- compare row for row "
              "with the model's own table:")
        print(f"   {'height rule':>12}  {'measured':>10}  {'predicted':>10}  {'agreement':>10}")
        for r in sweep:
            print(f"   {r['height_threshold_mm']:>9.1f} mm  "
                  f"{r['measured_feasible_rate']:>10.4f}  "
                  f"{r['predicted_feasible_rate']:>10.4f}  {r['agreement']:>10.4f}")

    art = artifacts_dir(cfg)
    out = Path(args.out or art / f"results_geometric_baseline_fold{fold}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "split": args.split, "fold": fold, "n": len(cases.ids),
        "targets": list(mm_names),
        "regression": measured, "floors": floors,
        "limiters": limiters, "unmeasured": unmeasurable,
        "threshold_sensitivity": sweep,
        "reads": "cached CT intensities only -- never a segmentation mask",
    }, indent=2), encoding="utf-8")
    log.info("wrote %s", out)

    print("\nThis estimator is untuned on purpose. If it needs tuning to compete, "
          "\nthat is itself a finding about how much of the task is geometry.")


if __name__ == "__main__":
    main()
