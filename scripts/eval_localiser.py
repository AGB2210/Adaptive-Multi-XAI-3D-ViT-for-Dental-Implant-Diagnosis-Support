"""Measure the site localiser on its held-out fold, alone and end to end.

    python scripts/eval_localiser.py --config configs/localiser.yaml \
        --checkpoint artifacts_sites/localiser_runs/cv_fold0/best.pt \
        [--site-checkpoint artifacts_sites/runs/cv_fold0/best.pt]

Two questions, and the second is the one that matters:

  1. HOW FAR OFF ARE THE POSITIONS? Error in mm against the label builder's own
     site positions, on the test patients of the checkpoint's fold: in-plane,
     vertical and 3D, per tooth and per position tier, with the median's
     interval clustered by patient. Orientation accuracy, both ways up.

  2. WHAT DOES THAT COST THE MEASUREMENTS? With --site-checkpoint, the site
     model predicts height and width twice for every test site -- once from the
     patch the mask puts there, once from the patch the localiser puts there --
     and both are scored against the measured truth. The difference is the
     price of running without a segmentation, in the units a clinician reads.
     The two runs see the same patients, the same model and the same truth, so
     nothing else differs.

The localiser's position labels for the weaker tiers are themselves
extrapolations; error measured against them is error against an estimate, and
the per-tier table says which rows that applies to.
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

from src.data.site_dataset import cut_patch, load_sites, patch_centre  # noqa: E402
from src.data.splits import fold_assignment, load_folds  # noqa: E402
from src.data.taskdef import all_target_names, primary_dataset  # noqa: E402
from src.inference.checkpoint import load_bundle  # noqa: E402
from src.models.localiser import is_located, load_localiser, to_full_coords  # noqa: E402
from src.train.localiser import (  # noqa: E402
    localiser_datasets,
    predict_dataset,
    site_errors,
    summarise,
)
from src.train.targets import derived_feasible, to_report_units  # noqa: E402
from src.utils.config import artifacts_dir, cache_dir, load_config  # noqa: E402
from src.utils.log import get_logger  # noqa: E402
from src.xai.runner import clustered_ci  # noqa: E402

log = get_logger("eval_localiser")
HEIGHT, WIDTH = "available_height_mm", "ridge_width_mm"


def per_site_frame(pred: dict, teeth, sites: pd.DataFrame, factor: int, spacing: float) -> pd.DataFrame:
    err = site_errors(pred, factor, spacing)
    method = sites[sites.jaw == "lower"].set_index(["patient_id", "tooth"])["site_method"]
    rows = []
    for p, pid in enumerate(pred["patients"]):
        for s, tooth in enumerate(teeth):
            rows.append({"patient_id": pid, "tooth": int(tooth),
                         "method": method.get((pid, int(tooth)), "none"),
                         "error_mm": err["error_mm"][p, s], "xy_mm": err["xy_mm"][p, s],
                         "z_mm": err["z_mm"][p, s], "spread_mm": pred["spread"][p, s] * factor * spacing})
    return pd.DataFrame(rows)


@torch.no_grad()
def end_to_end(cfg, bundle, localiser, data, sites, device) -> pd.DataFrame:
    """Site-model outputs from mask-placed and localiser-placed patches, per test site."""
    targets = all_target_names(cfg)
    usable = load_sites(artifacts_dir(cfg) / cfg.task.sites_csv, targets=targets,
                        methods=tuple(cfg.task.site_methods), jaws=tuple(cfg.task.site_jaws))
    full_cache = cache_dir(cfg, primary_dataset(cfg))
    manifest = pd.read_csv(Path(cfg.localiser.cache_dir) / "manifest.csv",
                           dtype={"patient_id": str}).set_index("patient_id")
    size = bundle.img_size
    rows = []
    for i, pid in enumerate(data.patients):
        vol_path = full_cache / f"{pid}.npy"
        here = usable[usable.patient_id == pid]
        if here.empty or not vol_path.is_file():
            continue
        volume = np.load(vol_path, mmap_mode="r")
        x, *_ = data[i]
        out = localiser.model(x[None].to(device))
        offset = np.array([manifest.loc[pid, f"offset_{a}"] for a in "xyz"])
        located = to_full_coords(out["coords"][0].float().cpu().numpy(), offset, localiser.cfg.factor)
        in_view = torch.sigmoid(out["valid_logit"][0]).float().cpu().numpy()
        index = {t: k for k, t in enumerate(localiser.cfg.sites)}

        for r in here.itertuples():
            if int(r.tooth) not in index:
                continue
            lx, ly, lz = located[index[int(r.tooth)]]
            patches = []
            for row in ({"site_x": r.site_x, "site_y": r.site_y, "site_z": r.site_z, "jaw": "lower"},
                        {"site_x": lx, "site_y": ly, "site_z": lz, "jaw": "lower"}):
                patches.append(np.asarray(cut_patch(volume, patch_centre(row, volume.shape, size), size),
                                          dtype=np.float32))
            batch = torch.from_numpy(np.stack(patches))[:, None].to(device)
            rep = to_report_units(bundle.model(batch).float().cpu().numpy(), bundle.spec)
            names = bundle.names
            rows.append({
                "patient_id": pid, "tooth": int(r.tooth),
                "true_height": r.available_height_mm, "true_width": r.ridge_width_mm,
                "mask_height": rep[0, names.index(HEIGHT)], "mask_width": rep[0, names.index(WIDTH)],
                "loc_height": rep[1, names.index(HEIGHT)], "loc_width": rep[1, names.index(WIDTH)],
                "position_error_mm": float(np.linalg.norm(
                    (np.array([lx, ly, lz]) - np.array([r.site_x, r.site_y, r.site_z]))
                    * localiser.cfg.spacing_mm)),
                # Whether the app would cut a patch here at all, by its own rule.
                "in_view_prob": float(in_view[index[int(r.tooth)]]),
                "predicted": is_located(in_view[index[int(r.tooth)]], (lx, ly, lz), volume.shape),
            })
        log.info("%d/%d patients", i + 1, len(data.patients))
    return pd.DataFrame(rows)


def end_to_end_report(e2e: pd.DataFrame, rules: dict) -> dict:
    """Height / width MAE and feasibility agreement, mask-placed against localiser-placed.

    Reported twice. Over EVERY test site, which is the localiser's positions
    taken at face value; and over the sites the app would actually predict --
    in view by the localiser's own head, and inside the volume -- with the
    mask-placed patches scored on that same subset so the two rows compare
    like with like. `coverage` is the share of sites in the second group: the
    rest the app shows as "no position", and an error averaged over them was
    never going to reach a clinician.
    """
    names = [HEIGHT, WIDTH]

    def score(frame: pd.DataFrame) -> dict:
        truth = frame[["true_height", "true_width"]].to_numpy(dtype=float)
        feas_true = derived_feasible(truth, names, rules)
        out = {}
        for tag in ("mask", "loc"):
            pred = frame[[f"{tag}_height", f"{tag}_width"]].to_numpy(dtype=float)
            feas = derived_feasible(pred, names, rules)
            seen = np.isfinite(feas) & np.isfinite(feas_true)
            out[tag] = {
                "height_mae_mm": float(np.nanmean(np.abs(pred[:, 0] - truth[:, 0]))) if len(frame) else float("nan"),
                "width_mae_mm": float(np.nanmean(np.abs(pred[:, 1] - truth[:, 1]))) if len(frame) else float("nan"),
                "feasibility_agreement": float((feas[seen] == feas_true[seen]).mean()) if seen.any() else float("nan"),
            }
        return out

    kept = e2e[e2e.predicted.astype(bool)]
    return {"all_sites": score(e2e), "predicted_sites": score(kept),
            "n_sites": int(len(e2e)), "n_predicted": int(len(kept)),
            "coverage": float(len(kept) / len(e2e)) if len(e2e) else float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/localiser.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--site-checkpoint", dest="site_checkpoint", default=None,
                    help="a site model of the SAME fold, for the end-to-end cost")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    localiser = load_localiser(args.checkpoint, device, allow_unsafe=False)
    fold = localiser.meta.get("fold")
    if fold is None:
        raise SystemExit("the checkpoint records no fold, so its held-out patients are unknown")
    lcfg, data = localiser_datasets(cfg, int(fold))
    data = data[args.split]
    sites = pd.read_csv(artifacts_dir(cfg) / cfg.task.sites_csv, dtype={"patient_id": str})
    split = fold_assignment(load_folds(artifacts_dir(cfg) / "cv_folds.json"), int(fold))
    assert set(data.patients) <= set(split[args.split])

    upright = predict_dataset(localiser.model, data, device)
    flipped = predict_dataset(localiser.model, data, device, flip=True)
    summary = summarise(upright, lcfg.factor, lcfg.spacing_mm)
    summary["orientation_accuracy"] = float(np.mean(
        np.concatenate([upright["flip_prob"] < 0.5, flipped["flip_prob"] >= 0.5])))
    frame = per_site_frame(upright, lcfg.sites, sites, lcfg.factor, lcfg.spacing_mm)
    point, lo, hi = clustered_ci(frame, "error_mm")
    summary["median_error_ci_mm"] = [lo, hi]

    print("\n" + "=" * 78)
    print(f"SITE LOCALISER -- fold {fold} {args.split}, {summary['n_patients']} patients, "
          f"{summary['n_sites_3d']} sites with all three coordinates labelled")
    print("=" * 78)
    print(f"3D error     median {point:.2f} mm  [{lo:.2f}, {hi:.2f}] (patient-clustered 95% CI)"
          f"   p90 {summary['p90_error_mm']:.2f} mm   within 3 mm {summary['within_3mm']:.1%}")
    print(f"in-plane     median {summary['median_xy_mm']:.2f} mm   vertical median {summary['median_z_mm']:.2f} mm")
    print(f"orientation  {summary['orientation_accuracy']:.1%} correct over both ways up")
    print(f"in view      {summary['in_view_accuracy']:.1%} correct")
    print("\nby position tier (labels for the weaker tiers are themselves extrapolations):")
    print(frame.groupby("method")["error_mm"].agg(["median", "count"]).round(2).to_string())
    print("\nby tooth:")
    print(frame.groupby("tooth")["error_mm"].median().round(2).to_string())

    result = {"fold": fold, "split": args.split, "localisation": summary,
              "by_tier": frame.groupby("method")["error_mm"].median().to_dict(),
              "by_tooth": {int(k): v for k, v in frame.groupby("tooth")["error_mm"].median().items()}}

    if args.site_checkpoint:
        bundle = load_bundle(args.site_checkpoint, device, fallback_config=args.config)
        e2e = end_to_end(cfg, bundle, localiser, data, sites, device)
        if e2e.empty:
            raise SystemExit(
                f"no {args.split} site could be scored end to end: none of the "
                f"{len(data.patients)} patients has both a full-resolution volume under "
                f"{cache_dir(cfg, primary_dataset(cfg))} and a usable site in "
                f"{cfg.task.sites_csv}. Build the site cache (RUNBOOK 4b) first.")
        rules = {k: float(v) for k, v in vars(cfg.sites).items() if k.startswith("min_")}
        report = end_to_end_report(e2e, rules)
        e2e["extra_height_err"] = (np.abs(e2e.loc_height - e2e.true_height)
                                   - np.abs(e2e.mask_height - e2e.true_height))
        kept = e2e[e2e.predicted.astype(bool)]
        print("\n" + "=" * 78)
        print(f"END TO END -- the same site model, mask-placed vs localiser-placed patches "
              f"({len(e2e)} sites, {e2e.patient_id.nunique()} patients)")
        print("=" * 78)
        print(f"the app would predict {report['n_predicted']} of {report['n_sites']} of these sites "
              f"({report['coverage']:.1%}); the rest it shows as 'no position'")
        print(f"\n{'':46}{'height MAE':>12}{'width MAE':>12}{'feasibility':>14}")
        for group, title in (("predicted_sites", "the sites the app predicts"),
                             ("all_sites", "every site, at face value")):
            for tag, label in (("mask", "mask-placed"), ("loc", "localiser-placed")):
                r = report[group][tag]
                print(f"{title + ', ' + label:46}{r['height_mae_mm']:>10.2f}mm"
                      f"{r['width_mae_mm']:>10.2f}mm{r['feasibility_agreement']:>13.1%}")
        extra = {}
        for name, frame in (("predicted_sites", kept), ("all_sites", e2e)):
            if frame.empty:
                continue
            _, dlo, dhi = clustered_ci(frame, "extra_height_err", stat=np.mean)
            extra[name] = {"mean_mm": float(frame.extra_height_err.mean()), "ci_mm": [dlo, dhi]}
            print(f"extra height error from localising, {name.replace('_', ' ')}: "
                  f"mean {extra[name]['mean_mm']:+.2f} mm [{dlo:+.2f}, {dhi:+.2f}] "
                  f"(patient-clustered 95% CI)")
        result["end_to_end"] = {**report, "extra_height_error": extra}
        e2e.to_csv(Path(args.checkpoint).parent / f"end_to_end_{args.split}.csv", index=False)

    out = Path(args.out or Path(args.checkpoint).parent / f"eval_{args.split}.json")
    out.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
