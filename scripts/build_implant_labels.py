"""Derive per-site implant need and feasibility labels from ToothFairy3 masks.

    python scripts/build_implant_labels.py --config configs/sites.yaml

Writes <artifacts_dir>/sites_<dataset>.csv: one row per scan per tooth position, 28
positions per scan (third molars excluded), so ~14,900 rows from 532 scans.

WHAT EACH ROW ANSWERS

    needs_implant   nothing occupies this position -- no tooth, no crown, no
                    bridge pontic, no existing implant
    feasible        there is enough bone to place one safely

MILLIMETRES ARE STORED, NOT JUST THE VERDICT. Every row carries the measured
available height, ridge width, and which structure limited it, alongside the
threshold it was judged against. A clinician revising a threshold re-runs
`rescore_sites` over this CSV in seconds instead of reprocessing 28 GB of scans.
That separation is the whole reason the thresholds are not hard-coded anywhere
in the measurement path.

THE LABELS ARE ONLY AS GOOD AS THE RULES. The defaults come from published
guidance -- a 2 mm safety zone above the inferior alveolar canal, 10 mm of
residual bone below the maxillary sinus, a 6 mm ridge for a conventional
3.75-4 mm implant -- but they are defaults, not clinical sign-off. Read the
prevalence summary this prints before trusting anything downstream of it.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dental_arch import ARCH, site_positions, within_volume  # noqa: E402
from src.data.implant_sites import (  # noqa: E402
    feasibility,
    measure_site,
    prepare_mask,
    site_is_occupied,
    tooth_centroids,
)
from src.data.taskdef import primary_dataset  # noqa: E402
from src.utils.config import (  # noqa: E402
    artifacts_dir,
    dataset_root,
    label_path,
    list_cases,
    load_config,
)
from src.utils.log import get_logger  # noqa: E402

log = get_logger("implant_labels")


MAX_SKIP_FRACTION = 0.02


def rules_from(cfg, args) -> dict:
    """Thresholds, config first and command line on top. Never silently defaulted."""
    site = getattr(cfg, "sites", None)

    def pick(name):
        override = getattr(args, name, None)
        if override is not None:
            return float(override)
        value = getattr(site, name, None) if site is not None else None
        if value is None:
            raise SystemExit(
                f"sites.{name} is not set in the config.\n"
                "These are clinical thresholds. Set them explicitly rather than "
                "inheriting a default nobody has reviewed."
            )
        return float(value)

    return {
        "min_height_mandible_mm": pick("min_height_mandible_mm"),
        "min_height_maxilla_mm": pick("min_height_maxilla_mm"),
        "min_width_mm": pick("min_width_mm"),
    }


def score_one(mask, spacing, rules: dict) -> list[dict]:
    """Every site in one scan."""
    oriented, sign, stats = prepare_mask(mask)
    centroids = tooth_centroids(oriented, stats)
    positions = site_positions(oriented, centroids)

    rows = []
    for jaw in ("upper", "lower"):
        for tooth in ARCH[jaw]:
            info = positions[tooth]
            xy = info["xy"]
            row = {
                "tooth": tooth,
                "jaw": jaw,
                "site_method": info["method"],
                "site_anchors": info["anchors"],
                "orientation_sign": sign,
                # Voxel coordinates in the ORIENTED volume, so the training
                # patch can be cut without re-reading the mask. Without these
                # every epoch would have to redo the arch fit.
                "site_x": float(xy[0]) if xy else np.nan,
                "site_y": float(xy[1]) if xy else np.nan,
                "spacing_z_mm": float(spacing[2]),
            }

            if not within_volume(info["xy"], oriented.shape):
                # Extrapolating past the teeth that defined the arch can land
                # outside the field of view. That is not a site with no bone; it
                # is a site we never saw.
                row.update({"needs_implant": pd.NA, "feasible": pd.NA,
                            "occupied_by": "", "reason": "outside_volume",
                            "available_height_mm": np.nan, "ridge_width_mm": np.nan,
                            "limiting_structure": "", "crest_mm": np.nan,
                            "n_bone_voxels": 0, "height_ok": pd.NA, "width_ok": pd.NA,
                            "required_height_mm": np.nan, "required_width_mm": np.nan,
                            "site_z": np.nan, "occupying_tooth": 0})
                rows.append(row)
                continue

            centre = (info["xy"][0], info["xy"][1], 0)
            occ = site_is_occupied(oriented, centre, spacing, centroids, tooth=tooth)
            m = measure_site(oriented, centre, jaw, spacing)
            verdict = feasibility(m, **rules)

            row.update(m.as_row())
            row.update(verdict)
            # The patch is cut around the ridge, not around whatever z the arch
            # fit happened to carry, so the box straddles crest and canal.
            row["site_z"] = (float(m.crest_mm) / float(spacing[2])
                             if np.isfinite(m.crest_mm) else np.nan)
            row["needs_implant"] = int(not occ["occupied"])
            row["occupied_by"] = occ["by"]
            row["occupying_tooth"] = occ["tooth_id"]
            # Feasibility is measured everywhere, but it only MEANS anything
            # where an implant is actually wanted. Keeping the measurement on
            # occupied rows costs nothing and lets the rules be re-scored later.
            rows.append(row)
    return rows


def summarise(df: pd.DataFrame) -> None:
    total = len(df)
    print("\n" + "=" * 72)
    print("SITE PREVALENCE -- read this before trusting anything downstream")
    print("=" * 72)
    print(f"rows: {total}  ({df.patient_id.nunique()} scans x 28 sites)")

    print("\nhow each site position was located")
    for method, n in df.site_method.value_counts().items():
        print(f"  {method:<14} {n:>6}  ({100 * n / total:5.1f}%)")

    usable = df[df.needs_implant.notna()]
    needs = usable[usable.needs_implant == 1]
    print(f"\nsites needing an implant : {len(needs):>6}  "
          f"({100 * len(needs) / max(len(usable), 1):5.1f}% of measurable sites)")

    if len(needs):
        feas = int((needs.feasible == True).sum())  # noqa: E712
        infeas = len(needs) - feas
        print(f"  of those, feasible     : {feas:>6}  ({100 * feas / len(needs):5.1f}%)")
        print(f"  of those, NOT feasible : {infeas:>6}  ({100 * infeas / len(needs):5.1f}%)")
        print("\n  why not feasible")
        for reason, n in Counter(needs[needs.feasible != True].reason).most_common():  # noqa: E712
            print(f"    {reason:<22} {n:>6}")

        share = 100 * infeas / max(len(usable), 1)
        print()
        if share < 2.0:
            print(f"!! 'needs an implant but not feasible' is {share:.2f}% of all sites.")
            print("   That is the clinically interesting class and it is very rare.")
            print("   Consider predicting available height in mm as a regression")
            print("   target instead of a yes/no, before spending GPU time on this.")
        else:
            print(f"the interesting class is {share:.1f}% of all sites -- workable.")

    print("\nwhat occupies the sites that do NOT need an implant")
    for by, n in usable[usable.needs_implant == 0].occupied_by.value_counts().items():
        print(f"  {by or '(none)':<14} {n:>6}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sites.yaml")
    ap.add_argument("--dataset", default=None, help="defaults to data.primary")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke-test only: the first N cases ALPHABETICALLY, which is not a sample -- ToothFairy3's prefixes group by sub-cohort, so the first 14 are all ToothFairy3F and averaged 104.9 MB against 48 MB cohort-wide. Extrapolating from that is what produced the retracted 52 GB estimate")
    ap.add_argument("--min-height-mandible-mm", dest="min_height_mandible_mm",
                    type=float, default=None, help="overrides sites.* in the config")
    ap.add_argument("--min-height-maxilla-mm", dest="min_height_maxilla_mm",
                    type=float, default=None)
    ap.add_argument("--min-width-mm", dest="min_width_mm", type=float, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    dataset = args.dataset or primary_dataset(cfg)
    rules = rules_from(cfg, args)

    root = dataset_root(cfg, dataset)
    if not root.is_dir():
        raise SystemExit(f"dataset root not found: {root.resolve()}")

    cases = list_cases(cfg, dataset)
    if args.limit:
        cases = cases[: args.limit]
    log.info("%s: %d scans | rules %s", dataset, len(cases), rules)

    rows, skipped = [], 0
    for k, pid in enumerate(cases, 1):
        path = label_path(cfg, dataset, pid)
        if path is None or not path.is_file():
            skipped += 1
            log.warning("no mask for %s -- skipped", pid)
            continue

        img = nib.load(str(path))
        spacing = np.asarray(img.header.get_zooms()[:3], dtype=float)
        try:
            scan_rows = score_one(np.asarray(img.dataobj), spacing, rules)
        except Exception as exc:  # noqa: BLE001 - one bad scan must not lose the run
            skipped += 1
            log.error("%s: %s -- skipped", pid, exc)
            continue

        for row in scan_rows:
            row["patient_id"] = pid
        rows.extend(scan_rows)
        if k % 25 == 0:
            log.info("  %d/%d scans", k, len(cases))

    if not rows:
        raise SystemExit("no scan produced any site")

    df = pd.DataFrame(rows)
    front = ["patient_id", "tooth", "jaw", "needs_implant", "feasible",
             "available_height_mm", "ridge_width_mm", "limiting_structure", "reason",
             "site_x", "site_y", "site_z", "crest_mm", "site_method"]
    df = df[front + [c for c in df.columns if c not in front]]

    out = artifacts_dir(cfg) / f"sites_{dataset}.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    log.info("wrote %s (%d rows, %d scans skipped)", out, len(df), skipped)

    # A skip is a scan we could not measure, and a build that skips most of the
    # cohort still exits 0 with a plausible-looking CSV. That is not theoretical:
    # a refactor once made ridge_width return a bare float where a tuple was
    # expected, every scan raised, and the run reported success over 141 rows.
    # Past a few percent this is a broken build, not a few awkward scans.
    seen = len(df.patient_id.unique()) + skipped
    if seen and skipped / seen > MAX_SKIP_FRACTION:
        raise SystemExit(
            f"{skipped}/{seen} scans skipped ({skipped / seen:.1%}) -- over the "
            f"{MAX_SKIP_FRACTION:.0%} ceiling. The CSV was written; do not trust it. "
            f"Read the ERROR lines above: a systematic fault looks exactly like this."
        )
    summarise(df)


if __name__ == "__main__":
    main()
