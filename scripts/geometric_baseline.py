"""Threshold-and-measure baseline: no learning, no GPU, no parameters to fit.

    python scripts/geometric_baseline.py --config configs/sites.yaml --fold 0

WHY THIS EXISTS, AND WHY IT IS THE MOST DANGEROUS SCRIPT IN THE REPOSITORY.

The model is trained to predict `available_height_mm` and `ridge_width_mm`, and
the clinical verdict is then a fixed rule applied to those two numbers. But the
ground truth for both is itself a geometric measurement taken from voxels --
crest to canal roof, and bucco-lingual thickness below the crest. So the obvious
question a reviewer asks first is:

    if the target is a measurement, why not just take the measurement?

This script takes it. It reads the same 96^3 patch the model reads, thresholds
for bone, finds the crest, finds the canal, and measures the same two distances
with no trained parameters at all. Then it applies the same clinical rule and
prints the same agreement table as `train.py`, so the two are directly
comparable line for line.

IF THIS SCRIPT MATCHES THE MODEL, THE ARCHITECTURE STORY IS OVER. That is the
point of running it, and it is better to find out before a cross-validation is
written around the model. A baseline that exists to be beaten is worth nothing;
this one is built to win if it can.

WHAT IT IS ALLOWED TO USE. Exactly what the model gets: the cached patch, cut by
the same `patch_centre` so the frames cannot drift. It does NOT read the
segmentation masks -- that would be measuring the ground truth against itself.
It does inherit one piece of information from the mask pipeline, and this must be
stated in the paper: the patch is CENTRED using `site_z`, which came from the
mask's crest. The model inherits that too, so the comparison is fair, but neither
is a deployable pipeline. `--find-crest` re-derives the crest from the image
instead, which is the honest variant and the one to quote if a reviewer presses.

NO TUNED THRESHOLDS. Bone is separated by Otsu, which has no free parameter, on
each patch independently. The alternative -- a fixed cut in z-scored units --
would be a knob, and a knob fitted on the same data the baseline is scored on
would make this a weak learner rather than a baseline.

STATUS: ONE CORRECTNESS FIX APPLIED, THEN STOPPED ON PURPOSE.

The first draft measured a median ridge of 5.10 mm against a truth of 11.78 mm.
The cause was a single Otsu threshold trying to separate two kinds of bone: it
cuts between dense cortical bone and everything else, so trabecular bone inside
the ridge fell below it and the width probe measured one cortical plate. See
`solid_bone` for the fix -- enclosed cavities are filled per axial slice before
the width is probed, while the canal detector keeps the unfilled mask because a
canal is a cavity that must stay one.

That is a correctness fix: it makes the baseline measure the thing it claims to
measure. It is NOT tuning, and the distinction is the point. Exactly one fix was
applied and the result was then reported unchanged, because a baseline adjusted
repeatedly by whoever reports the comparison stops being a baseline and becomes
a fitted model with the fitting hidden in the commit history.

If it is still weak after this, that is the finding. If it is strong, that is a
much more important finding, and it is the one worth having before folds 1-4.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.site_dataset import (  # noqa: E402
    cut_patch,
    load_sites,
    patch_centre,
    restrict_sites_to_cache,
)
from src.data.splits import fold_assignment, load_folds  # noqa: E402
from src.data.taskdef import primary_dataset  # noqa: E402
from src.utils.config import artifacts_dir, load_config  # noqa: E402
from src.utils.log import get_logger  # noqa: E402

log = get_logger("geometric_baseline")

SPACING_MM = 0.3          # ToothFairy3 is isotropic; the cache does not resample
DISC_RADIUS_MM = 3.0      # same cylinder `measure_site` uses
WIDTH_PROBE_MM = 2.0      # same depth below the crest `ridge_width` probes
MIN_CANAL_MM = 1.5        # a dark run shorter than this is noise, not a canal


def otsu(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's threshold. Parameter-free by construction, which is the point.

    A fixed cut in z-scored units would be a knob, and a knob chosen by looking
    at these patches would quietly turn the baseline into a fitted model.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    counts, edges = np.histogram(finite, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2.0
    w = np.cumsum(counts)
    total = w[-1]
    if total == 0 or w[-1] == w[0]:
        return float("nan")
    cw = np.cumsum(counts * centres)
    mean_total = cw[-1] / total
    with np.errstate(invalid="ignore", divide="ignore"):
        w0 = w / total
        w1 = 1.0 - w0
        m0 = cw / np.maximum(w, 1)
        m1 = (cw[-1] - cw) / np.maximum(total - w, 1)
        between = w0 * w1 * (m0 - m1) ** 2
    between[~np.isfinite(between)] = -np.inf
    return float(centres[int(np.argmax(between))]) if np.isfinite(mean_total) else float("nan")


def solid_bone(patch: np.ndarray, threshold: float) -> np.ndarray:
    """Bone with enclosed cavities filled, per axial slice.

    ONE THRESHOLD CANNOT SEPARATE TWO KINDS OF BONE. Otsu lands around +1.5 in
    z-scored units and cuts between dense cortical bone and everything else, so
    the trabecular bone filling the ridge falls below it. A run through the
    centre then reads cortex, marrow, cortex; the centre voxel fails; the
    nearest-run fallback fires; and what gets measured is ONE CORTICAL PLATE.
    That is what turned a true 11.78 mm median ridge into a measured 5.10 mm.

    It is the same fault v2.0.0 fixed in the mask pipeline from the other side,
    where a bone-only slab had a hole at every occupied site and a 6.00 mm ridge
    measured 1.80 mm. Here it arrives through intensity instead of labels.

    Filling per SLICE, not in 3D: the canal is a tube running roughly in-plane,
    so it is enclosed in most axial cross-sections and a 3D fill would swallow
    it along with the marrow. `canal_roof` therefore keeps using the unfilled
    mask, and only the width probe uses this one.
    """
    filled = np.empty(patch.shape, dtype=bool)
    bone = patch > threshold
    for z in range(patch.shape[2]):
        filled[:, :, z] = ndimage.binary_fill_holes(bone[:, :, z])
    return filled


def central_disc(shape2d, radius_mm: float = DISC_RADIUS_MM) -> np.ndarray:
    """Boolean disc of `radius_mm` about the patch centre, in the (x, y) plane."""
    cx, cy = shape2d[0] / 2.0, shape2d[1] / 2.0
    xs = (np.arange(shape2d[0]) - cx) * SPACING_MM
    ys = (np.arange(shape2d[1]) - cy) * SPACING_MM
    return xs[:, None] ** 2 + ys[None, :] ** 2 <= radius_mm ** 2


def crest_from_image(bone: np.ndarray, disc: np.ndarray) -> float:
    """Highest bone in each column, median over the disc. Median, not maximum.

    `implant_sites.crest_index` records why: a single spike at the edge of the
    cylinder sets the level for the whole site, the width probe then samples
    below a level the centre never reaches, and a 6 mm ridge measures 1.8 mm.
    """
    inside = bone & disc[:, :, None]
    has = inside.any(axis=2)
    if not has.any():
        return float("nan")
    tops = bone.shape[2] - 1 - np.argmax(inside[:, :, ::-1], axis=2)
    return float(np.median(tops[has]))


def canal_roof(bone: np.ndarray, disc: np.ndarray, crest: int) -> float:
    """First sustained non-bone run below the crest that has bone above AND below.

    The inferior alveolar canal is a dark tube running INSIDE bone, so the
    signature is a gap in an otherwise bone-filled column -- not simply the
    bottom of the bone, which is the inferior border of the mandible.

    Returns the z index of the canal roof, or nan when no enclosed gap is found.
    A nan here is a refusal, not a zero: anterior sites genuinely have no canal
    above them, and reporting a height measured to the wrong structure would be
    worse than reporting nothing.
    """
    frac = (bone & disc[:, :, None]).sum(axis=(0, 1)) / max(disc.sum(), 1)
    filled = frac > 0.5                       # this slice is mostly bone
    top = int(min(max(crest, 0), len(filled) - 1))

    min_run = max(1, int(round(MIN_CANAL_MM / SPACING_MM)))
    z = top
    while z >= 0:
        if not filled[z]:
            end = z
            while end >= 0 and not filled[end]:
                end -= 1
            if (z - end) >= min_run and end >= 0 and filled[end]:
                return float(z)               # bone above, gap, bone below
            z = end
        z -= 1
    return float("nan")


def run_through(line: np.ndarray, index: int) -> tuple[int, bool]:
    """Length of the contiguous True run containing `index`, and whether it fell back.

    Mirrors `implant_sites.run_through`, including the nearest-run fallback, and
    like the fixed version it REPORTS the fallback rather than hiding it. That
    fallback is what measured one cortical plate in isolation and turned a 6.00 mm
    ridge into 1.80 mm.
    """
    n = len(line)
    if n == 0 or not line.any():
        return 0, False
    index = int(np.clip(index, 0, n - 1))
    fell_back = False
    if not line[index]:
        hits = np.where(line)[0]
        index = int(hits[np.argmin(np.abs(hits - index))])
        fell_back = True
    lo = hi = index
    while lo > 0 and line[lo - 1]:
        lo -= 1
    while hi < n - 1 and line[hi + 1]:
        hi += 1
    return hi - lo + 1, fell_back


def measure(patch: np.ndarray, crest_index: float) -> dict:
    """Height and width in millimetres from image intensities alone."""
    patch = np.asarray(patch, dtype=np.float32)
    thr = otsu(patch)
    if not np.isfinite(thr):
        return {"height_mm": np.nan, "width_mm": np.nan, "crest_z": np.nan,
                "canal_z": np.nan, "width_fallback": False, "reason": "no_threshold"}

    bone = patch > thr
    disc = central_disc(patch.shape[:2])

    crest = crest_index
    if not np.isfinite(crest):
        crest = crest_from_image(bone, disc)
    if not np.isfinite(crest):
        return {"height_mm": np.nan, "width_mm": np.nan, "crest_z": np.nan,
                "canal_z": np.nan, "width_fallback": False, "reason": "no_bone"}
    crest = int(round(crest))

    roof = canal_roof(bone, disc, crest)
    height = (crest - roof) * SPACING_MM if np.isfinite(roof) else np.nan

    # Width: the SHORTER of the two in-plane runs, probed below the crest.
    step = max(1, int(round(WIDTH_PROBE_MM / SPACING_MM)))
    z = crest - step
    width, fell_back = np.nan, False
    if 0 <= z < bone.shape[2]:
        # The FILLED mask here, the unfilled one for the canal above. A ridge is
        # cortex plus the trabecular bone it encloses; the canal is a cavity that
        # must stay a cavity.
        slab = solid_bone(patch, thr)[:, :, z]
        cx, cy = slab.shape[0] // 2, slab.shape[1] // 2
        runs = []
        for line, idx, mm in ((slab[:, cy], cx, SPACING_MM), (slab[cx, :], cy, SPACING_MM)):
            length, fb = run_through(line, idx)
            fell_back = fell_back or fb
            if length:
                runs.append(length * mm)
        width = float(min(runs)) if runs else 0.0

    return {"height_mm": float(height) if np.isfinite(height) else np.nan,
            "width_mm": width, "crest_z": float(crest),
            "canal_z": float(roof) if np.isfinite(roof) else np.nan,
            "width_fallback": bool(fell_back),
            "reason": "ok" if np.isfinite(height) else "no_canal_in_patch"}


def agreement_table(df: pd.DataFrame, min_width_mm: float, rules) -> pd.DataFrame:
    """The same sweep `train.py` prints, so the two can be read side by side."""
    rows = []
    ok = df[df.height_mm.notna() & df.ridge_width_mm.notna()
            & df.available_height_mm.notna() & df.width_mm.notna()]
    for rule in rules:
        measured = (ok.available_height_mm >= rule) & (ok.ridge_width_mm >= min_width_mm)
        predicted = (ok.height_mm >= rule) & (ok.width_mm >= min_width_mm)
        rows.append({"height_threshold_mm": rule,
                     "measured_feasible_rate": float(measured.mean()),
                     "predicted_feasible_rate": float(predicted.mean()),
                     "agreement": float((measured == predicted).mean()),
                     "n": int(len(ok))})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sites.yaml")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"],
                    help="val by default: test is scored once, by pool_cv, at the end")
    ap.add_argument("--find-crest", action="store_true",
                    help="re-derive the crest from the image instead of using site_z. "
                         "Slower and worse, and the honest variant to quote.")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    dataset = primary_dataset(cfg)
    cache = Path(cfg.data.cache_dir) / dataset
    art = artifacts_dir(cfg)

    targets = ["needs_implant", "available_height_mm", "ridge_width_mm"]
    sites = load_sites(art / cfg.task.sites_csv, targets=targets,
                       methods=tuple(getattr(cfg.task, "site_methods", ("teeth",))),
                       jaws=tuple(getattr(cfg.task, "site_jaws", ("lower",))))
    sites = restrict_sites_to_cache(sites, cache)

    folds = load_folds(art / "cv_folds.json")
    split = fold_assignment(folds, args.fold)[args.split]
    sites = sites[sites.patient_id.isin(set(split))].reset_index(drop=True)
    if args.limit:
        sites = sites.head(args.limit)
    log.info("%s split of fold %d: %d sites, %d patients",
             args.split, args.fold, len(sites), sites.patient_id.nunique())

    patch_size = int(getattr(cfg.model, "img_size", 96))
    volumes: dict[str, np.ndarray] = {}
    rows = []
    for i, row in sites.iterrows():
        pid = row.patient_id
        if pid not in volumes:
            volumes[pid] = np.load(cache / f"{pid}.npy", mmap_mode="r")
        vol = volumes[pid]
        centre = patch_centre(row, vol.shape, patch_size)
        patch = cut_patch(vol, centre, patch_size)
        # site_z is the crest; the patch is quarter-shifted, so it sits here.
        known_crest = np.nan if args.find_crest else patch_size / 2 + patch_size // 4
        out = measure(patch, known_crest)
        out.update({"patient_id": pid, "tooth": row.tooth,
                    "available_height_mm": row.available_height_mm,
                    "ridge_width_mm": row.ridge_width_mm})
        rows.append(out)
        if (i + 1) % 200 == 0:
            log.info("  %d/%d", i + 1, len(sites))

    df = pd.DataFrame(rows)
    out_path = art / f"geometric_baseline_fold{args.fold}_{args.split}.csv"
    df.to_csv(out_path, index=False)

    scored = df[df.height_mm.notna()]
    print("\n" + "=" * 66)
    print(f"GEOMETRIC BASELINE -- no learning, fold {args.fold} {args.split}")
    print("=" * 66)
    print(f"sites {len(df)} | measurable {len(scored)} "
          f"({100 * len(scored) / max(len(df), 1):.1f}%) | "
          f"width fell back on {100 * df.width_fallback.mean():.1f}%")
    for why, n in df.reason.value_counts().items():
        print(f"  {why:<22} {n}")

    if len(scored):
        for col, truth in (("height_mm", "available_height_mm"),
                           ("width_mm", "ridge_width_mm")):
            err = (scored[col] - scored[truth]).abs()
            floor = (scored[truth] - scored[truth].mean()).abs().mean()
            print(f"\n{truth:<22} MAE {err.mean():6.3f} mm   "
                  f"floor {floor:6.3f} mm   "
                  f"{'BELOW floor' if err.mean() < floor else 'ABOVE floor -- no better than the mean'}")

        rules = [r["height_threshold_mm"] for r in
                 ([{"height_threshold_mm": v} for v in (10.0, 11.0, 12.0, 13.0, 14.0)])]
        tab = agreement_table(df, float(cfg.sites.min_width_mm), rules)
        print("\nFEASIBILITY AGREEMENT -- compare line for line with train.py")
        print("   height rule   measured   predicted   agreement")
        for _, r in tab.iterrows():
            print(f"      {r.height_threshold_mm:5.1f} mm      {r.measured_feasible_rate:.3f}"
                  f"       {r.predicted_feasible_rate:.3f}       {r.agreement:.3f}")
        print("\n  If these agreements match the model's, the model has learned")
        print("  nothing a threshold could not measure directly.")

    log.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
