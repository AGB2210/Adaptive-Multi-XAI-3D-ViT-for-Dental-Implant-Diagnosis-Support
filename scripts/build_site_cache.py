"""Cache whole scans at NATIVE resolution, in the same frame as the site labels.

    python scripts/build_site_cache.py --config configs/sites.yaml

Writes artifacts_sites/cache/<patient>.npy -- one float16 array per scan, at the
scan's own 0.3 mm, in exactly the voxel frame that sites_*.csv indexes. Training
patches are cut from these in the data loader.

WHY NOT REUSE scripts/build_cache.py. That one crops to the head, reorients to
RAS+ via nib.as_closest_canonical, and resamples onto a fixed 128^3 grid. Every
one of those steps moves voxel coordinates, and site_x / site_y / site_z were
measured in the raw mask's frame. Feeding those indices into a cropped,
reoriented, resampled volume would sample confidently from the wrong anatomy and
nothing downstream would notice. The two caches answer different questions and
are kept separate rather than parameterised into one another.

WHAT THIS DOES DO, and it has to match src/data/implant_sites.py exactly:

  * loads the image WITHOUT canonical reorientation
  * applies the same z flip the mask needed, derived from the mask's anatomy
  * clips to the fixed HU window and z-scores on foreground, as before

DISK. A 410x410x273 scan at float16 is ~92 MB, so 532 scans is roughly 49 GB.
That is why this is a rented-box step: it will not fit on the laptop the earlier
128^3 cache was built for. It is also why nothing is stored per site -- 28
patches per scan would triple the space for no additional information.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.implant_sites import superior_sign  # noqa: E402
from src.data.taskdef import primary_dataset  # noqa: E402
from src.utils.config import (  # noqa: E402
    cache_dir,
    dataset_root,
    label_path,
    list_cases,
    load_config,
    volume_path,
)
from src.utils.log import get_logger  # noqa: E402

log = get_logger("site_cache")

SETTINGS = "_cache_settings.json"


def normalise(volume: np.ndarray, clip_window, air_threshold: float):
    """Fixed-window clip then z-score on foreground. Same contract as before.

    The window is fixed rather than a per-patient percentile for the reason
    recorded in src/data/preprocess.py: implants are hyperdense and a 99.5th
    percentile clip flattens them onto cortical bone, erasing the very thing the
    model has to see.
    """
    lo, hi = float(clip_window[0]), float(clip_window[1])
    volume = np.clip(volume, lo, hi)

    foreground = volume[volume > air_threshold]
    if foreground.size < 100:
        foreground = volume
    mean, std = float(foreground.mean()), float(foreground.std())
    if std < 1e-6:
        raise ValueError("zero-variance volume")
    return ((volume - mean) / std).astype(np.float16), mean, std


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sites.yaml")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="rebuild volumes already cached")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dataset = args.dataset or primary_dataset(cfg)
    if getattr(cfg.preprocess, "target_spacing", None) is not None:
        raise SystemExit(
            "preprocess.target_spacing must be null for the site cache.\n"
            "This builder exists to keep the scan at its own resolution; a "
            "target spacing means you want scripts/build_cache.py instead."
        )

    root = dataset_root(cfg, dataset)
    if not root.is_dir():
        raise SystemExit(f"dataset root not found: {root.resolve()}")

    out_dir = cache_dir(cfg, dataset)
    out_dir.mkdir(parents=True, exist_ok=True)

    clip_window = tuple(cfg.preprocess.clip_window)
    air = float(getattr(cfg.preprocess, "air_threshold", -500.0))
    cases = list_cases(cfg, dataset)
    if args.limit:
        cases = cases[: args.limit]
    log.info("%s: %d scans -> %s (native resolution)", dataset, len(cases), out_dir)

    rows, skipped = [], 0
    for k, pid in enumerate(cases, 1):
        target = out_dir / f"{pid}.npy"
        if target.exists() and not args.force:
            continue

        mask_file = label_path(cfg, dataset, pid)
        image_file = volume_path(cfg, dataset, pid)
        if mask_file is None or not mask_file.is_file() or not image_file.is_file():
            skipped += 1
            log.warning("%s: missing image or mask -- skipped", pid)
            continue

        try:
            # The flip is decided by the MASK's anatomy and applied to the image,
            # so both end up in the frame the site coordinates were measured in.
            sign = superior_sign(np.asarray(nib.load(str(mask_file)).dataobj))

            img = nib.load(str(image_file))
            spacing = np.asarray(img.header.get_zooms()[:3], dtype=float)
            volume = np.asarray(img.dataobj, dtype=np.float32)
            np.nan_to_num(volume, copy=False, nan=air, posinf=air, neginf=air)
            if sign == -1:
                volume = volume[:, :, ::-1]

            out, mean, std = normalise(volume, clip_window, air)
            np.save(target, np.ascontiguousarray(out))
            rows.append({
                "patient_id": pid, "status": "ok",
                "shape": "x".join(map(str, out.shape)),
                "spacing_mm": ",".join(f"{v:.4f}" for v in spacing),
                "orientation_sign": sign, "fg_mean": mean, "fg_std": std,
                "mb": round(out.nbytes / 1e6, 1),
            })
        except Exception as exc:  # noqa: BLE001 - one bad scan must not lose the run
            skipped += 1
            log.error("%s: %s -- skipped", pid, exc)
            rows.append({"patient_id": pid, "status": "failed", "shape": "",
                         "spacing_mm": "", "orientation_sign": 0,
                         "fg_mean": np.nan, "fg_std": np.nan, "mb": 0.0})

        if k % 20 == 0:
            log.info("  %d/%d", k, len(cases))

    if rows:
        manifest = pd.DataFrame(rows)
        manifest.to_csv(out_dir.parent / "site_cache_manifest.csv", index=False)
        good = manifest[manifest.status == "ok"]
        log.info("cached %d scans, %.1f GB total, %d skipped",
                 len(good), good.mb.sum() / 1000.0, skipped)
    else:
        log.info("nothing to do: every scan is already cached (use --force to rebuild)")


if __name__ == "__main__":
    main()
