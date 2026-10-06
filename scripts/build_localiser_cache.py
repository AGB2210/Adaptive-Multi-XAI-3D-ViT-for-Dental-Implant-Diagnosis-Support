"""Cache every scan at the site localiser's resolution.

    python scripts/build_localiser_cache.py --config configs/localiser.yaml

Writes <localiser.cache_dir>/<patient>.npy (float16, the fixed grid) and
manifest.csv with the offset `localiser_input` applied, which is what turns a
site's full-resolution coordinates into grid coordinates.

Two sources for the full-resolution volume, in order:

  1. the site cache (`build_site_cache.py`), which IS `prepare_volume`'s output
     -- reading it is minutes, not the half hour of decompressing every scan;
  2. the raw image, oriented by the sign the label build recorded for it
     (`sites_toothfairy3.csv`, orientation_sign) and prepared by the same
     `prepare_volume`.

Either way the grid is produced by `localiser_input`, the function the app
calls on an uploaded scan, so the training input and the app's input are the
same transform of the same volume. A scan whose orientation never resolved
(10 of 532) has no sign and no site labels, and is skipped.

At 1.2 mm a scan is ~1.3 MB, so the whole cohort is well under a gigabyte.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.scan import prepare_volume  # noqa: E402
from src.data.taskdef import primary_dataset  # noqa: E402
from src.models.localiser import localiser_input  # noqa: E402
from src.utils.config import artifacts_dir, cache_dir, load_config, volume_path  # noqa: E402
from src.utils.log import get_logger  # noqa: E402

log = get_logger("localiser_cache")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/localiser.yaml")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke-test only: the first N patients alphabetically, which is not a sample")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    loc = cfg.localiser
    dataset = primary_dataset(cfg)
    sites = pd.read_csv(artifacts_dir(cfg) / cfg.task.sites_csv, dtype={"patient_id": str})
    signs = (sites.dropna(subset=["orientation_sign"]).groupby("patient_id")["orientation_sign"]
             .first().astype(int))
    patients = sorted(signs.index)
    if args.limit:
        patients = patients[: args.limit]

    out_dir = Path(loc.cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    site_cache = cache_dir(cfg, dataset)
    clip = tuple(cfg.preprocess.clip_window)
    air = float(getattr(cfg.preprocess, "air_threshold", -500.0))
    shape = tuple(loc.input_shape)
    log.info("%d patients -> %s (factor %d, grid %s)", len(patients), out_dir, loc.factor, shape)

    rows = []
    for k, pid in enumerate(patients, 1):
        target = out_dir / f"{pid}.npy"
        cached = site_cache / f"{pid}.npy"
        try:
            if cached.is_file():
                volume, source = np.load(cached, mmap_mode="r"), "site cache"
            else:
                image = np.array(nib.load(str(volume_path(cfg, dataset, pid))).dataobj,
                                 dtype=np.float32)
                volume, *_ = prepare_volume(image, int(signs[pid]), clip, air)
                source = "raw image"
            grid, offset = localiser_input(volume, loc.factor, shape)
            if not target.exists() or args.force:
                tmp = target.with_name(target.stem + ".tmp.npy")
                np.save(tmp, grid.astype(np.float16))
                tmp.replace(target)
            rows.append({"patient_id": pid, "status": "ok", "source": source,
                         "full_shape": "x".join(map(str, volume.shape)),
                         "offset_x": int(offset[0]), "offset_y": int(offset[1]),
                         "offset_z": int(offset[2]), "cropped": bool((offset < 0).any())})
        except Exception as exc:  # noqa: BLE001 - one bad scan must not lose the run
            log.error("%s: %s -- skipped", pid, exc)
            rows.append({"patient_id": pid, "status": "failed", "source": str(exc)})
        if k % 25 == 0:
            log.info("  %d/%d", k, len(patients))

    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    ok = manifest[manifest.status == "ok"]
    log.info("cached %d, failed %d, cropped %d", len(ok), len(manifest) - len(ok),
             int(ok.get("cropped", pd.Series(dtype=bool)).sum()))
    if len(ok) and ok.cropped.any():
        log.warning("some scans exceed the grid and were cropped; sites near their borders "
                    "can fall outside it -- see manifest.csv")


if __name__ == "__main__":
    main()
