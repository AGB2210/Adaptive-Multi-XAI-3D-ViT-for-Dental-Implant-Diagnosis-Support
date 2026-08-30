"""Build the offline preprocessing cache.

    python scripts/build_cache.py --dataset toothfairy3 --limit 10 --preview 8
    python scripts/build_cache.py --dataset all --workers 4

Resumable (already-cached patients are skipped unless --force) and parallel.
Writes artifacts/preprocess_manifest.csv and artifacts/preview/*.png.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocess import info_to_row, preprocess_volume  # noqa: E402
from src.utils.config import (  # noqa: E402
    artifacts_dir,
    dataset_names,
    list_cases,
    load_config,
    validate_roots,
    volume_path,
)
from src.utils.config import (
    cache_dir as cache_dir_for,
)
from src.utils.log import get_logger  # noqa: E402

log = get_logger("build_cache")

MANIFEST = "preprocess_manifest.csv"
SETTINGS = "_cache_settings.json"


def stale_settings(cache_dir: Path, kwargs: dict) -> dict | None:
    """Which preprocessing settings differ from the ones this cache was built with.

    The cached file is named {patient_id}.npy with no resolution, spacing or clip
    window in it, so a cache built at 1 mm is indistinguishable on disk from one
    built at 0.5 mm. Without this check, rebuilding with changed settings skips
    all N cases as "already cached" and reports success, leaving every downstream
    run silently training on the old preprocessing.

    Returns None when the cache is absent or was built with these exact settings.
    """
    if not any(cache_dir.glob("*.npy")):
        return None

    path = cache_dir / SETTINGS
    if not path.exists():
        # A cache with no stamp predates this check; its provenance is unknown,
        # so it cannot be certified as matching.
        return {"(no recorded settings)": ("unknown", "current")}

    old = json.loads(path.read_text(encoding="utf-8"))
    new = json.loads(json.dumps(kwargs, default=list))  # tuples -> lists, as stored
    return {k: (old.get(k), new[k]) for k in new if old.get(k) != new[k]} or None


def write_settings(cache_dir: Path, kwargs: dict) -> None:
    (cache_dir / SETTINGS).write_text(
        json.dumps(kwargs, indent=2, sort_keys=True, default=list), encoding="utf-8"
    )


def find_volume(cfg, dataset: str, pid: str) -> Path | None:
    """Resolve one case's volume from the config's path pattern.

    A .nii.gz pattern also matches a plain .nii and vice versa: cohorts are
    inconsistent about which they ship and that is not worth a config entry.
    """
    path = volume_path(cfg, dataset, pid)
    candidates = [path]
    if path.name.endswith(".nii.gz"):
        candidates.append(path.with_name(path.name[: -len(".gz")]))
    elif path.name.endswith(".nii"):
        candidates.append(path.with_name(path.name + ".gz"))
    for c in candidates:
        if c.is_file():
            return c
    return None


def list_patients(cfg, dataset: str) -> list[tuple[str, Path]]:
    """(case id, volume path) for every case the dataset enumerates."""
    out = []
    for pid in list_cases(cfg, dataset):
        vol = find_volume(cfg, dataset, pid)
        if vol is not None:
            out.append((pid, vol))
        else:
            log.warning("no volume for %s/%s -- skipping", dataset, pid)
    return out


def _job(args) -> tuple[dict, str | None]:
    pid, path, dataset, out_path, kwargs = args
    arr, info = preprocess_volume(Path(path), pid, dataset, **kwargs)
    if arr is not None:
        tmp = Path(str(out_path) + ".tmp.npy")
        np.save(tmp, arr)
        tmp.replace(out_path)  # atomic: a partial file can never look cached
    return info_to_row(info), (str(out_path) if arr is not None else None)


def montage(arr: np.ndarray, path: Path, title: str) -> None:
    """Axial / coronal / sagittal mid-slices, so orientation can be eyeballed."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = arr.astype(np.float32)
    mid = [s // 2 for s in a.shape]
    planes = [
        ("axial (z mid)", a[:, :, mid[2]].T),
        ("coronal (y mid)", a[:, mid[1], :].T),
        ("sagittal (x mid)", a[mid[0], :, :].T),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, (name, plane) in zip(axes, planes):
        ax.imshow(np.flipud(plane), cmap="gray", vmin=-2, vmax=2)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{title}   (z-scored, window [-2, 2])", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run_dataset(cfg, dataset: str, args) -> pd.DataFrame:
    cache_dir = cache_dir_for(cfg, dataset)
    cache_dir.mkdir(parents=True, exist_ok=True)

    patients = list_patients(cfg, dataset)
    if args.limit:
        patients = patients[: args.limit]

    kwargs = dict(
        out_shape=tuple(cfg.preprocess.out_shape),
        margin=cfg.preprocess.fg_margin,
        clip_percentiles=tuple(cfg.preprocess.clip_percentiles),
        clip_mode=getattr(cfg.preprocess, "clip_mode", "percentile"),
        clip_window=tuple(getattr(cfg.preprocess, "clip_window", (-1000.0, 6000.0))),
        fit_mode=args.fit_mode or cfg.preprocess.fit_mode,
        target_spacing=cfg.preprocess.target_spacing,
    )

    # A settings change invalidates every cached volume, so the resume-by-skipping
    # behaviour must not apply -- otherwise the build is a silent no-op and the
    # cache keeps the old preprocessing under the same filenames.
    changed = stale_settings(cache_dir, kwargs)
    rebuild = changed is not None
    if rebuild:
        log.warning("%s: cache was built with different preprocessing -- rebuilding all %d",
                    dataset, len(patients))
        for key, (old, new) in sorted(changed.items()):
            log.warning("    %-18s %s -> %s", key, old, new)

    todo, skipped = [], 0
    for pid, vol in patients:
        out_path = cache_dir / f"{pid}.npy"
        if out_path.exists() and not args.force and not rebuild:
            skipped += 1
            continue
        todo.append((pid, str(vol), dataset, out_path, kwargs))

    log.info("%s: %d patients, %d already cached, %d to process (workers=%d)",
             dataset, len(patients), skipped, len(todo), args.workers)

    rows: list[dict] = []
    if todo:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_job, job): job[0] for job in todo}
            for i, fut in enumerate(as_completed(futures), 1):
                row, _ = fut.result()
                rows.append(row)
                status = row["status"]
                if status != "ok":
                    log.error("[%s] %s FAILED: %s", dataset, row["patient_id"], row["warning"])
                elif row["warning"]:
                    log.warning("[%s] %s: %s", dataset, row["patient_id"], row["warning"])
                if i % 10 == 0 or i == len(todo):
                    ok = sum(r["status"] == "ok" for r in rows)
                    log.info("[%s] %d/%d done (%d ok, %d failed)", dataset, i, len(todo), ok, len(rows) - ok)

    df = pd.DataFrame(rows)

    # Stamp only once the build actually succeeded, so an interrupted or failing
    # run never certifies a half-written cache as matching these settings.
    failed = 0 if df.empty else int((df["status"] != "ok").sum())
    if failed:
        log.error("%s: %d case(s) failed -- settings NOT stamped, cache stays uncertified",
                  dataset, failed)
    else:
        write_settings(cache_dir, kwargs)

    if args.preview:
        preview_dir = artifacts_dir(cfg) / "preview"
        cached = sorted(cache_dir.glob("*.npy"))[: args.preview]
        for p in cached:
            try:
                montage(np.load(p), preview_dir / f"{dataset}_{p.stem}.png", f"{dataset} / {p.stem}")
            except Exception as exc:  # noqa: BLE001
                log.warning("preview failed for %s: %s", p.stem, exc)
        log.info("%s: wrote %d preview montages to %s", dataset, len(cached), preview_dir)

    return df


def summarise(df: pd.DataFrame) -> None:
    if df.empty:
        print("nothing processed this run.")
        return
    ok = df[df["status"] == "ok"]
    print("\n" + "=" * 78)
    print(f"processed {len(df)}  |  ok {len(ok)}  |  failed {len(df) - len(ok)}")
    if not ok.empty:
        print(f"foreground fraction : {ok['fg_fraction'].min():.3f} .. {ok['fg_fraction'].max():.3f} "
              f"(median {ok['fg_fraction'].median():.3f})")
        print(f"elapsed per volume  : {ok['elapsed_s'].min():.1f}s .. {ok['elapsed_s'].max():.1f}s "
              f"(median {ok['elapsed_s'].median():.1f}s)")
        eff = ok["effective_spacing_mm"].str.split(",").str[0].astype(float)
        print(f"effective spacing   : {eff.min():.2f} .. {eff.max():.2f} mm/voxel (median {eff.median():.2f})")
    warned = df[(df["status"] == "ok") & df["warning"].notna() & (df["warning"].astype(str).str.strip() != "")]
    if not warned.empty:
        print(f"\nwarnings on {len(warned)} volumes:")
        for _, r in warned.head(10).iterrows():
            print(f"   {r['patient_id']:<10} {r['warning']}")
    failed = df[df["status"] != "ok"]
    if not failed.empty:
        print(f"\nFAILED {len(failed)}:")
        for _, r in failed.iterrows():
            print(f"   {r['patient_id']:<10} {r['warning']}")
    print("=" * 78)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset", default="all",
                    help="a dataset name from the config, or 'all'")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke-test only: the first N cases ALPHABETICALLY, which is not a sample -- ToothFairy3's prefixes group by sub-cohort, so the first 14 are all ToothFairy3F and averaged 104.9 MB against 48 MB cohort-wide. Extrapolating from that is what produced the retracted 52 GB estimate")
    ap.add_argument("--workers", type=int, default=0, help="override cfg.preprocess.workers")
    ap.add_argument("--preview", type=int, default=0, help="write N preview montages per dataset")
    ap.add_argument("--force", action="store_true", help="re-process already-cached volumes")
    ap.add_argument("--fit-mode", dest="fit_mode", choices=["isotropic_pad", "resize"], default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    args.workers = args.workers or cfg.preprocess.workers
    datasets = dataset_names(cfg) if args.dataset == "all" else [args.dataset]
    validate_roots(cfg, need=tuple(datasets))

    frames = [run_dataset(cfg, ds, args) for ds in datasets]
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(not f.empty for f in frames) else pd.DataFrame()

    # Merge into any existing manifest so resumed runs keep earlier rows.
    path = artifacts_dir(cfg) / MANIFEST
    if not df.empty:
        if path.exists():
            old = pd.read_csv(path, dtype={"patient_id": str})
            keys = set(zip(df["dataset"], df["patient_id"]))
            old = old[~old.apply(lambda r: (r["dataset"], str(r["patient_id"])) in keys, axis=1)]
            df = pd.concat([old, df], ignore_index=True)
        df.to_csv(path, index=False, encoding="utf-8")
        log.info("manifest -> %s (%d rows)", path, len(df))

    summarise(df)


if __name__ == "__main__":
    main()
