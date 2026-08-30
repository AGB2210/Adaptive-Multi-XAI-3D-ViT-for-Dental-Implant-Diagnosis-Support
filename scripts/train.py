"""Train a model on the primary cohort's cache.

    python scripts/train.py --config configs/default.yaml                  # ViT
    python scripts/train.py --config configs/default.yaml --model cnn3d    # CNN baseline
    python scripts/train.py --config configs/synthetic.yaml --synthetic    # convergence check

The TEST split is not scored unless --test is passed. Selecting a model while
watching its test score is leakage even when no threshold is tuned on it.

Splits are created once and persisted to artifacts/splits.json; every later run
(and the whole XAI phase) reuses them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.augment import Augment3D  # noqa: E402
from src.data.dataset import (  # noqa: E402
    CachedVolumeDataset,
    load_label_matrix,
    pos_weight_from_labels,
    restrict_to_cache,
)
from src.data.site_dataset import (  # noqa: E402
    SitePatchDataset,
    load_sites,
    patient_label_matrix,
    restrict_sites_to_cache,
    sites_for_patients,
    target_matrix,
)
from src.data.splits import (  # noqa: E402
    check_disjoint,
    check_folds_disjoint,
    fold_assignment,
    load_folds,
    load_splits,
    make_cv_folds,
    make_splits,
    save_folds,
    save_splits,
    split_prevalence,
)
from src.data.taskdef import (  # noqa: E402
    all_target_names,
    label_names_for,
    primary_dataset,
    regression_names_for,
)
from src.models import build_model
from src.models.prevalence import PrevalenceBaseline  # noqa: E402
from src.train.loop import Trainer, predict  # noqa: E402
from src.train.metrics import evaluate, format_metrics, no_information_bce  # noqa: E402
from src.train.targets import (  # noqa: E402
    format_regression,
    no_information_regression,
    regression_metrics,
    spec_from_config,
    threshold_sensitivity,
)
from src.utils.config import artifacts_dir, load_config  # noqa: E402
from src.utils.log import get_logger  # noqa: E402
from src.utils.seed import set_seed, worker_init_fn  # noqa: E402
from src.xai.runner import model_img_size  # noqa: E402

log = get_logger("train_script")


def build_cv_folds(cfg, patient_ids, y, label_names, n_folds: int, force: bool = False):
    """Persisted K-fold partition. Created once, then reused by every round.

    Regenerating between rounds would put a case in training for one round and
    testing for another under a different partition, which quietly destroys the
    guarantee that each case is scored exactly once by a model that never saw it.
    """
    path = artifacts_dir(cfg) / "cv_folds.json"
    if path.exists() and not force:
        folds = load_folds(path)
        known = set(patient_ids)
        unknown = [p for f in folds for p in f if p not in known]
        if unknown:
            raise ValueError(
                f"{path} references {len(unknown)} cases absent from the label/cache set "
                f"(e.g. {unknown[:5]}). Delete it or pass --new-splits to rebuild."
            )
        if len(folds) != n_folds:
            raise ValueError(f"{path} holds {len(folds)} folds but --folds asked for {n_folds}; "
                             "pass --new-splits to rebuild rather than mixing partitions")
        log.info("reusing %d-fold partition from %s", len(folds), path)
    else:
        folds = make_cv_folds(patient_ids, y, n_folds=n_folds, seed=cfg.split.seed)
        # `patient_ids` comes from `patient_label_matrix`, which groups by
        # patient, so its length is a patient count. Recording it as `n_cases`
        # read as the site count (486 against ~6,787) to anyone sizing the
        # cohort from this file, and was right only while the superseded task
        # had one case per patient.
        save_folds(folds, path, meta={"seed": cfg.split.seed, "n_folds": n_folds,
                                      "labels": label_names,
                                      "n_patients": len(patient_ids)})
        log.info("wrote %d-fold partition to %s", n_folds, path)

    check_folds_disjoint(folds)
    return folds


def build_splits(cfg, patient_ids, y, label_names, force: bool = False):
    path = artifacts_dir(cfg) / "splits.json"
    if path.exists() and not force:
        splits = load_splits(path)
        known = set(patient_ids)
        unknown = [p for ids in splits.values() for p in ids if p not in known]
        if unknown:
            raise ValueError(
                f"{path} references {len(unknown)} patients absent from the label/cache set "
                f"(e.g. {unknown[:5]}). Delete it or pass --new-splits to rebuild."
            )
        log.info("reusing splits from %s", path)
    else:
        splits = make_splits(patient_ids, y, tuple(cfg.split.ratios), seed=cfg.split.seed)
        save_splits(
            splits,
            path,
            meta={
                "seed": cfg.split.seed,
                "ratios": list(cfg.split.ratios),
                "labels": label_names,
                "n_patients": len(patient_ids),
                "prevalence": split_prevalence(splits, patient_ids, y),
            },
        )
        log.info("wrote new splits to %s", path)

    check_disjoint(splits)
    return splits


def make_loaders(cfg, splits, patient_ids, y, cache_dir, args):
    index = {pid: i for i, pid in enumerate(patient_ids)}
    augment = Augment3D(cfg.augment)
    loaders, matrices = {}, {}

    for name in ("train", "val", "test"):
        ids = [p for p in splits[name] if p in index]
        rows = np.array([index[p] for p in ids], dtype=int)
        labels = y[rows]
        matrices[name] = (ids, labels)

        dataset = CachedVolumeDataset(
            cache_dir,
            ids,
            labels,
            augment=augment if name == "train" else None,
            preload=args.preload,
            seed=cfg.seed,
        )
        loaders[name] = DataLoader(
            dataset,
            batch_size=cfg.train.batch_size,
            shuffle=(name == "train"),
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            worker_init_fn=worker_init_fn,
            persistent_workers=args.num_workers > 0,
        )
    return loaders, matrices


def make_site_loaders(cfg, splits, sites, cache_dir, args, targets):
    """Loaders over tooth SITES, split by patient.

    The split arrives as lists of patient ids and is expanded to their site rows
    here. Doing it the other way round -- splitting rows -- would put a
    patient's left molars in train and their right molars in test.
    """
    augment = Augment3D(cfg.augment)
    patch = int(getattr(cfg.model, "img_size", 96))
    loaders, matrices = {}, {}

    for name in ("train", "val", "test"):
        subset = sites_for_patients(sites, splits[name])
        # One PATIENT id per SITE row, so this slot holds ~14 duplicates per
        # patient where `make_loaders` puts one id per sample. Only `len()`
        # reads it, which gives the right site count -- but `set()` or
        # `nunique()` here would silently return patients, and that is the
        # confusion that cost this project a published interval claim.
        matrices[name] = (subset.patient_id.tolist(), target_matrix(subset, targets))
        dataset = SitePatchDataset(
            cache_dir, subset, targets=targets, patch_size=patch,
            augment=augment if name == "train" else None, seed=cfg.seed,
        )
        loaders[name] = DataLoader(
            dataset,
            batch_size=cfg.train.batch_size,
            shuffle=(name == "train"),
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            worker_init_fn=worker_init_fn,
            persistent_workers=args.num_workers > 0,
        )
    return loaders, matrices


def synthetic_loaders(cfg, n_train: int = 96, n_val: int = 32):
    """Tiny in-memory synthetic task -- convergence here is a prerequisite for
    launching any real run."""
    from tests.synthetic import make_dataset

    # The gate must build volumes the model can actually take, and the site
    # task has no fixed preprocessing grid to read that from.
    shape = (model_img_size(cfg),) * 3
    n_bin = len(label_names_for(cfg))
    n_mm = len(regression_names_for(cfg))
    if n_mm:
        # A hybrid head needs continuous targets for its regression columns, or
        # the gate trains them on zeros and ones and reports a meaningless MAE.
        from tests.synthetic import make_hybrid_dataset
        xtr, ytr = make_hybrid_dataset(n_train, seed=0, shape=shape,
                                       n_binary=n_bin, n_mm=n_mm)
        xva, yva = make_hybrid_dataset(n_val, seed=10_000, shape=shape,
                                       n_binary=n_bin, n_mm=n_mm)
    else:
        xtr, ytr = make_dataset(n_train, seed=0, shape=shape, n_labels=n_bin)
        xva, yva = make_dataset(n_val, seed=10_000, shape=shape, n_labels=n_bin)

    def loader(x, y, shuffle):
        ds = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).float())
        return DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=shuffle)

    return {"train": loader(xtr, ytr, True), "val": loader(xva, yva, False)}, {
        "train": (None, ytr),
        "val": (None, yva),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default=None, choices=["vit3d", "cnn3d"])
    ap.add_argument("--synthetic", action="store_true", help="train on planted-signal data")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--num-workers", dest="num_workers", type=int, default=None)
    ap.add_argument("--preload", action="store_true", help="hold the cache in RAM (~1.7 GB)")
    ap.add_argument("--new-splits", dest="new_splits", action="store_true")
    ap.add_argument("--fold", type=int, default=None,
                    help="cross-validation round to run; omit for a single train/val/test split")
    ap.add_argument("--folds", type=int, default=5, help="number of CV folds (with --fold)")
    ap.add_argument(
        "--test",
        action="store_true",
        help="also score the TEST split. Off by default on purpose: see the note in the source.",
    )
    ap.add_argument("--resume", default=None, help="checkpoint to resume from")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg.model.name = args.model
    args.num_workers = cfg.train.num_workers if args.num_workers is None else args.num_workers

    # The head count follows the label subset, never the other way round -- a
    # config whose num_classes disagreed with data.train_labels would train
    # silently against misaligned columns.
    # Every output column, binary first then millimetres. num_classes counts the
    # WHOLE head: a hybrid task with one binary and two millimetre targets needs
    # three outputs, and sizing it from the binary block alone would silently
    # drop the regression heads.
    labels = all_target_names(cfg)
    spec = spec_from_config(cfg)
    if cfg.model.num_classes != len(labels):
        log.info("num_classes %d -> %d to match the declared targets",
                 cfg.model.num_classes, len(labels))
        cfg.model.num_classes = len(labels)
    log.info("training on %d labels: %s", len(labels), ", ".join(labels))

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s model=%s", device, cfg.model.name)
    if device.type == "cpu":
        log.warning("no CUDA device: a 12-layer 3D ViT on CPU is impractical for a full run")

    # A cross-validation round MUST get its own directory. Without the fold in
    # the name every round writes over the last one, a five-fold run leaves a
    # single checkpoint, and scripts/pool_cv.py -- which looks for cv_fold{k} --
    # either fails or, worse, pools one model against itself. This used to
    # depend on remembering --out.
    run_name = cfg.model.name + ("_synthetic" if args.synthetic else "")
    if args.fold is not None and not args.synthetic:
        run_name = f"cv_fold{args.fold}"
    out_dir = Path(args.out or Path(cfg.train.out_dir) / run_name)

    # ---- data ----------------------------------------------------------
    if args.synthetic:
        loaders, matrices = synthetic_loaders(cfg)
        y_train = matrices["train"][1]
    else:
        dataset = primary_dataset(cfg)
        cache_dir = Path(cfg.data.cache_dir) / dataset
        sites_csv = getattr(cfg.task, "sites_csv", None)

        if sites_csv:
            # Per-site task: one sample per tooth position, split by patient.
            methods = tuple(getattr(cfg.task, "site_methods", ("teeth",)))
            jaws = tuple(getattr(cfg.task, "site_jaws", ("lower",)))
            sites = load_sites(artifacts_dir(cfg) / sites_csv, targets=labels,
                               methods=methods, jaws=jaws)
            sites = restrict_sites_to_cache(sites, cache_dir)
            patient_ids, y = patient_label_matrix(sites, labels)
            log.info("%d sites across %d patients | jaws: %s | tiers: %s",
                     len(sites), len(patient_ids), ", ".join(jaws), ", ".join(methods))
        else:
            patient_ids, y = load_label_matrix(
                artifacts_dir(cfg) / f"labels_{dataset}.csv", labels)
            patient_ids, y = restrict_to_cache(patient_ids, y, cache_dir)
            sites = None
        log.info("%d patients with both labels and a cached volume", len(patient_ids))
        if not patient_ids:
            raise SystemExit("no cached volumes found -- run scripts/build_cache.py first")

        if args.fold is None:
            splits = build_splits(cfg, patient_ids, y, labels, force=args.new_splits)
        else:
            folds = build_cv_folds(cfg, patient_ids, y, labels, args.folds, force=args.new_splits)
            splits = fold_assignment(folds, args.fold)
            check_disjoint(splits)
            log.info("cross-validation round %d of %d: test=fold %d, val=fold %d",
                     args.fold, len(folds), args.fold, (args.fold + 1) % len(folds))
        if sites is not None:
            loaders, matrices = make_site_loaders(cfg, splits, sites, cache_dir, args, labels)
        else:
            loaders, matrices = make_loaders(cfg, splits, patient_ids, y, cache_dir, args)
        y_train = matrices["train"][1]
        for name in ("train", "val", "test"):
            log.info("%-5s n=%3d prevalence=%s", name, len(matrices[name][0]),
                     np.round(matrices[name][1].mean(axis=0), 3).tolist())

    # ---- prevalence baseline (free, always reported) --------------------
    # The no-information floor: the loss of a model that ignores the image.
    # Printed before training because a loss is uninterpretable without it, and
    # because the floor moves with the label set -- the previous three-label
    # task's floor was 1.0652 and quoting it here would be a category error.
    binary_names = label_names_for(cfg)
    mm_names = regression_names_for(cfg)
    n_bin = len(binary_names)

    # The standardiser is fitted on the TRAINING split only. Fitting it on
    # everything would leak the validation distribution into the output scale --
    # a small leak, but the kind that is invisible in every metric.
    spec.fit(y_train)

    pos_weight = pos_weight_from_labels(y_train[:, :n_bin]) if n_bin else None
    print()
    print("=" * 62)
    print("NO-INFORMATION FLOOR -- train")
    print("=" * 62)
    if n_bin:
        floor = no_information_bce(y_train[:, :n_bin], pos_weight)
        for name, f, prev in zip(binary_names, floor["per_label"], floor["prevalence"]):
            print(f"  {name:<22} floor {f:6.4f}   prevalence {prev:6.4f}")
        print(f"  {'MACRO (BCE)':<22} floor {floor['floor']:6.4f}")
    if mm_names:
        # Millimetre targets get the same discipline: a model predicting the
        # median still scores something, and an MAE quoted without it is
        # unreadable.
        mm_floor = no_information_regression(y_train[:, n_bin:], mm_names)
        for name, m in mm_floor.items():
            print(f"  {name:<22} MAE floor {m['mae']:6.3f} mm   "
                  f"RMSE floor {m['rmse']:6.3f} mm")
    # NOT comparable to the training loss on the hybrid task. That loss is
    # BCE(binary) + mm_weight * Huber(STANDARDISED mm); the binary floor is a
    # BCE over one block and the millimetre floor is an MAE in RAW mm, so
    # neither is on the loss's scale and their sum is not either -- ~6.9 mm of
    # height floor sits beside a Huber term of order 0.4. Each floor belongs to
    # its own head's metric, which is what `validation_skill` already does.
    print("  each floor belongs to its own head's METRIC, not to the training "
          "loss -- a head at or above its floor has learned nothing")

    if n_bin:
        baseline = PrevalenceBaseline(n_bin).fit(y_train[:, :n_bin])
        for name in ("val", "test"):
            if name not in matrices:
                continue
            y_true = matrices[name][1][:, :n_bin]
            metrics = evaluate(y_true, baseline.predict_proba(len(y_true)), binary_names)
            print("\n" + format_metrics(metrics, binary_names,
                                        f"PREVALENCE BASELINE -- {name}"))

    # ---- model ---------------------------------------------------------
    # The site task sets model.img_size directly and leaves preprocess.out_shape
    # null, because nothing is resampled onto a fixed grid -- patches are cut at
    # the scan's own resolution.
    model = build_model(cfg.model, img_size=model_img_size(cfg))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("%s: %.1fM trainable parameters", cfg.model.name, n_params / 1e6)

    trainer = Trainer(
        model,
        cfg,
        labels,
        pos_weight=pos_weight,
        device=device,
        out_dir=out_dir,
        spec=spec,
    )
    if args.resume:
        trainer.load_checkpoint(args.resume)

    result = trainer.fit(loaders["train"], loaders["val"], epochs=args.epochs)
    log.info("best val %s: %.4f", "skill" if spec.is_hybrid else "macro AUROC",
             result["best_macro_auroc"])

    # ---- final evaluation ----------------------------------------------
    best = out_dir / "best.pt"
    if best.exists():
        trainer.load_checkpoint(best, resume=False)

    def score(split):
        """Binary metrics on the binary block, millimetre metrics on the rest.

        Running `evaluate` over every column treated 18.0 mm as a probability:
        it reported available_height_mm at "prevalence 20.57" with an AP of
        20.226, numbers that are not wrong so much as meaningless. Each block
        gets the metrics its units admit.
        """
        out, y = predict(trainer.model, loaders[split], device, trainer.amp, spec=spec)
        res = {}
        if n_bin:
            # On the site task one patient contributes up to fourteen rows, so
            # the interval is clustered on patients. Passing None here would
            # resample rows and understate every interval -- which is what this
            # call did until v3.4.0.
            ds = loaders[split].dataset
            if getattr(ds, "sites", None) is not None:
                # Site task: one row per tooth position, several per patient.
                groups = ds.sites["patient_id"].to_numpy()
            elif isinstance(ds, TensorDataset) or getattr(ds, "patient_ids", None) is not None:
                # One row per independent unit already: the whole-volume task,
                # and synthetic data where every sample is drawn on its own.
                groups = None
            else:
                raise AttributeError(
                    f"{type(ds).__name__} exposes neither `sites` nor `patient_ids`, "
                    "so the bootstrap cannot tell which rows share a patient. Say "
                    "which explicitly rather than letting it default to a row "
                    "bootstrap -- that silent default is what understated every "
                    "interval before v3.4.0."
                )
            res["classification"] = evaluate(
                y[:, :n_bin], out[:, :n_bin], binary_names,
                n_boot=cfg.eval.bootstrap_n, ci=cfg.eval.bootstrap_ci,
                groups=groups)
        if mm_names:
            res["regression"] = regression_metrics(y[:, n_bin:], out[:, n_bin:], mm_names)
            # Feasibility is recovered HERE, from configuration, across a sweep
            # of thresholds the model never saw. This is the payoff for
            # regressing millimetres, and it is a stronger result than any
            # single threshold could be.
            res["threshold_sensitivity"] = threshold_sensitivity(
                y[:, n_bin:], out[:, n_bin:], mm_names, dict(vars(cfg.sites)),
                jaw=(list(getattr(cfg.task, "site_jaws", ["lower"])) or ["lower"])[0])
        return res, out, y

    val_res, val_out, val_y = score("val")
    if "classification" in val_res:
        print("\n" + format_metrics(val_res["classification"], binary_names,
                                    f"{cfg.model.name.upper()} -- val"))
    if "regression" in val_res:
        print(format_regression(val_res["regression"],
                                f"{cfg.model.name} -- val, millimetres"))
        print()
        print("FEASIBILITY RECOVERED AT INFERENCE (val)")
        print(f"{'height rule':>14}{'measured':>12}{'predicted':>12}{'agreement':>12}")
        for r in val_res["threshold_sensitivity"]:
            print(f"{r['height_threshold_mm']:>11.1f} mm{r['measured_feasible_rate']:>12.3f}"
                  f"{r['predicted_feasible_rate']:>12.3f}{r['agreement']:>12.3f}")
        print("  the threshold is applied here, not compiled into the weights:")
        print("  revising it is a re-score, not five folds of retraining.")

    summary = {"val": val_res, "n_params": n_params, "model": cfg.model.name,
               "labels": labels, "binary": binary_names, "millimetres": mm_names,
               "target_spec": spec.state_dict()}

    # The test split is scored only when asked for. Printing it after every run
    # leaks it by repetition: a number you have seen has influenced you whether
    # or not you tuned a threshold on it.
    if args.test and "test" in loaders:
        log.warning("scoring the TEST split -- do this once, after model selection is final")
        test_res, _, _ = score("test")
        if "classification" in test_res:
            print("\n" + format_metrics(test_res["classification"], binary_names,
                                        f"{cfg.model.name.upper()} -- test"))
        if "regression" in test_res:
            print(format_regression(test_res["regression"],
                                    f"{cfg.model.name} -- test, millimetres"))
        summary["test"] = test_res

    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("wrote %s", out_dir / "metrics.json")


if __name__ == "__main__":
    main()
