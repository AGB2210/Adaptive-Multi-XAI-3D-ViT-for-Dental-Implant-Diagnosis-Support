"""Train the site localiser on one cross-validation round.

    python scripts/train_localiser.py --config configs/localiser.yaml --fold 0

Uses `cv_folds.json` -- the SAME partition as the site model -- so the
localiser for round k has never seen round k's test patients, and the
end-to-end error `eval_localiser.py` measures on them is not optimistic.
Selected on validation median 3D position error; the test fold is not touched.

Writes <localiser.out_dir>/cv_fold<k>/best.pt, last.pt and history.csv. The
checkpoint carries `kind: localiser`, its config, and its validation numbers,
so the app can be handed it with nothing beside it.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.localiser import SiteLocaliser, localiser_loss  # noqa: E402
from src.train.localiser import localiser_datasets, predict_dataset, summarise  # noqa: E402
from src.train.loop import cosine_warmup  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.log import get_logger  # noqa: E402
from src.utils.seed import set_seed, worker_init_fn  # noqa: E402

log = get_logger("train_localiser")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/localiser.yaml")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0, help="smoke-test only: patients per split")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    loc = cfg.localiser
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lcfg, data = localiser_datasets(cfg, args.fold, args.limit)
    log.info("fold %d: %d train, %d val patients | device %s", args.fold,
             len(data["train"]), len(data["val"]), device)

    loader = DataLoader(data["train"], batch_size=int(loc.batch_size), shuffle=True,
                        num_workers=int(getattr(loc, "num_workers", 0)), drop_last=False,
                        worker_init_fn=worker_init_fn,
                        persistent_workers=int(getattr(loc, "num_workers", 0)) > 0)
    model = SiteLocaliser(lcfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(loc.lr), weight_decay=float(loc.weight_decay))
    epochs = int(args.epochs or loc.epochs)
    total, warm = epochs * max(1, len(loader)), max(1, len(loader))
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    voxel_mm = lcfg.factor * lcfg.spacing_mm

    out_dir = Path(args.out or Path(loc.out_dir) / f"cv_fold{args.fold}")
    out_dir.mkdir(parents=True, exist_ok=True)
    best, since, patience = math.inf, 0, int(getattr(loc, "early_stop_patience", 0) or 0)
    history = []

    def save(name, epoch, val):
        payload = {"kind": "localiser", "epoch": epoch, "fold": args.fold,
                   "model": model.state_dict(), "localiser_config": lcfg.to_dict(),
                   "val": val, "config": str(args.config)}
        tmp = out_dir / (name + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(out_dir / name)

    for epoch in range(epochs):
        model.train()
        t0, losses = time.time(), []
        for step, (x, coords, mask, valid, flipped) in enumerate(loader):
            for g in opt.param_groups:
                g["lr"] = float(loc.lr) * cosine_warmup(epoch * len(loader) + step, warm, total)
            x, coords, mask = x.to(device), coords.to(device), mask.to(device)
            valid, flipped = valid.to(device), flipped.to(device)
            with torch.autocast(device_type=device.type, enabled=amp):
                out = model(x)
            loss = localiser_loss({k: v.float() for k, v in out.items()}, coords, mask,
                                  valid, flipped, voxel_mm,
                                  spread_weight=float(getattr(loc, "spread_weight", 0.05)))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss["total"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss["coord_mm"].detach()))

        upright_val = predict_dataset(model, data["val"], device, flip=False)
        flipped_val = predict_dataset(model, data["val"], device, flip=True)
        val = summarise(upright_val, lcfg.factor, lcfg.spacing_mm)
        # Both ways up: a head that always answered "upright" would score 50%.
        val["orientation_accuracy"] = float(np.mean(
            np.concatenate([flipped_val["flip_prob"] >= 0.5, upright_val["flip_prob"] < 0.5])))
        row = {"epoch": epoch, "train_coord_mm": round(float(np.mean(losses)), 4),
               **{k: round(v, 4) if isinstance(v, float) else v for k, v in val.items()},
               "seconds": round(time.time() - t0, 1)}
        history.append(row)
        with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)

        score = val["median_error_mm"]
        improved = np.isfinite(score) and score < best
        if improved:
            best, since = score, 0
            save("best.pt", epoch, val)
        else:
            since += 1
        save("last.pt", epoch, val)
        log.info("epoch %3d | train %.2f mm | val median %.2f mm, p90 %.2f mm, orientation %.3f%s",
                 epoch, row["train_coord_mm"], val["median_error_mm"], val["p90_error_mm"],
                 val["orientation_accuracy"], "  <- best" if improved else "")
        if patience and since >= patience:
            log.info("early stopping: no validation improvement for %d epochs", patience)
            break

    (out_dir / "summary.json").write_text(json.dumps(
        {"fold": args.fold, "best_val_median_error_mm": best, "epochs_run": len(history)},
        indent=2), encoding="utf-8")
    log.info("best validation median error %.2f mm -> %s", best, out_dir / "best.pt")


if __name__ == "__main__":
    main()
