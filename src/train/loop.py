"""Training loop: AMP, cosine schedule with warmup, gradient clipping, checkpoint/resume."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.train.metrics import evaluate
from src.utils.log import get_logger

log = get_logger("train")


def load_checkpoint_file(path, map_location):
    """torch.load for our own checkpoints, preferring the safe path.

    weights_only=True refuses to unpickle arbitrary objects, which is what makes
    a downloaded .pt file safe to open. Our checkpoints also carry plain metadata
    (epoch, label names, metrics), so if the safe loader rejects one we retry the
    permissive path and say so -- loudly, because at that point the file is being
    trusted to run code.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:  # noqa: BLE001 - any rejection means fall back
        log.warning(
            "%s could not be read with weights_only=True (%s); retrying with "
            "weights_only=False. Only do this for checkpoints you produced yourself.",
            path, type(exc).__name__,
        )
        return torch.load(path, map_location=map_location, weights_only=False)


def cosine_warmup(step: int, warmup_steps: int, total_steps: int, min_ratio: float = 0.01) -> float:
    """LR multiplier: linear warmup, then cosine decay to min_ratio."""
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Returns (probabilities, targets)."""
    model.eval()
    probs, targets = [], []
    autocast = torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda")
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with autocast:
            logits = model(x)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(probs), np.concatenate(targets)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        cfg,
        label_names: list[str],
        pos_weight: torch.Tensor | None = None,
        device: torch.device | None = None,
        out_dir: str | Path | None = None,
    ):
        self.cfg = cfg
        self.label_names = label_names
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.out_dir = Path(out_dir or cfg.train.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        if pos_weight is not None:
            pos_weight = pos_weight.to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # No weight decay on norms, biases, or the positional/CLS parameters.
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            (no_decay if param.ndim <= 1 or name.endswith(("pos_embed", "cls_token")) else decay).append(param)
        self.optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": cfg.train.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=cfg.train.lr,
            betas=(0.9, 0.999),
        )

        # AMP only helps on CUDA; on CPU it is a slowdown.
        self.amp = bool(cfg.train.amp) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)

        self.start_epoch = 0
        self.best_macro_auroc = -1.0
        self.history: list[dict] = []
        self.writer = None

    def _tensorboard(self):
        if self.writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(str(self.out_dir / "tb"))
            except ImportError:
                self.writer = False  # tensorboard not installed; CSV logging still works
        return self.writer or None

    def train_epoch(self, loader: DataLoader, epoch: int, total_steps: int, warmup_steps: int) -> float:
        self.model.train()
        accum = max(1, int(getattr(self.cfg.train, "accum_steps", 1)))
        running, n_batches = 0.0, 0
        autocast = torch.autocast(device_type=self.device.type, enabled=self.amp)

        self.optimizer.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(loader):
            step = epoch * len(loader) + i
            lr_scale = cosine_warmup(step, warmup_steps, total_steps)
            for group in self.optimizer.param_groups:
                group["lr"] = self.cfg.train.lr * lr_scale

            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            with autocast:
                loss = self.criterion(self.model(x), y) / accum

            self.scaler.scale(loss).backward()

            if (i + 1) % accum == 0 or (i + 1) == len(loader):
                if self.cfg.train.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            running += float(loss.item()) * accum
            n_batches += 1

        return running / max(1, n_batches)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int | None = None) -> dict:
        epochs = epochs or self.cfg.train.epochs
        total_steps = epochs * len(train_loader)
        warmup_steps = int(self.cfg.train.warmup_epochs * len(train_loader))
        patience = int(getattr(self.cfg.train, "early_stop_patience", 0) or 0)
        since_best = 0

        for epoch in range(self.start_epoch, epochs):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader, epoch, total_steps, warmup_steps)

            probs, targets = predict(self.model, val_loader, self.device, self.amp)
            metrics = evaluate(targets, probs, self.label_names)
            macro = metrics["macro_auroc"]

            row = {
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "val_macro_auroc": round(macro, 5) if np.isfinite(macro) else float("nan"),
                "val_macro_ap": round(metrics["macro_ap"], 5),
                "val_macro_f1": round(metrics["macro_f1"], 5),
                "lr": self.optimizer.param_groups[0]["lr"],
                "seconds": round(time.time() - t0, 1),
            }
            self.history.append(row)
            self._log_row(row)

            improved = np.isfinite(macro) and macro > self.best_macro_auroc
            if improved:
                self.best_macro_auroc = macro
                since_best = 0
                self.save_checkpoint(epoch, "best.pt", metrics=metrics)
            else:
                since_best += 1
            self.save_checkpoint(epoch, "last.pt")

            log.info(
                "epoch %3d | loss %.4f | val macro AUROC %.4f%s | %.0fs",
                epoch, train_loss, macro, "  <- best" if improved else "", row["seconds"],
            )

            if patience and since_best >= patience:
                log.info("early stopping: no val improvement for %d epochs", patience)
                break

        return {"best_macro_auroc": self.best_macro_auroc, "history": self.history}

    def _log_row(self, row: dict) -> None:
        path = self.out_dir / "history.csv"
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        tb = self._tensorboard()
        if tb:
            for key, value in row.items():
                if key != "epoch" and isinstance(value, (int, float)):
                    tb.add_scalar(key, value, row["epoch"])

    def save_checkpoint(self, epoch: int, name: str, metrics: dict | None = None) -> None:
        payload = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_macro_auroc": self.best_macro_auroc,
            "label_names": self.label_names,
        }
        torch.save(payload, self.out_dir / name)
        if metrics is not None:
            (self.out_dir / "best_val_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    def load_checkpoint(self, path: str | Path, resume: bool = True) -> None:
        ckpt = load_checkpoint_file(path, self.device)
        self.model.load_state_dict(ckpt["model"])
        if resume:
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scaler.load_state_dict(ckpt["scaler"])
            self.start_epoch = ckpt["epoch"] + 1
            self.best_macro_auroc = ckpt.get("best_macro_auroc", -1.0)
            log.info("resumed from %s at epoch %d", path, self.start_epoch)
