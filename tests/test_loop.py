"""The training loop, which had no test file at all.

CI exercises it end to end on synthetic data -- config, loaders, model, loss,
metrics, checkpoint -- so it was never unexercised. But nothing pinned the
pieces that decide *which model ships* or *what update is applied*, and both
fail silently when wrong: a mis-weighted accumulation trains a slightly
different model than the config asks for, and a checkpoint rule that mishandles
NaN keeps whichever epoch happened to run first.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.train import loop
from src.train.loop import Trainer, cosine_warmup, predict

# --- the LR schedule ----------------------------------------------------------

def test_warmup_rises_from_near_zero_to_one():
    warm, total = 10, 100
    assert cosine_warmup(0, warm, total) == pytest.approx(0.1)
    assert cosine_warmup(warm - 1, warm, total) == pytest.approx(1.0)


def test_cosine_decays_to_min_ratio_and_never_below():
    warm, total, floor = 10, 100, 0.01
    values = [cosine_warmup(s, warm, total, floor) for s in range(total + 20)]
    assert min(values) >= floor - 1e-12, "the schedule must not undershoot its floor"
    assert values[-1] == pytest.approx(floor, abs=1e-9)


def test_the_schedule_is_monotone_after_warmup():
    warm, total = 10, 100
    after = [cosine_warmup(s, warm, total) for s in range(warm, total + 1)]
    assert all(b <= a + 1e-12 for a, b in zip(after, after[1:])), (
        "cosine decay must not rise again mid-run")


def test_a_step_past_the_end_is_clamped_not_extrapolated():
    """Past total_steps the cosine would turn back up if progress were unclamped."""
    warm, total = 5, 50
    assert cosine_warmup(total * 3, warm, total) == pytest.approx(
        cosine_warmup(total, warm, total))


# --- gradient accumulation ----------------------------------------------------

def _tiny_cfg(accum: int, lr: float = 0.1):
    return SimpleNamespace(
        seed=0,
        train=SimpleNamespace(
            epochs=1, batch_size=2, accum_steps=accum, lr=lr, weight_decay=0.0,
            warmup_epochs=0, grad_clip=0.0, amp=False, num_workers=0,
            early_stop_patience=0, out_dir="artifacts/_test_loop",
        ),
        eval=SimpleNamespace(bootstrap_n=0, bootstrap_ci=0.95),
    )


class _Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 1)

    def forward(self, x):
        return self.fc(x)


def _loader(x, y, batch_size):
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def _train_once(accum: int, batch_size: int, monkeypatch, seed: int = 0):
    """One epoch with the LR schedule pinned flat.

    `train_epoch` sets the LR per MICRO-BATCH from `cosine_warmup(step)`, so an
    accum=4 run and an accum=1 run reach their optimiser step having last seen
    different multipliers. That is a property of the schedule, not of
    accumulation, and it has to be held constant or this test measures the wrong
    thing. (It is also worth knowing on its own: the effective LR of an
    accumulated step is whichever micro-batch happened to be last in the window.)
    """
    monkeypatch.setattr(loop, "cosine_warmup", lambda *a, **k: 1.0)

    torch.manual_seed(seed)
    model = _Linear()
    start = model.fc.weight.detach().clone()

    torch.manual_seed(123)
    x = torch.randn(8, 4)
    y = (x.sum(1, keepdim=True) > 0).float()

    trainer = Trainer(model, _tiny_cfg(accum), ["a"], device=torch.device("cpu"))
    trainer.train_epoch(_loader(x, y, batch_size), epoch=0, total_steps=4, warmup_steps=0)
    return start, model.fc.weight.detach().clone()


def test_accumulation_matches_the_equivalent_single_step(monkeypatch):
    """Four batches of 2 with accum=4 must move the weights like one batch of 8.

    If the loss were not divided by accum, or the optimiser stepped on the wrong
    batches, this diverges -- and nothing else in the run would say so.
    """
    _, w_accum = _train_once(accum=4, batch_size=2, monkeypatch=monkeypatch)
    _, w_single = _train_once(accum=1, batch_size=8, monkeypatch=monkeypatch)
    assert torch.allclose(w_accum, w_single, atol=1e-6), (
        f"accumulated update {w_accum} differs from the single-batch update {w_single}")


def test_accumulation_actually_moves_the_weights(monkeypatch):
    """Guard the guard: two identical no-ops would also pass the test above."""
    start, end = _train_once(accum=4, batch_size=2, monkeypatch=monkeypatch)
    assert not torch.allclose(start, end), "the epoch applied no update at all"


# --- predict ------------------------------------------------------------------

def test_predict_returns_one_row_per_sample_in_order():
    torch.manual_seed(0)
    model = _Linear().eval()
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    y = torch.zeros(3, 1)
    out, truth = predict(model, _loader(x, y, 2), torch.device("cpu"))
    assert out.shape == (3, 1)
    assert truth.shape == (3, 1)
    # `predict` is the REPORTING path: it sigmoids the binary block, so compare
    # against sigmoid(logits), not the logits. Order must survive batching, or
    # every metric pairs the wrong rows.
    with torch.no_grad():
        direct = torch.sigmoid(model(x)).numpy()
    assert np.allclose(out, direct, atol=1e-6)
