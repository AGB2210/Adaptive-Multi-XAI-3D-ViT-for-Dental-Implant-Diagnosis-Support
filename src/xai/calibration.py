"""Temperature scaling and calibration diagnostics.

Raw sigmoid outputs are not probabilities. A confidence gate built on
uncalibrated scores is gating on an arbitrary monotone transform of the logits,
so calibration is a prerequisite for the adaptive layer, not a nicety.

Temperature is fitted on the VALIDATION split only.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def fit_temperature(
    logits: np.ndarray,
    targets: np.ndarray,
    max_iter: int = 200,
    lr: float = 0.01,
) -> float:
    """Single scalar temperature minimising BCE on the validation split.

    One temperature for all labels: with ~60 validation patients, per-label
    temperatures would be fitted on ~25 positives each and would overfit.
    """
    z = torch.tensor(np.asarray(logits), dtype=torch.float32)
    y = torch.tensor(np.asarray(targets), dtype=torch.float32)

    log_t = torch.zeros(1, requires_grad=True)  # optimise log T to keep T > 0
    optimizer = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(z / log_t.exp(), y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().item())


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(logits) / temperature))


def expected_calibration_error(
    probs: np.ndarray, targets: np.ndarray, n_bins: int = 10
) -> tuple[float, list[dict]]:
    """ECE over all label predictions pooled, plus per-bin detail for the diagram.

    Multi-label, so every (patient, label) pair is one binary prediction.
    """
    p = np.asarray(probs, dtype=np.float64).ravel()
    y = np.asarray(targets, dtype=np.float64).ravel()

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, bins = 0.0, []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        n = int(mask.sum())
        if n == 0:
            bins.append({"bin_lo": lo, "bin_hi": hi, "n": 0, "confidence": np.nan, "accuracy": np.nan})
            continue
        confidence = float(p[mask].mean())
        accuracy = float(y[mask].mean())
        ece += (n / len(p)) * abs(accuracy - confidence)
        bins.append({"bin_lo": lo, "bin_hi": hi, "n": n, "confidence": confidence, "accuracy": accuracy})

    return float(ece), bins


def reliability_diagram(bins_before: list[dict], bins_after: list[dict],
                        ece_before: float, ece_after: float, path, title: str = "") -> None:
    """Reliability diagram before vs after temperature scaling."""
    import matplotlib

    matplotlib.use("Agg")
    from pathlib import Path

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, bins, ece, name in (
        (axes[0], bins_before, ece_before, "before"),
        (axes[1], bins_after, ece_after, "after"),
    ):
        centres = [(b["bin_lo"] + b["bin_hi"]) / 2 for b in bins]
        acc = [b["accuracy"] for b in bins]
        counts = [b["n"] for b in bins]

        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
        ax.bar(centres, acc, width=0.09, color="#4C72B0", edgecolor="white", label="observed frequency")
        ax.set_xlabel("predicted probability")
        ax.set_title(f"{name} temperature scaling\nECE = {ece:.4f}  (n = {sum(counts):,} predictions)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("observed positive rate")
    axes[0].legend(loc="upper left", fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def uncertainty(probs: np.ndarray, mode: str = "margin") -> np.ndarray:
    """Per-case uncertainty. Higher = less certain.

    margin  : min over labels of |p - 0.5|, negated -- how close any label sits
              to the decision boundary (the spec's default).
    entropy : mean per-label binary entropy.
    """
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-7, 1 - 1e-7)
    if mode == "margin":
        return -np.min(np.abs(p - 0.5), axis=1)
    if mode == "entropy":
        return np.mean(-(p * np.log(p) + (1 - p) * np.log(1 - p)), axis=1)
    raise ValueError(f"unknown uncertainty mode {mode!r} (expected 'margin' or 'entropy')")
