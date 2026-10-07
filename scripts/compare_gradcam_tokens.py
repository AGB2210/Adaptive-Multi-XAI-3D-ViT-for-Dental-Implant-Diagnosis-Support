"""Score both Grad-CAM token sources on a task where the answer is known.

    python scripts/compare_gradcam_tokens.py [--seeds 0 1 2 3] [--cases 60]

`src/xai/gradcam.py` explains why there are two. This script is where the
numbers in that docstring come from, so they can be reproduced and not merely
quoted.

A ViT3D is trained on the planted-signal task (`tests/synthetic.py`: a blob at a
known site per label) until its validation macro AUROC passes 0.97 -- against a
model that has not learned the signal the comparison tests nothing. Each method
is then scored on unseen single-label cases:

  empty maps     share of cases where the map is zero everywhere. Grad-CAM
                 ends in a ReLU, so a case with no positive evidence at that
                 layer has no map at all. Counted, not averaged in: an empty
                 map has no enrichment, and its deletion curve would follow
                 voxel order.
  enrichment     saliency mass inside the planted region over the mass a
                 uniform map would put there. 1.0 is chance. Median over the
                 maps that are not empty.
  above chance   share of ALL cases with enrichment over 1.0, so an empty map
                 counts against the method.
  deletion AUC   area under the class probability as the highest-saliency
                 voxels are removed. Lower: the map found what the model uses.
                 Mean over the maps that are not empty.

Training on a GPU is not bit-reproducible, so the figures move from run to run
by more than their last digit. Read the pattern across seeds, not one row.

Integrated Gradients and attention rollout are scored beside the two Grad-CAMs
as a scale. Needs no dataset and no checkpoint; a few minutes on a CPU, seconds
per seed on a GPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.vit3d import ViT3D  # noqa: E402
from src.train.metrics import auroc  # noqa: E402
from src.xai import build_method  # noqa: E402
from src.xai.faithfulness import deletion_insertion  # noqa: E402
from tests.synthetic import make_case, make_dataset, signal_mask  # noqa: E402

SHAPE, LABELS = (32, 32, 32), 6
METHODS = ("gradcam", "gradcam_input", "integrated_gradients", "attention_rollout")
LEARNED = 0.97


def train(seed: int, device: torch.device, max_epochs: int = 40) -> tuple[ViT3D, float]:
    """A small ViT that has learned the planted signal, and its validation macro AUROC."""
    torch.manual_seed(seed)
    model = ViT3D(num_classes=LABELS, img_size=SHAPE[0], stem_channels=16, embed_dim=128,
                  patch_size=4, depth=4, num_heads=4, drop_path=0.0).to(device)
    x, y = make_dataset(3000, seed=10_000 + seed, shape=SHAPE, n_labels=LABELS)
    xv, yv = make_dataset(400, seed=90_000 + seed, shape=SHAPE, n_labels=LABELS)
    x, y, xv = torch.from_numpy(x), torch.from_numpy(y).float(), torch.from_numpy(xv).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    macro = 0.0
    for _ in range(max_epochs):
        model.train()
        order = torch.randperm(len(x))
        for i in range(0, len(order), 32):
            idx = order[i:i + 32]
            loss = F.binary_cross_entropy_with_logits(model(x[idx].to(device)), y[idx].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(torch.cat([model(xv[i:i + 100]) for i in range(0, len(xv), 100)]))
        p = p.cpu().numpy()
        macro = float(np.mean([auroc(yv[:, j], p[:, j]) for j in range(LABELS)]))
        if macro > LEARNED:
            break
    for param in model.parameters():
        param.requires_grad_(False)
    return model.eval(), macro


def score(model: ViT3D, device: torch.device, n_cases: int) -> dict:
    methods = {name: build_method(name, model, device) for name in METHODS}
    out = {name: {"enrichment": [], "deletion": []} for name in METHODS}
    for case in range(n_cases):
        labels = np.zeros(LABELS, dtype=np.int64)
        target = case % LABELS
        labels[target] = 1
        volume = torch.from_numpy(make_case(500_000 + case, shape=SHAPE, labels=labels)[0])
        volume = volume[None, None].to(device)
        mask = torch.from_numpy(signal_mask(labels, shape=SHAPE)).to(device)
        chance = float(mask.float().mean())
        for name, method in methods.items():
            saliency = method.attribute(volume, target)
            total = float(saliency.sum())
            if total <= 0:
                out[name]["enrichment"].append(float("nan"))
                out[name]["deletion"].append(float("nan"))
                continue
            out[name]["enrichment"].append(float(saliency[mask].sum()) / total / chance)
            out[name]["deletion"].append(
                deletion_insertion(model, volume, saliency, target, steps=32)["deletion_auc"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3])
    ap.add_argument("--cases", type=int, default=60)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{'seed':>5}{'val AUROC':>11}  {'method':22}{'empty maps':>12}{'median enrichment':>19}"
          f"{'above chance':>14}{'deletion AUC':>14}")
    for seed in args.seeds:
        model, macro = train(seed, device)
        if macro <= LEARNED:
            print(f"{seed:>5}{macro:>11.3f}  did not learn the planted signal; not scored")
            continue
        for name, r in score(model, device, args.cases).items():
            e = np.asarray(r["enrichment"], dtype=float)
            d = np.asarray(r["deletion"], dtype=float)
            some = np.isfinite(e).any()
            print(f"{seed:>5}{macro:>11.3f}  {name:22}{np.mean(~np.isfinite(e)):>12.2f}"
                  f"{np.nanmedian(e) if some else float('nan'):>19.2f}{np.mean(e > 1.0):>14.2f}"
                  f"{np.nanmean(d) if some else float('nan'):>14.4f}")
        print()


if __name__ == "__main__":
    main()
