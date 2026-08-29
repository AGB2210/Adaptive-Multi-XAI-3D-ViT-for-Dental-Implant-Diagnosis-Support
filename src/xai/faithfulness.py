"""Faithfulness metrics that need no annotation.

These hold on any cohort, including one with no masks at all, which is why they
are the backbone of the evaluation. ToothFairy3 does ship voxel-level masks, so
mask-based localisation scores (IoU, Dice, pointing game) become possible on top
of these -- but as an addition, never a replacement: a saliency map can land
squarely on the implant and still not be what the model actually used, and only
the deletion/insertion tests below can tell the difference.

What is measurable without any annotation:

  deletion / insertion AUC   -- does removing/restoring high-saliency voxels
                                actually move the model's output?
  model-randomisation check  -- does the map degrade as weights are destroyed?
                                (Adebayo et al., 2018)
  inter-method agreement     -- Spearman and top-k Jaccard between methods
  bone-mass plausibility     -- a coarse, threshold-derived proxy; see caveat below
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.xai.base import make_baseline


# --------------------------------------------------------------------------
# rank statistics
# --------------------------------------------------------------------------
def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks, matching scipy.stats.rankdata's tie handling."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sorted_x = x[order]
    i = 0
    while i < len(sorted_x):
        j = i
        while j + 1 < len(sorted_x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(a: np.ndarray, b: np.ndarray, subsample: int = 200_000, seed: int = 0) -> float:
    """Spearman rank correlation. Subsampled -- ranking 2M voxels per pair is
    needless when a 200k sample fixes the estimate to ~0.002."""
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if a.size != b.size:
        raise ValueError(f"size mismatch: {a.size} vs {b.size}")
    if a.size > subsample:
        idx = np.random.default_rng(seed).choice(a.size, subsample, replace=False)
        a, b = a[idx], b[idx]

    ra, rb = _rankdata(a), _rankdata(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def topk_jaccard(a: np.ndarray, b: np.ndarray, k_fraction: float = 0.01) -> float:
    """Jaccard overlap of the top-k% voxels of each map."""
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    k = max(1, int(round(k_fraction * a.size)))
    ta = set(np.argpartition(-a, k - 1)[:k].tolist())
    tb = set(np.argpartition(-b, k - 1)[:k].tolist())
    union = len(ta | tb)
    return float(len(ta & tb) / union) if union else float("nan")


def ssim3d(a: torch.Tensor, b: torch.Tensor, window: int = 7, data_range: float = 1.0) -> float:
    """3D SSIM with a uniform window (no scikit-image dependency)."""
    a = a.float()[None, None]
    b = b.float()[None, None]
    pad = window // 2
    kernel = torch.ones(1, 1, window, window, window, device=a.device) / (window**3)

    def blur(t):
        return F.conv3d(F.pad(t, [pad] * 6, mode="replicate"), kernel)

    mu_a, mu_b = blur(a), blur(b)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sig_a = blur(a * a) - mu_a2
    sig_b = blur(b * b) - mu_b2
    sig_ab = blur(a * b) - mu_ab

    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    ssim = ((2 * mu_ab + c1) * (2 * sig_ab + c2)) / ((mu_a2 + mu_b2 + c1) * (sig_a + sig_b + c2))
    return float(ssim.mean())


# --------------------------------------------------------------------------
# deletion / insertion
# --------------------------------------------------------------------------
@torch.no_grad()
def deletion_insertion(
    model: nn.Module,
    volume: torch.Tensor,
    saliency: torch.Tensor,
    target_label: int,
    baseline: torch.Tensor | None = None,
    steps: int = 100,
    batch_size: int = 8,
    baseline_kind: str = "blur",
    target_is_probability: bool = True,
) -> dict:
    """Deletion and insertion curves and their AUCs.

    Deletion: replace the highest-saliency voxels with the baseline, step by step.
        The target response should FALL fast -> lower AUC is better.
    Insertion: start from the baseline and restore the highest-saliency voxels.
        The target response should RISE fast -> higher AUC is better.

    `target_is_probability=False` for a millimetre head, where the output is a
    standardised length and sigmoid is not the right reading of it. Sigmoid is
    monotone, so it does not reverse a single curve -- but AUC is an integral,
    and the integral of a NONLINEAR monotone transform can reorder two methods
    that the untransformed integral ranks the other way. Un-standardising to
    millimetres would be an affine map, which cannot reorder anything, so the
    raw output is used directly and every comparison here is exactly the
    comparison you would get in millimetres.
    """
    model.eval()
    device = volume.device
    if baseline is None:
        baseline = make_baseline(volume, baseline_kind)

    n_voxels = saliency.numel()
    order = torch.argsort(saliency.reshape(-1), descending=True)
    per_step = max(1, n_voxels // steps)

    def run(start_from_baseline: bool) -> np.ndarray:
        probs = []
        for start in range(0, steps + 1, batch_size):
            batch = []
            for s in range(start, min(start + batch_size, steps + 1)):
                n = min(s * per_step, n_voxels)
                mask = torch.ones(n_voxels, device=device) if not start_from_baseline else torch.zeros(n_voxels, device=device)
                if n > 0:
                    mask[order[:n]] = 0.0 if not start_from_baseline else 1.0
                mask = mask.reshape(1, 1, *saliency.shape)
                batch.append(volume * mask + baseline * (1 - mask))
            if not batch:
                continue
            logits = model(torch.cat(batch, dim=0))
            response = (torch.sigmoid(logits[:, target_label]) if target_is_probability
                        else logits[:, target_label])
            probs.append(response.cpu().numpy())
        return np.concatenate(probs)

    deletion = run(start_from_baseline=False)
    insertion = run(start_from_baseline=True)
    fractions = np.linspace(0.0, 1.0, len(deletion))

    return {
        "deletion_auc": float(np.trapezoid(deletion, fractions)),
        "insertion_auc": float(np.trapezoid(insertion, fractions)),
        "deletion_curve": deletion.tolist(),
        "insertion_curve": insertion.tolist(),
        "fractions": fractions.tolist(),
    }


# --------------------------------------------------------------------------
# model randomisation sanity check (Adebayo et al., 2018)
# --------------------------------------------------------------------------
def randomize_progressively(model: nn.Module) -> list[str]:
    """Names of the parameter groups to destroy, output end first.

    The cascade must reach the INPUT end. An earlier version stopped after the
    transformer blocks, leaving the conv stem and patch embedding intact -- 51.8%
    of this model's parameters untouched, and precisely the layers that give
    gradient-based attributions their edge-like structure.

    That biased the test rather than merely weakening it: attention rollout reads
    the attention matrices inside the blocks, which WERE destroyed, so it
    decorrelated; Integrated Gradients backpropagates through a still-trained
    convolutional front end, so it did not. The apparent conclusion "IG is an
    edge detector, rollout is faithful" was an artefact of which half of the
    network the cascade happened to cover.
    """
    stages = []
    if hasattr(model, "head"):
        stages.append("head")
    if hasattr(model, "norm"):
        stages.append("norm")
    if hasattr(model, "blocks"):
        stages += [f"blocks.{i}" for i in reversed(range(len(model.blocks)))]
    elif hasattr(model, "stages"):
        stages += [f"stages.{i}" for i in reversed(range(len(model.stages)))]
    # The input end, last: patch embedding then the conv stem.
    for name in ("patch_embed", "stem"):
        if hasattr(model, name):
            stages.append(name)
    # Bare Parameters hanging off the model, not inside any submodule. pos_embed
    # shapes attention directly, so leaving it trained would spare rollout.
    for name in ("pos_embed", "cls_token"):
        if isinstance(getattr(model, name, None), nn.Parameter):
            stages.append(name)
    return stages


def reinitialize(module, generator: torch.Generator) -> None:
    """Re-initialise a module (or bare Parameter) to a fresh untrained state.

    Adebayo's protocol RE-INITIALISES; it does not nullify. The distinction is
    not cosmetic here. An earlier version zeroed every 1-D parameter, which
    includes LayerNorm's gain -- and a zeroed gain makes the block emit exactly
    zero, collapsing the whole network into a constant function. Every
    gradient-based attribution then returns exactly 0.0, which looks like a
    perfect pass and is really a measurement of nothing: there is no input
    dependence left to attribute. Stages after the first LayerNorm carried no
    information at all.

    So each module is reset the way PyTorch would build it fresh, via its own
    `reset_parameters` where one exists (LayerNorm -> gain 1, bias 0), and only
    weight matrices with no such method fall back to trunc_normal. Biases go to
    zero, which is their standard init and does not nullify anything.
    """
    if isinstance(module, nn.Parameter):
        # pos_embed and cls_token are learned INPUTS, not gains, so they get the
        # same trunc_normal the model uses to build them -- never zeroed.
        buf = torch.empty(module.shape, device="cpu")
        nn.init.trunc_normal_(buf, std=0.02, generator=generator)
        with torch.no_grad():
            module.copy_(buf.to(module.device))
        return

    # `reset_parameters` draws from the GLOBAL rng, not the generator passed in,
    # so seed it from that generator: the check stays reproducible run to run.
    seed = int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item())
    torch.manual_seed(seed)

    for sub in module.modules():
        reset = getattr(sub, "reset_parameters", None)
        if callable(reset):
            reset()
            continue
        for _, param in sub.named_parameters(recurse=False):
            with torch.no_grad():
                buf = torch.empty(param.shape, device="cpu")
                if param.ndim > 1:
                    nn.init.trunc_normal_(buf, std=0.02, generator=generator)
                else:
                    buf.zero_()
                param.copy_(buf.to(param.device))


def model_randomization_check(
    model: nn.Module,
    method_factory,
    volume: torch.Tensor,
    target_label: int,
    intact_map: torch.Tensor,
    seed: int = 0,
) -> list[dict]:
    """Cascade weight randomisation and watch the saliency map degrade.

    A faithful map should decorrelate from the intact map as weights are
    destroyed. One that barely moves is responding to the input's edges rather
    than to anything the model learned -- that is reported explicitly, not
    smoothed over.
    """
    scrambled = copy.deepcopy(model)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    intact_np = intact_map.detach().cpu().numpy()

    rows = []
    for stage in randomize_progressively(scrambled):
        module = scrambled
        for part in stage.split("."):
            module = module[int(part)] if part.isdigit() else getattr(module, part)

        reinitialize(module, generator)

        method = method_factory(scrambled)
        try:
            degraded = method.attribute(volume, target_label)
        except Exception as exc:  # noqa: BLE001 - a broken stage is a result, not a crash
            rows.append({"stage": stage, "error": f"{type(exc).__name__}: {exc}"})
            continue

        degraded_np = degraded.detach().cpu().numpy()
        # Once the input end is destroyed a method can return a perfectly flat
        # map. Spearman against a constant is undefined, not zero, and the two
        # must not be conflated: a flat map is the STRONGEST possible pass of
        # this check, so it is flagged rather than left as a bare nan.
        constant = bool(np.ptp(degraded_np) == 0)
        rows.append({
            "stage": stage,
            "spearman_vs_intact": spearman(intact_np, degraded_np),
            "ssim_vs_intact": ssim3d(intact_map, degraded),
            "degraded_is_constant": constant,
        })
    return rows


# --------------------------------------------------------------------------
# inter-method agreement
# --------------------------------------------------------------------------
def agreement_matrix(maps: dict[str, torch.Tensor], k_fractions=(0.01, 0.05)) -> dict:
    names = list(maps)
    arrays = {n: maps[n].detach().cpu().numpy().ravel() for n in names}

    out: dict = {"methods": names, "spearman": {}, "jaccard": {f"top{int(k*100)}pct": {} for k in k_fractions}}
    for i, a in enumerate(names):
        for b in names[i:]:
            rho = spearman(arrays[a], arrays[b])
            out["spearman"][f"{a}|{b}"] = rho
            for k in k_fractions:
                out["jaccard"][f"top{int(k*100)}pct"][f"{a}|{b}"] = topk_jaccard(arrays[a], arrays[b], k)
    return out


# --------------------------------------------------------------------------
# bone-mass plausibility proxy
# --------------------------------------------------------------------------
def bone_mask(volume: torch.Tensor, percentile: float = 95.0) -> torch.Tensor:
    """Dense-structure mask by intensity threshold on the z-scored volume.

    CAVEAT (state this in any report that uses it): this is a coarse, threshold-
    derived proxy, NOT clinical ground truth. Enamel and cortical bone are the
    brightest structures in CBCT, so a high percentile mostly selects them -- but
    it cannot distinguish WHICH anatomy. It says nothing about the inferior
    alveolar canal, or any specific structure. It is a plausibility check only.
    """
    v = volume.detach().float().reshape(-1)
    threshold = torch.quantile(v[:: max(1, v.numel() // 1_000_000)], percentile / 100.0)
    return (volume.detach().float() > threshold).squeeze()


def bone_mass_fraction(saliency: torch.Tensor, mask: torch.Tensor) -> dict:
    """Fraction of total saliency mass inside the dense-structure mask."""
    s = saliency.detach().float()
    m = mask.to(s.device).bool()
    total = float(s.sum())
    if total <= 0:
        return {"bone_mass_fraction": float("nan"), "mask_volume_fraction": float(m.float().mean())}

    inside = float(s[m].sum())
    mask_fraction = float(m.float().mean())
    return {
        "bone_mass_fraction": inside / total,
        "mask_volume_fraction": mask_fraction,
        # >1 means saliency concentrates on dense structure more than chance.
        "enrichment": (inside / total) / mask_fraction if mask_fraction > 0 else float("nan"),
    }
