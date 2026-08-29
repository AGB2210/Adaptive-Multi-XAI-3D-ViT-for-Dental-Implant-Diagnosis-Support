"""Score an explanation against the ground-truth mask of what it should point at.

Every other faithfulness measure in this project is annotation-free: deletion
and insertion ask whether the model's own score moves when the highlighted
voxels are removed, which tests self-consistency, not correctness. A method can
be perfectly self-consistent and still be pointing at the wrong anatomy.

ToothFairy3 ships voxel masks, so for once the question "is the explanation
pointing at the inferior alveolar canal?" has a ground-truth answer. That is
what makes this a testbed rather than a demo: the label is a known deterministic
function of a known, voxel-localised structure, so we know exactly what evidence
a correct model must use.

Choosing the metric matters more than it looks. The canal crossing a 96^3 patch
at 0.3 mm occupies a few thousand voxels out of 884,736 -- a fraction of a
percent. Against a target that small:

  * IoU and Dice are near-zero for ANY diffuse map and mostly measure how
    peaked a method is, not whether it is correct. They are reported at a fixed
    top-k so they are at least comparable between methods, never as a headline.
  * mass-inside-mask is interpretable but tiny for everything, so raw values
    look like failure even when a method is working well.
  * ENRICHMENT -- observed mass inside the mask divided by the mass a uniform
    map would put there -- is the honest primary number. 1.0 is chance. 50x is
    a method that has genuinely found the object.
  * the POINTING GAME (does the single hottest voxel land inside the mask?) is
    the strictest and the easiest to explain in a paper.

A caveat that belongs beside any result from this module: attention rollout and
Grad-CAM are native to the TOKEN grid, not the voxel grid. On the site config
that is 12x12x12 tokens over a 96^3 patch, so one token covers 2.4 mm -- the
same order as the 2-3 mm canal, which is why model.patch_size is 4 and not 8.
At patch_size 8 a token spans 4.8 mm, wider than the structure it is being
scored against, and the comparison stops meaning anything.

Read the grid off the model rather than from this docstring. run_xai.py prints
it from the checkpoint and warns when a token is wider than the canal. Their
apparent voxel precision is upsampling, not evidence, and they are penalised
here for a resolution limit rather than for being wrong. Compare them to each
other, and to the per-voxel methods only with that stated.

An achievable-ceiling control is what would put a scale under these numbers: run
the same methods on the synthetic planted-signal task, where the correct answer
is known exactly, and report enrichment relative to what each method can reach
at its own resolution. Without it, "3.2x enrichment" has no denominator. Not
implemented yet; it is CPU-only and cheap, and it would strengthen every
localisation result in the paper.

BUT IT CANNOT BE BUILT ON THE EXISTING PLANTED SIGNAL UNCHANGED. Enrichment is
bounded by resolution, so a ceiling transfers between two targets only if they
have a similar SHAPE, not merely a similar volume. A token here is `patch_size`
4 times the conv stem's stride 2, so 8 input voxels, which is 2.4 mm:

    planted blob   0.71% of the patch, compact sphere, ~2.9 tokens across
    nerve canal    0.48% of the patch, long thin tube, ~0.94 tokens across

The fractions are within 1.5x of each other, which is exactly what makes this
easy to miss. The aspect ratios are not close at all: a method that places mass
at token granularity can fill a three-token sphere and cannot fill a sub-token
tube. A ceiling measured on the blob would come out optimistic, and every real
method -- GradCAM and rollout worst -- would be scored against a bar their
resolution never allowed them to reach.

Whoever writes the control must plant a TUBE with the canal's cross section.
`tests/test_localization.py::TestTheCeilingControlPremise` pins both numbers so
the premise is checked rather than assumed. The second premise, that the
synthetic millimetre heads are trained on lengths rather than on 0/1, does
already hold -- see `tests/synthetic.make_hybrid_dataset` -- and is pinned in
the same place.
"""

from __future__ import annotations

import numpy as np
import torch


def _as_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def pointing_game(saliency, mask) -> bool:
    """Does the single highest-attribution voxel fall inside the mask?

    Strict, unsmoothed, and the standard formulation. With ties (common for
    upsampled token maps, where a whole 16 mm block shares one value) argmax
    takes the first, which is arbitrary but not biased toward the mask.
    """
    s, m = _as_numpy(saliency), _as_numpy(mask) > 0
    if not m.any():
        raise ValueError("pointing_game needs a non-empty mask")
    return bool(m.flat[int(np.argmax(s))])


def mass_inside(saliency, mask) -> float:
    """Share of total attribution mass that lands inside the mask."""
    s, m = _as_numpy(saliency), _as_numpy(mask) > 0
    s = np.clip(s, 0.0, None)
    total = float(s.sum())
    return float(s[m].sum() / total) if total > 0 else float("nan")


def enrichment(saliency, mask) -> float:
    """mass_inside divided by the mask's share of the volume.

    The chance-corrected number, and the one to lead with: 1.0 means the method
    put no more mass on the object than spreading it uniformly would.
    """
    m = _as_numpy(mask) > 0
    chance = float(m.mean())
    if chance <= 0:
        raise ValueError("enrichment needs a non-empty mask")
    inside = mass_inside(saliency, m)
    return float(inside / chance) if np.isfinite(inside) else float("nan")


def overlap_at_topk(saliency, mask, k_fraction: float = 0.001) -> dict:
    """IoU and Dice after keeping the top k_fraction of voxels.

    k_fraction defaults to 0.1%, roughly 2,000 voxels at 128^3 -- the same order
    as a large restoration, so the binarised map and the target are at least
    comparable in size. A fixed k also keeps methods comparable: thresholding
    each map at its own value would reward whichever method happens to be
    peakiest.
    """
    s, m = _as_numpy(saliency), _as_numpy(mask) > 0
    n_keep = max(1, int(round(k_fraction * s.size)))
    flat = s.reshape(-1)
    top = np.zeros(flat.shape, dtype=bool)
    top[np.argpartition(-flat, n_keep - 1)[:n_keep]] = True
    top = top.reshape(s.shape)

    inter = float(np.logical_and(top, m).sum())
    union = float(np.logical_or(top, m).sum())
    return {
        "iou": inter / union if union > 0 else float("nan"),
        "dice": 2 * inter / (top.sum() + m.sum()) if (top.sum() + m.sum()) > 0 else float("nan"),
        "topk_voxels": n_keep,
        "mask_voxels": int(m.sum()),
    }


def localization_scores(saliency, mask, k_fraction: float = 0.001) -> dict:
    """Every metric for one (saliency, mask) pair."""
    m = _as_numpy(mask) > 0
    scores = {
        "pointing_hit": pointing_game(saliency, m),
        "mass_inside": mass_inside(saliency, m),
        "enrichment": enrichment(saliency, m),
        "mask_fraction": float(m.mean()),
    }
    scores.update(overlap_at_topk(saliency, m, k_fraction))
    return scores


def competing_structure_ratio(saliency, target_mask, other_mask) -> float:
    """Enrichment on the target divided by enrichment on a competing structure.

    This is the project's central question in one number. In ToothFairy3, 68 of
    74 implant cases also contain a crown or bridge, so a model can score well on
    `implant` by detecting the restoration sitting on top of it. If the implant
    explanation is really about the implant this is > 1; at or below 1 the
    explanation is landing on the neighbour, and the classifier's implant
    performance should not be described as implant detection.

    Returns nan when either structure is absent from the case -- there is no
    comparison to make. Returns inf when the competitor receives no mass at all,
    which is the strongest possible result and must not be confused with the
    absent case; aggregate these with a MEDIAN, since a single inf destroys a
    mean.
    """
    t, o = _as_numpy(target_mask) > 0, _as_numpy(other_mask) > 0
    if not t.any() or not o.any():
        return float("nan")

    num, denom = enrichment(saliency, t), enrichment(saliency, o)
    if not np.isfinite(num) or not np.isfinite(denom):
        return float("nan")
    if denom == 0:
        # No mass on the competitor at all. Zero on both means the map avoided
        # both structures, which is uninformative, not a win.
        return float("inf") if num > 0 else float("nan")
    return float(num / denom)
