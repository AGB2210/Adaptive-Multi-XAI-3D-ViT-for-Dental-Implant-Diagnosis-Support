"""A hand-crafted geometric estimator: the baseline that can end the paper.

WHY THIS EXISTS
---------------
`available_height_mm` and `ridge_width_mm` are not opinions. They are
measurements, and the ground truth for both is produced by `implant_sites.py`
running a few dozen lines of geometry over a segmentation mask: find the crest,
find the roof of the canal, subtract, and probe the bucco-lingual run a couple
of millimetres below the crest.

So the honest question a reviewer asks about a 5.8M-parameter 3D transformer is
not "does it beat chance" -- the floors already answer that -- but **"does it
beat someone measuring the picture with a ruler?"** If a threshold-and-measure
estimator lands near the ViT, the architecture story is empty, and it is far
better to know that before twenty hours of results are written around it.

WHAT MAKES THIS A FAIR COMPARISON, AND WHAT WOULD MAKE IT A FRAUD
-----------------------------------------------------------------
The ground-truth labels are computed **from the segmentation mask**. An
estimator that also reads the mask would reproduce them almost exactly and would
prove nothing at all -- it would be measuring the same object with the same
ruler and reporting the agreement as a result.

This estimator therefore reads **only the cached CT intensities**, from
**exactly the patch the ViT is given** -- same `patch_centre`, same
`cut_patch`, same 96^3 box. It never opens a mask. Whatever it recovers, it
recovers from the image, which is the same thing being asked of the model.

WHAT IT IS NOT
--------------
It is not tuned, and it is not meant to be. Every constant below is either
inherited from `implant_sites.py` (the 3 mm cylinder, the 2 mm probe) or is the
first sensible value, chosen once and left. A baseline that has been tuned
against the test split is not a baseline. If it needs tuning to be competitive,
report that -- it is itself informative about how much of the task is geometry.

The intensity threshold is the one real free parameter, and it is Otsu rather
than a constant: the cache is z-scored per patient after a fixed HU window
(`build_site_cache.normalise`), so a fixed cut-off would drift with each
patient's foreground statistics while Otsu adapts to the patch it is given.
"""

from __future__ import annotations

import numpy as np

# Inherited from src/data/implant_sites.py so the two measure the same thing:
# a 3 mm-radius column about the site, probed 2 mm into the bone for width.
COLUMN_RADIUS_MM = 3.0
WIDTH_PROBE_MM = 2.0
# A run of dark voxels has to be at least this tall to count as the canal rather
# than as marrow texture or a noise speckle. The inferior alveolar canal is
# roughly 2-3 mm across, so this is deliberately under one canal diameter.
MIN_CANAL_RUN_MM = 1.2


def otsu(values: np.ndarray, bins: int = 128) -> float:
    """Threshold that best separates the intensity histogram into two classes.

    Standard between-class variance, `w0 * w1 * (m0 - m1)^2`, with one
    departure: when the maximum is a PLATEAU the middle of it is taken rather
    than the first index. Two well-separated peaks leave an empty gap between
    them, and every threshold inside that gap scores identically -- `argmax`
    then returns the left edge of the gap, which sits against the darker peak
    instead of between the two. On a phantom with peaks at -1 and +1 that put
    the threshold at -0.68.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0 or float(v.max() - v.min()) < 1e-9:
        return float("nan")
    counts, edges = np.histogram(v, bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    w = np.cumsum(counts)
    total = int(w[-1])
    if total == 0:
        return float("nan")
    cum = np.cumsum(counts * centres)
    w0 = w[:-1]
    w1 = total - w0
    ok = (w0 > 0) & (w1 > 0)
    if not ok.any():
        return float("nan")
    m0 = np.where(w0 > 0, cum[:-1] / np.maximum(w0, 1), 0.0)
    m1 = np.where(w1 > 0, (cum[-1] - cum[:-1]) / np.maximum(w1, 1), 0.0)
    between = np.where(ok, w0 * w1 * (m0 - m1) ** 2, -np.inf)
    best = float(between.max())
    plateau = np.flatnonzero(between >= best - 1e-9 * max(abs(best), 1.0))
    return float(centres[int(np.median(plateau))])


# Below this, a histogram is not two classes and thresholding it invents a
# boundary. MEASURED, not chosen: the separation ratio -- the gap between the
# two class means over the 0.5-99.5 percentile range -- is 0.92 on the bone
# phantom, 0.82 on a clean two-peak histogram, and 0.31 on pure Gaussian noise.
# 0.5 sits between them with room on both sides. It gates both passes: the
# second Otsu is only trusted when the sub-threshold voxels really are two
# tissues rather than one tissue and its noise.
MIN_SEPARATION = 0.5


# How much of the bone column's z-extent must read as solid before a threshold
# is believed. MEASURED across phantoms, with the disc that `measure_patch` uses:
#
#                        first cut          second cut
#   three-class ridge    0.03  <- shell     0.65  <- column      take the second
#   two-class ridge      0.65  <- column    0.89  <- swallowed   keep the first
#   pure noise           0.61               1.00                 rejected earlier
#   soft tissue only     0.52               1.00                 rejected earlier
#
# The last two never reach this decision: their first-pass separation is 0.31,
# below MIN_SEPARATION. So the only judgement left is shell versus column, and
# 0.03 against 0.65 is not a close call.
def _split(v: np.ndarray) -> tuple[float, float]:
    """Otsu threshold over `v`, and how separated the two classes it makes are.

    Separation is the gap between the two class means over the 0.5-99.5
    percentile range, which is what tells a genuinely two-class histogram from
    noise cut down the middle.
    """
    v = np.asarray(v, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size < 64:
        return float("nan"), 0.0
    cut = otsu(v)
    if not np.isfinite(cut):
        return float("nan"), 0.0
    dark, bright = v[v <= cut], v[v > cut]
    lo, hi = np.percentile(v, [0.5, 99.5])
    spread = float(hi - lo)
    if dark.size == 0 or bright.size == 0 or spread < 1e-9:
        return float("nan"), 0.0
    return cut, float(bright.mean() - dark.mean()) / spread


SHELL_SOLID_MAX = 0.20      # below this, the first cut found a shell, not a column
COLUMN_SOLID_MAX = 0.85     # above this, a cut has swallowed the soft tissue too


def _solid_fraction(patch: np.ndarray, cut: float, spacing) -> float:
    """Fraction of slices whose central column is at least half bone."""
    if not np.isfinite(cut):
        return 0.0
    disc = _column(patch, COLUMN_RADIUS_MM, spacing)
    n = max(int(disc.sum()), 1)
    profile = ((patch > cut) & disc[:, :, None]).sum(axis=(0, 1)) / n
    return float((profile >= 0.5).mean())


def bone_threshold(patch: np.ndarray, spacing=(0.3, 0.3, 0.3)) -> float:
    """Where bone starts. NaN when the patch has no two classes to separate.

    ONE OTSU CANNOT SEPARATE THREE TISSUE CLASSES, AND THIS PATCH HAS THREE. A
    site patch is mostly soft tissue, with a thin bright cortical shell around a
    ridge whose interior is trabecular bone barely above soft tissue. Otsu
    splits off the brightest class, which is cortex -- measured at 1.52 in
    z-scored units on the three-class phantom, leaving only 4.8% of voxels on
    the bone side. A run through the centre then reads cortex, marrow, cortex,
    and what gets measured is ONE CORTICAL PLATE: 0.9 mm where the ridge is
    9.0 mm, and 0.6 mm where the height is 9.0 mm.

    This is the same fault v2.0.0 fixed in the mask pipeline from the other
    side, where a bone-only slab had a hole at every occupied site and a 6.00 mm
    ridge measured 1.80 mm (`implant_sites.ridge_width`). Here it arrives
    through intensity instead of labels, and it was found in review rather than
    by this project's own tests.

    The remedy is to run Otsu AGAIN over everything below the first cut. That
    second histogram is soft tissue against trabecular bone, and its threshold
    is the one that puts the whole ridge on the bone side.

    THE SECOND CUT IS NOT ALWAYS RIGHT, so it is not chosen by the histogram. On
    a patch where the first cut already found a solid column, the second one
    sits inside the background and swallows it. The decision is made on the
    outcome instead: take the second threshold only when the first found a SHELL
    -- almost no solid column at all -- and only when the second finds a column
    rather than the whole patch. Both bounds are measured; see the table above.

    Filling the cortical ring per axial slice would also recover the interior,
    and it is what a mask-based pipeline would do. It is rejected here because
    it assumes topology this input does not guarantee: a 96^3 patch at 0.3 mm is
    28.8 mm across, a short segment of the arch rather than a closed ring, so
    whether the shell encloses anything in-plane depends on where round the jaw
    the site sits. A threshold assumes nothing about shape.
    """
    v = np.asarray(patch, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    cut, separation = _split(v)
    if not np.isfinite(cut) or separation < MIN_SEPARATION:
        return float("nan")

    if _solid_fraction(patch, cut, spacing) > SHELL_SOLID_MAX:
        return float(cut)                    # already a column; leave it alone

    inner_cut, _ = _split(v[v <= cut])
    if not np.isfinite(inner_cut):
        return float(cut)
    if _solid_fraction(patch, inner_cut, spacing) > COLUMN_SOLID_MAX:
        return float(cut)                    # would swallow the soft tissue
    return float(inner_cut)


def _column(patch: np.ndarray, radius_mm: float, spacing) -> np.ndarray:
    """Boolean (x, y) disc of `radius_mm` about the patch centre."""
    nx, ny = patch.shape[0], patch.shape[1]
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    xx = (np.arange(nx) - cx) * float(spacing[0])
    yy = (np.arange(ny) - cy) * float(spacing[1])
    return (xx[:, None] ** 2 + yy[None, :] ** 2) <= radius_mm ** 2


def _longest_run(flags: np.ndarray) -> tuple[int, int] | None:
    """Start and end (exclusive) of the longest True run, or None."""
    if not flags.any():
        return None
    idx = np.flatnonzero(
        np.diff(np.concatenate(([0], flags.view(np.int8), [0]))) != 0
    )
    starts, ends = idx[0::2], idx[1::2]
    k = int(np.argmax(ends - starts))
    return int(starts[k]), int(ends[k])


def measure_patch(patch: np.ndarray, jaw: str, spacing) -> dict:
    """Height and width in millimetres, from image intensity alone.

    `patch` is the model's own input: (x, y, z), z increasing upward, exactly
    what `cut_patch` returns. Returns NaN for a measurement it cannot make
    rather than a plausible number, because a baseline that always answers is
    worse than one that says when it could not.
    """
    patch = np.asarray(patch, dtype=np.float32)
    out = {"available_height_mm": float("nan"), "ridge_width_mm": float("nan"),
           "limiter": "no_bone", "crest_z": -1, "canal_z": -1}
    if patch.ndim != 3 or patch.size == 0:
        return out

    disc = _column(patch, COLUMN_RADIUS_MM, spacing)
    cut = bone_threshold(patch, spacing)
    if not np.isfinite(cut):
        return out

    bone = patch > cut
    # Fraction of the column that is bone, per axial slice.
    profile = (bone & disc[:, :, None]).sum(axis=(0, 1)) / max(int(disc.sum()), 1)
    solid = profile >= 0.5
    filled = np.flatnonzero(solid)
    if filled.size == 0:
        return out
    lo, hi = int(filled[0]), int(filled[-1])

    # The crest is the bone surface facing the missing tooth: the TOPMOST bone
    # slice in the mandible, the bottom-most in the maxilla, because the sinus
    # sits above the maxillary crest and the canal below the mandibular one.
    # This is the sign convention of implant_sites.measure_site.
    #
    # It is the extreme slice, NOT the end of the longest contiguous run. The
    # canal splits the bone column in two, and once it sits high enough the
    # block BELOW it is the longer one -- so a longest-run crest jumps to the
    # underside of the canal and then finds no canal beneath it. That reported
    # 12.3 mm for a phantom built with 6.0 mm, and it got worse as the true
    # height got smaller, which is the shape of a fault rather than of noise.
    upward = jaw == "lower"
    crest = hi if upward else lo
    out["crest_z"] = int(crest)

    # ---- height ---------------------------------------------------------
    # Walk from the crest into the bone looking for the limiting structure: a
    # dark run inside the column, which is the canal in the mandible and the
    # sinus floor in the maxilla. Both are air- or fluid-filled and read dark
    # against bone, which is the whole reason this is measurable from intensity.
    interior = np.arange(lo, crest)[::-1] if upward else np.arange(crest + 1, hi + 1)
    # With the threshold now sitting below trabecular bone, the canal is once
    # again what it physically is: a hole in an otherwise solid column.
    dark = ~solid
    min_run = max(1, int(round(MIN_CANAL_RUN_MM / float(spacing[2]))))

    limiter_z = None
    streak = 0
    for z in interior:
        if dark[z]:
            streak += 1
            if streak >= min_run:
                limiter_z = z + streak - 1 if upward else z - streak + 1
                break
        else:
            streak = 0

    if limiter_z is not None:
        out["limiter"] = "nerve" if upward else "sinus"
        out["canal_z"] = int(limiter_z)
        height = abs(crest - limiter_z) * float(spacing[2])
    else:
        # No dark structure inside the column: bone extent is the limit, which
        # is the same fallback measure_site takes.
        out["limiter"] = "bone_extent"
        height = (crest - lo if upward else hi - crest) * float(spacing[2])
    out["available_height_mm"] = float(height)

    # ---- width ----------------------------------------------------------
    # Probed a little into the bone, never at the crest itself: the top of a
    # ridge tapers to a knife edge and would report a width near zero for a site
    # that is perfectly implantable. Same reasoning, and the same 2 mm, as
    # implant_sites.ridge_width.
    step = int(round(WIDTH_PROBE_MM / float(spacing[2])))
    probe = crest - step if upward else crest + step
    probe = int(np.clip(probe, 0, patch.shape[2] - 1))
    slab = bone[:, :, probe]
    cx, cy = (patch.shape[0] - 1) // 2, (patch.shape[1] - 1) // 2
    runs = []
    for axis, pitch in ((0, float(spacing[0])), (1, float(spacing[1]))):
        line = slab[:, cy] if axis == 0 else slab[cx, :]
        seg = _longest_run(line)
        if seg is not None:
            runs.append((seg[1] - seg[0]) * pitch)
    if runs:
        # The SHORTER of the two in-plane runs: a ridge is long in the direction
        # it runs and thin across it, and which axis that is depends on where
        # round the arch the site sits.
        out["ridge_width_mm"] = float(min(runs))
    return out
