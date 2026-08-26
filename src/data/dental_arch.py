"""Where is tooth position 36, when tooth 36 is not there?

Measuring an implant site needs a location, and a missing tooth leaves none. The
obvious fix -- interpolate between the neighbours -- covers far less than it
looks. Measured across 60 ToothFairy3 scans, of every missing site:

    21.5%   has a present tooth on both sides           (interpolation works)
    45.0%   has one on a single side                    (needs extrapolation)
    33.5%   sits in a quadrant with no teeth at all     (needs the other jaw)

and 30.7% of missing sites are in a jaw that is entirely edentulous. Those are
precisely the patients who most need implants, so an approach that quietly drops
them would bias the dataset against its own clinical purpose.

So position is resolved by a chain, and every row records WHICH step produced
it. That way coverage is a reported number rather than an assumption, and a
reviewer can exclude the weaker tiers without rebuilding anything:

    teeth          >=3 teeth in this jaw -> fit the arch, read off the position
    sparse          1-2 teeth            -> same fit, heavily extrapolated
    opposite_jaw    none                 -> borrow the other jaw's arch
    failed          neither jaw has any  -> no position, recorded as such

THE ARCH MODEL. Along the arch, FDI numbering is a strict order -- 17,16,...,11,
21,...,27 sweeps right to left. So a tooth's index in that sequence maps
monotonically onto its position around the curve, and present teeth pin that
mapping down. Fitting index -> (x, y) with a low-order polynomial keeps missing
positions on the curve the remaining teeth describe, instead of on an average
jaw that belongs to nobody.
"""

from __future__ import annotations

import numpy as np

from src.data.implant_sites import LOWER_TEETH, UPPER_TEETH, tooth_centroids

# Anatomical order around the arch, one end to the other. Not numeric order:
# 17..11 runs right-to-midline, then 21..27 continues midline-to-left.
UPPER_ARCH = (17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27)
LOWER_ARCH = (47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37)
ARCH = {"upper": UPPER_ARCH, "lower": LOWER_ARCH}

JAW_OF = {t: ("upper" if t in UPPER_TEETH else "lower") for t in UPPER_TEETH + LOWER_TEETH}


def arch_index(tooth: int) -> int:
    """Position of a tooth in the sweep around its own arch, 0..13."""
    jaw = JAW_OF.get(tooth)
    if jaw is None:
        raise ValueError(f"{tooth} is not one of the 28 sites this project scores")
    return ARCH[jaw].index(tooth)


def _fit_axis(idx: np.ndarray, values: np.ndarray, degree: int):
    """Least-squares polynomial, degree reduced to what the data can support."""
    degree = int(min(degree, len(idx) - 1))
    if degree < 1:
        # One anchor supports no slope at all; the best estimate is that value.
        return np.poly1d([float(values[0])])
    return np.poly1d(np.polyfit(idx, values, degree))


def fit_arch(centroids: dict, jaw: str, degree: int = 2):
    """Map arch index -> (x, y), fitted to whichever teeth this jaw still has.

    Returns None when the jaw has no teeth. A quadratic is the default because
    a dental arch is close to parabolic; with fewer than three anchors the
    degree drops automatically rather than over-fitting a straight line into
    nonsense.
    """
    pts = [(arch_index(t), c) for t, c in centroids.items() if JAW_OF.get(t) == jaw]
    if not pts:
        return None

    idx = np.array([p[0] for p in pts], dtype=float)
    xs = np.array([p[1][0] for p in pts], dtype=float)
    ys = np.array([p[1][1] for p in pts], dtype=float)
    fx, fy = _fit_axis(idx, xs, degree), _fit_axis(idx, ys, degree)
    return lambda i: (float(fx(i)), float(fy(i))), len(pts)


def site_positions(mask: np.ndarray, centroids: dict | None = None,
                   degree: int = 2) -> dict:
    """{tooth: {"xy": (x, y), "method": str, "anchors": int}} for all 28 sites.

    Pass `centroids` when the caller already has them: recomputing costs a
    whole extra pass over the volume for no new information.
    """
    if centroids is None:
        centroids = tooth_centroids(mask)
    fits = {}
    for jaw in ("upper", "lower"):
        got = fit_arch(centroids, jaw, degree)
        fits[jaw] = got if got else (None, 0)

    out = {}
    for jaw in ("upper", "lower"):
        curve, n = fits[jaw]
        other_curve, other_n = fits["lower" if jaw == "upper" else "upper"]

        for tooth in ARCH[jaw]:
            i = arch_index(tooth)
            if curve is not None and n >= 3:
                method, xy, anchors = "teeth", curve(i), n
            elif curve is not None:
                method, xy, anchors = "sparse", curve(i), n
            elif other_curve is not None:
                # The two arches are near-concentric in (x, y): the maxillary one
                # is slightly the larger, but for locating a column to measure,
                # borrowing the opposite jaw beats having no position at all.
                method, xy, anchors = "opposite_jaw", other_curve(i), other_n
            else:
                out[tooth] = {"xy": None, "method": "failed", "anchors": 0}
                continue
            out[tooth] = {"xy": xy, "method": method, "anchors": int(anchors)}
    return out


def within_volume(xy, shape, margin: int = 2) -> bool:
    """Is an estimated position actually inside the scan?

    Extrapolating an arch past the teeth that defined it can land outside the
    field of view entirely. Such a site is unmeasurable and must be recorded as
    that, not measured against whatever happens to sit at the clipped edge.
    """
    if xy is None:
        return False
    x, y = xy
    return (margin <= x < shape[0] - margin) and (margin <= y < shape[1] - margin)
