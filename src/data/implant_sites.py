"""Geometric measurements that decide whether an implant fits at a point in the jaw.

The clinical question this project answers is not "is an implant present?" but
"does this site need an implant, and is one anatomically possible?". The second
half is a measurement, and ToothFairy3's masks contain everything it needs:

    available height   crest of the alveolar ridge down to the nerve canal
                       (mandible) or up to the sinus floor (maxilla)
    ridge width        bucco-lingual thickness of bone below the crest

THRESHOLDS ARE NOT BAKED IN. Every function here returns millimetres. The yes/no
call happens in one place (`feasibility`), against values carried in the config,
so a clinician can revise a threshold without anyone reprocessing 28 GB of scans.

ORIENTATION IS MEASURED, NOT READ. ToothFairy3's affines report LPS, which says
the z index increases superiorly. The voxel data disagrees in every scan checked.
Call `orient_superior` on a mask before measuring it; after that the mandibular
crest really is the highest bone in a column with the nerve below it, and the
maxillary crest the lowest with the sinus above it.

THREE MISTAKES WORTH RECORDING, all found by measuring rather than by reading:

  * Taking the crest as the topmost jawbone voxel over the WHOLE mask lands on
    the condyle and coronoid process -- the jaw joint and a muscle attachment,
    tens of millimetres above the teeth. It put tooth centroids a median of
    14 mm "below the ridge". The crest is only meaningful PER COLUMN.
  * Doing that per column on the full array allocates several 68 MB boolean
    volumes per site and was killed by the OS. Every measurement here crops to
    the neighbourhood it actually needs first; the cylinder is ~20 voxels across
    against a 512x512x262 volume.
  * Trusting the affine inverted the whole superior-inferior axis, so bone
    height was measured up from the chin instead of down from the ridge. The
    mandible numbers still looked clinically plausible -- posterior sites came
    out short, anterior sites tall, exactly as real anatomy would -- and only
    the maxilla, where ridge width came out at 2.7 mm against a real 6-10 mm,
    exposed it. Plausible is not correct.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

# ToothFairy3 class indices. Read off its dataset.json, never guessed.
LOWER_JAW = 1
UPPER_JAW = 2
IAC = (3, 4)              # left, right inferior alveolar canal
INCISIVE_CANAL = (103, 104)   # its anterior continuation
SINUS = (5, 6)            # left, right maxillary sinus
BRIDGE, CROWN, IMPLANT = 8, 9, 10
RESTORATIONS = (BRIDGE, CROWN, IMPLANT)

UPPER_TEETH = (11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27)
LOWER_TEETH = (31, 32, 33, 34, 35, 36, 37, 41, 42, 43, 44, 45, 46, 47)
# Third molars are excluded by decision, not oversight: they are routinely absent
# and essentially never implanted, so counting them as "needs an implant" would
# fill the positive class with clinically meaningless sites.
THIRD_MOLARS = (18, 28, 38, 48)
TEETH = UPPER_TEETH + LOWER_TEETH

JAW_LABEL = {"lower": LOWER_JAW, "upper": UPPER_JAW}


def label_stats(mask: np.ndarray) -> dict:
    """{label: {"count": int, "centroid": (x, y, z)}} for every non-zero label.

    ONE pass over the volume. The obvious implementation -- a comparison per
    label -- costs 32 full passes over 68 million voxels just to find tooth
    centroids, and measured at 106 s per scan, which is 15.6 hours across
    ToothFairy3. Everything that needs whole-volume information is derived from
    this instead.
    """
    flat = np.flatnonzero(mask)
    if flat.size == 0:
        return {}

    values = mask.ravel()[flat].astype(np.int64)
    ny, nz = mask.shape[1], mask.shape[2]
    z = (flat % nz).astype(np.float64)
    y = ((flat // nz) % ny).astype(np.float64)
    x = (flat // (ny * nz)).astype(np.float64)

    size = int(values.max()) + 1
    count = np.bincount(values, minlength=size)
    sx = np.bincount(values, weights=x, minlength=size)
    sy = np.bincount(values, weights=y, minlength=size)
    sz = np.bincount(values, weights=z, minlength=size)

    out = {}
    for label in np.flatnonzero(count):
        n = int(count[label])
        out[int(label)] = {
            "count": n,
            "centroid": np.array([sx[label] / n, sy[label] / n, sz[label] / n]),
        }
    return out


def superior_sign(mask: np.ndarray, stats: dict | None = None) -> int:
    """+1 if the z index increases toward the head, -1 if it increases downward.

    DO NOT TRUST THE HEADER FOR THIS. ToothFairy3's affines report LPS, which
    says z increases superiorly, but the voxel data says the opposite in every
    scan checked: sinus sits at z~10, maxilla at z~25, upper teeth at z~55,
    lower teeth and the mandibular canal at z~150. Believing the header put the
    alveolar crest on the inferior border of the mandible and measured bone
    height upward from the chin. The mandible numbers still looked plausible,
    which is exactly why this is derived from anatomy and asserted instead.

    WHICH PAIRS ARE USED WAS MEASURED, NOT REASONED. Each candidate below was
    checked on 40 scans against the answer the two jawbones already give, and
    only the ones that agreed every single time are kept:

        UpperJaw / LowerJaw            39/39   100%
        UpperJaw / IAC                 39/39   100%
        upperTeeth / lowerTeeth        38/38   100%
        Sinus / IAC                    29/29   100%
        Sinus / upperTeeth             28/28   100%
        lowerTeeth / incisive canal    37/37   100%
        LowerJaw / incisive canal      37/37   100%

    Two plausible-sounding cues were REJECTED by that check and must not be
    added back: lowerTeeth over IAC managed only 82%, and LowerJaw over IAC
    46% -- no better than guessing. Tooth roots do sit above the nerve, but the
    canal climbs toward the mandibular foramen posteriorly and its centroid can
    end up above theirs. The incisive canal is reliable precisely because it is
    anterior and stays below the roots.

    The last two matter more than their position in the list suggests: 61 of the
    532 scans are mandible-only, with no maxilla and no sinus, and every
    upper-versus-lower cue is unusable on them. Without the incisive canal they
    would have been dropped from the dataset entirely.

    Pairs disagreeing raises rather than taking a majority.
    """
    if stats is None:
        stats = label_stats(mask)

    def centre_z(labels):
        picked = [stats[v] for v in labels if v in stats]
        if not picked:
            return None
        total = sum(p["count"] for p in picked)
        return float(sum(p["centroid"][2] * p["count"] for p in picked) / total)

    votes = []
    for upper, lower in (((UPPER_JAW,), (LOWER_JAW,)),
                         ((UPPER_JAW,), IAC),
                         (UPPER_TEETH, LOWER_TEETH),
                         (SINUS, IAC),
                         (SINUS, UPPER_TEETH),
                         (LOWER_TEETH, INCISIVE_CANAL),
                         ((LOWER_JAW,), INCISIVE_CANAL)):
        up, lo = centre_z(upper), centre_z(lower)
        if up is None or lo is None or up == lo:
            continue
        votes.append(1 if up > lo else -1)

    if not votes:
        raise ValueError("cannot determine orientation: no upper/lower structure pair present")
    if len(set(votes)) > 1:
        raise ValueError(f"orientation is ambiguous, structure pairs disagree: {votes}")
    return votes[0]


def orient_superior(mask: np.ndarray, stats: dict | None = None):
    """Return a view of `mask` in which the z index increases toward the head.

    Flipping once per scan keeps every measurement below able to say "the crest
    is the highest bone in this column" and mean it. The flip is a view, so it
    costs nothing.
    """
    sign = superior_sign(mask, stats)
    return (mask if sign == 1 else mask[:, :, ::-1]), sign


def flip_stats(stats: dict, n_z: int) -> dict:
    """Re-express centroids for a mask flipped along z. Counts are unchanged."""
    out = {}
    for label, entry in stats.items():
        c = entry["centroid"].copy()
        c[2] = (n_z - 1) - c[2]
        out[label] = {"count": entry["count"], "centroid": c}
    return out


def prepare_mask(mask: np.ndarray):
    """(oriented mask, orientation sign, label stats) from a single pass.

    Orientation, tooth centroids and the arch fit all need whole-volume
    information. Computing them separately meant several passes each; doing it
    once here is what took the label build from 106 s per scan to a few.
    """
    stats = label_stats(mask)
    sign = superior_sign(mask, stats)
    if sign == 1:
        return mask, sign, stats
    return mask[:, :, ::-1], sign, flip_stats(stats, mask.shape[2])


@dataclass
class SiteMeasurement:
    """Millimetres at one candidate implant position. No thresholds applied."""

    jaw: str                       # "upper" | "lower"
    crest_mm: float                # superior-inferior position of the ridge crest
    available_height_mm: float     # crest to nerve (lower) or crest to sinus (upper)
    ridge_width_mm: float          # bucco-lingual bone thickness below the crest
    limiting_structure: str        # "nerve" | "sinus" | "bone_extent" | "no_bone"
    n_bone_voxels: int             # support for the measurement; 0 means no bone

    def as_row(self) -> dict:
        return asdict(self)


def check_spacing(spacing) -> np.ndarray:
    spacing = np.asarray(spacing, dtype=float)
    if spacing.shape != (3,) or not np.all(spacing > 0):
        raise ValueError(f"spacing must be three positive values, got {spacing!r}")
    return spacing


def crop_around(mask: np.ndarray, centre, radius_mm: float, spacing):
    """Sub-volume within `radius_mm` of `centre` in x and y, full depth in z.

    Returns (sub, (cx, cy)) with the centre re-expressed in the sub-volume's
    coordinates. Cropping first is what keeps this from allocating whole-volume
    boolean arrays for a measurement that spans a few millimetres.
    """
    spacing = check_spacing(spacing)
    rx = int(np.ceil(radius_mm / spacing[0])) + 1
    ry = int(np.ceil(radius_mm / spacing[1])) + 1
    cx, cy = int(round(float(centre[0]))), int(round(float(centre[1])))

    x0, x1 = max(0, cx - rx), min(mask.shape[0], cx + rx + 1)
    y0, y1 = max(0, cy - ry), min(mask.shape[1], cy + ry + 1)
    if x0 >= x1 or y0 >= y1:
        return mask[:0, :0, :], (0, 0)
    return mask[x0:x1, y0:y1, :], (cx - x0, cy - y0)


def cylinder(shape2d, centre, radius_mm: float, spacing) -> np.ndarray:
    """Disc of radius `radius_mm` around `centre`, as an (x, y) boolean mask.

    Measurements are taken inside a cylinder rather than a single voxel column so
    one stray annotation voxel cannot set the crest height. The radius is a real
    distance, so it means the same thing at any scan spacing.
    """
    spacing = check_spacing(spacing)
    xs = (np.arange(shape2d[0]) - float(centre[0])) * spacing[0]
    ys = (np.arange(shape2d[1]) - float(centre[1])) * spacing[1]
    return xs[:, None] ** 2 + ys[None, :] ** 2 <= radius_mm ** 2


def z_extent(sub: np.ndarray, labels, disc: np.ndarray):
    """(lowest, highest) z index where any of `labels` appears inside `disc`.

    Returns (None, None) when the structure is absent, which is a normal outcome
    -- 28% of scans carry no sinus annotation -- and must be handled by the
    caller rather than silently becoming a zero.
    """
    hit = np.isin(sub, labels) & disc[:, :, None]
    if not hit.any():
        return None, None
    z = np.where(hit.any(axis=(0, 1)))[0]
    return int(z[0]), int(z[-1])


def crest_index(sub: np.ndarray, jaw: str, disc: np.ndarray):
    """Median height of the alveolar crest across the columns in the cylinder.

    NOT the extreme. Taking the single highest bone voxel in the cylinder lets
    one spike at its edge set the crest for the whole site: the width probe then
    samples 2 mm below a level the centre never reaches, finds no bone there,
    falls back to the nearest run, and reports a 2.7 mm ridge at a site holding
    23,250 voxels of bone. The median is robust to both a spike and a hole.

    Returns (crest_index, n_columns_with_bone), or (None, 0) if there is no bone.
    """
    bone = np.isin(sub, (JAW_LABEL[jaw],)) & disc[:, :, None]
    has = bone.any(axis=2)
    if not has.any():
        return None, 0

    if jaw == "lower":                      # crest is the top of the column
        tops = bone.shape[2] - 1 - np.argmax(bone[:, :, ::-1], axis=2)
    else:                                   # maxillary process hangs down
        tops = np.argmax(bone, axis=2)
    return int(np.median(tops[has])), int(has.sum())


def measure_site(
    mask: np.ndarray,
    centre,
    jaw: str,
    spacing,
    radius_mm: float = 3.0,
    width_probe_mm: float = 2.0,
    width_search_mm: float = 20.0,
) -> SiteMeasurement:
    """Measure the bone available for an implant at one point.

    `centre` is a voxel coordinate; only its x and y are used, because the crest
    is found by looking down the column rather than trusting the z it was given.
    """
    spacing = check_spacing(spacing)
    if jaw not in JAW_LABEL:
        raise ValueError(f"jaw must be 'upper' or 'lower', got {jaw!r}")

    sub, c = crop_around(mask, centre, radius_mm, spacing)
    if sub.size == 0:
        return SiteMeasurement(jaw, float("nan"), float("nan"), float("nan"), "no_bone", 0)

    disc = cylinder(sub.shape[:2], c, radius_mm, spacing)
    lo, hi = z_extent(sub, (JAW_LABEL[jaw],), disc)
    if lo is None:
        return SiteMeasurement(jaw, float("nan"), float("nan"), float("nan"), "no_bone", 0)

    n_bone = int((np.isin(sub, (JAW_LABEL[jaw],)) & disc[:, :, None]).sum())
    crest, _ = crest_index(sub, jaw, disc)
    if crest is None:
        return SiteMeasurement(jaw, float("nan"), float("nan"), float("nan"), "no_bone", 0)

    if jaw == "lower":
        limiter, height = "bone_extent", (crest - lo) * spacing[2]
        _, canal_top = z_extent(sub, IAC + INCISIVE_CANAL, disc)
        if canal_top is not None:
            # The nerve sits below the crest; usable bone stops at its roof.
            limiter, height = "nerve", (crest - canal_top) * spacing[2]
    else:
        limiter, height = "bone_extent", (hi - crest) * spacing[2]
        sinus_floor, _ = z_extent(sub, SINUS, disc)
        if sinus_floor is not None:
            # The sinus sits above the crest; usable bone stops at its floor.
            limiter, height = "sinus", (sinus_floor - crest) * spacing[2]

    width = ridge_width(mask, centre, crest, jaw, spacing,
                        probe_mm=width_probe_mm, search_mm=width_search_mm)
    return SiteMeasurement(jaw, float(crest * spacing[2]), float(max(height, 0.0)),
                           float(width), limiter, n_bone)


def ridge_width(mask, centre, crest_z: int, jaw: str, spacing,
                probe_mm: float = 2.0, search_mm: float = 20.0) -> float:
    """Bucco-lingual bone thickness in a slice a little below (or above) the crest.

    Probed a couple of millimetres into the bone rather than exactly at the crest,
    because the very top of a ridge tapers to a knife edge and would report a
    width near zero for a site that is perfectly implantable.

    The thickness reported is the SHORTER of the two in-plane runs through the
    point: the ridge is long in the direction it runs and thin across it, and
    which axis that is depends on where round the arch the site sits.
    """
    spacing = check_spacing(spacing)
    step = max(1, int(round(probe_mm / spacing[2])))
    z = crest_z - step if jaw == "lower" else crest_z + step
    if not (0 <= z < mask.shape[2]):
        return float("nan")

    sub, c = crop_around(mask[:, :, z:z + 1], centre, search_mm, spacing)
    if sub.size == 0:
        return float("nan")
    slab = sub[:, :, 0] == JAW_LABEL[jaw]
    if not slab.any():
        return 0.0

    cx = int(np.clip(c[0], 0, slab.shape[0] - 1))
    cy = int(np.clip(c[1], 0, slab.shape[1] - 1))
    widths = []
    for line, index, mm in ((slab[:, cy], cx, spacing[0]),
                            (slab[cx, :], cy, spacing[1])):
        if line.any():
            widths.append(float(run_through(line, index) * mm))
    return min(widths) if widths else 0.0


def run_through(line: np.ndarray, index: int) -> int:
    """Length of the contiguous True run containing `index`.

    If `index` is not inside a run, the nearest run is measured instead: an arch
    position estimated from surrounding anatomy can sit a voxel or two off the
    bone without the site being unmeasurable. Returns 0 only when the line is
    entirely empty.
    """
    n = len(line)
    if n == 0 or not line.any():
        return 0
    index = int(np.clip(index, 0, n - 1))
    if not line[index]:
        hits = np.where(line)[0]
        index = int(hits[np.argmin(np.abs(hits - index))])

    lo = index
    while lo > 0 and line[lo - 1]:
        lo -= 1
    hi = index
    while hi < n - 1 and line[hi + 1]:
        hi += 1
    return hi - lo + 1


def tooth_centroids(mask: np.ndarray, stats: dict | None = None) -> dict:
    """{FDI number: (x, y, z) centroid} for every tooth annotated in this scan.

    Computed once per scan and passed into `site_is_occupied`, because scanning
    the volume per candidate site would dominate the whole label build.
    """
    if stats is None:
        stats = label_stats(mask)
    return {t: stats[t]["centroid"] for t in TEETH + THIRD_MOLARS if t in stats}


def site_is_occupied(mask, centre, spacing, centroids=None,
                     tooth_reach_mm: float = 4.0,
                     restoration_radius_mm: float = 2.0,
                     min_restoration_voxels: int = 200) -> dict:
    """Is there already a tooth or a restoration at this position?

    A site holding a tooth, crown, bridge pontic or existing implant does not
    need a new implant. Crowns sit on natural roots and bridges span a gap, but
    either way the position is restored and the clinical answer is the same --
    which keeps the label objective rather than turning on whether a dentist
    would have preferred an implant to the bridge already there.

    TEETH ARE MATCHED BY CENTROID, NOT BY CONTACT. In a single-tooth gap the
    neighbours sit only ~3.5 mm away, so any cylinder wide enough to see the site
    also clips them. Counting voxels called every gap occupied and found ONE
    empty site in 25 scans. A tooth centred here has its centroid within a
    few millimetres; a neighbour's is a whole tooth-width away, so centroid
    distance separates them cleanly and voxel contact does not.

    Restorations keep the volume test, since a bridge pontic has no meaningful
    centroid of its own -- the mask spans several sites at once -- but it does
    physically fill the gap it is restoring.
    """
    empty = {"occupied": False, "by": "", "tooth_id": 0, "occupancy_mm": float("nan")}
    spacing = check_spacing(spacing)

    if centroids is None:
        centroids = tooth_centroids(mask)

    # 1. A tooth whose centre of mass is at this site.
    best_id, best_d = 0, float("inf")
    for tid, c in centroids.items():
        d = float(np.hypot((c[0] - float(centre[0])) * spacing[0],
                           (c[1] - float(centre[1])) * spacing[1]))
        if d < best_d:
            best_id, best_d = tid, d
    if best_d <= tooth_reach_mm:
        return {"occupied": True, "by": "tooth", "tooth_id": int(best_id),
                "occupancy_mm": best_d}

    # 2. A restoration physically filling the gap.
    sub, c = crop_around(mask, centre, restoration_radius_mm, spacing)
    if sub.size:
        disc = cylinder(sub.shape[:2], c, restoration_radius_mm, spacing)
        inside = sub[disc]
        for label, name in ((IMPLANT, "implant"), (BRIDGE, "bridge"), (CROWN, "crown")):
            n = int((inside == label).sum())
            if n >= min_restoration_voxels:
                return {"occupied": True, "by": name, "tooth_id": 0,
                        "occupancy_mm": float(n) * float(np.prod(spacing))}
    return empty


def feasibility(m: SiteMeasurement, min_height_mandible_mm: float,
                min_height_maxilla_mm: float, min_width_mm: float) -> dict:
    """Turn millimetres into the yes/no call, in exactly one place.

    Separating this from `measure_site` is what lets a clinician revise a
    threshold without any scan being read again: the builder stores the
    millimetres, and this is re-run over the CSV.
    """
    need_h = min_height_mandible_mm if m.jaw == "lower" else min_height_maxilla_mm
    measurable = bool(np.isfinite(m.available_height_mm) and np.isfinite(m.ridge_width_mm))
    h_ok = bool(measurable and m.available_height_mm >= need_h)
    w_ok = bool(measurable and m.ridge_width_mm >= min_width_mm)

    if not measurable:
        reason = "unmeasurable"
    elif h_ok and w_ok:
        reason = "ok"
    elif not h_ok and not w_ok:
        reason = "height_and_width"
    elif not h_ok:
        reason = f"height_{m.limiting_structure}"
    else:
        reason = "width"

    return {
        "feasible": bool(h_ok and w_ok),
        "height_ok": h_ok,
        "width_ok": w_ok,
        "required_height_mm": float(need_h),
        "required_width_mm": float(min_width_mm),
        "reason": reason,
    }
