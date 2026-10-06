"""Where each tooth site is, in the oriented volume, and what the mask measured there.

Two sources, one shape of answer:

  mask       `site_scoring.score_one` -- the label builder's own function -- so a
             site is located by the same arch fit and measured by the same
             geometry the training labels were. The measurements come back as
             ground truth beside the prediction.
  localiser  a trained network predicting the positions from the image alone,
             for a scan that arrives without a segmentation. See
             `src/models/localiser.py`. It measures nothing, so its sites carry
             no ground truth.

A site is a dict, not a class, because it is serialised to the browser as-is.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.data.dental_arch import ARCH
from src.data.site_scoring import score_one

# The columns of a `score_one` row that are measurements of the scan, reported
# as ground truth. Everything else is bookkeeping.
TRUTH_FIELDS = ("needs_implant", "feasible", "available_height_mm", "ridge_width_mm",
                "limiting_structure", "reason", "occupied_by")


def _plain(value):
    """pandas NA / numpy scalars -> JSON-safe Python values."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def has_position(site: dict) -> bool:
    return all(site.get(k) is not None and math.isfinite(site[k])
               for k in ("site_x", "site_y", "site_z"))


def sites_from_mask(mask: np.ndarray, spacing, rules: dict, jaws=("lower",)) -> list[dict]:
    """Every site of the requested jaws, located and measured from the mask.

    `mask` is in the scan's RAW frame; `score_one` orients it itself and returns
    coordinates in the oriented frame, which is the frame the prepared volume is
    in. `rules` only affects the measured `feasible` verdict reported as truth --
    the model's own verdict is re-derived from its millimetres at any threshold.
    """
    rows = score_one(mask, spacing, rules)
    out = []
    for row in rows:
        if row["jaw"] not in jaws:
            continue
        site = {
            "tooth": int(row["tooth"]),
            "jaw": row["jaw"],
            "site_x": _plain(row["site_x"]),
            "site_y": _plain(row["site_y"]),
            "site_z": _plain(row.get("site_z")),
            "source": "mask",
            "method": row["site_method"],
            "anchors": _plain(row.get("site_anchors")),
            "truth": {k: _plain(row.get(k)) for k in TRUTH_FIELDS},
        }
        # The model was trained on the `teeth` tier alone -- positions fitted to
        # three or more real teeth. A weaker tier is a position the model never
        # saw the like of, and the result has to say so.
        if site["method"] != "teeth":
            site["note"] = (f"position from the '{site['method']}' tier "
                            f"({site['anchors']} anchor teeth); the model was trained "
                            f"on the 'teeth' tier only")
        if not has_position(site):
            site["note"] = "no position: the arch fit left the field of view or found no bone"
        out.append(site)
    order = {t: i for i, t in enumerate(ARCH["lower"] + ARCH["upper"])}
    return sorted(out, key=lambda s: order.get(s["tooth"], 99))
