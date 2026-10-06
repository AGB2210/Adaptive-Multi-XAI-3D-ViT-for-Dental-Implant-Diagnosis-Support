"""Every tooth site of one scan, located and measured from its segmentation mask.

This is the body of `scripts/build_implant_labels.py`, lifted into `src/` so the
inference app can run it on an uploaded scan that comes with a mask. The label
build and the app then locate a site by the same arch fit and measure it by the
same geometry; a second copy would be free to disagree about where tooth 36 is,
and a patch cut 3 mm from the training position still looks like a jaw.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.dental_arch import ARCH, site_positions, within_volume
from src.data.implant_sites import (
    feasibility,
    measure_site,
    prepare_mask,
    site_is_occupied,
    tooth_centroids,
)


def score_one(mask, spacing, rules: dict) -> list[dict]:
    """Every site in one scan."""
    oriented, sign, stats = prepare_mask(mask)
    centroids = tooth_centroids(oriented, stats)
    positions = site_positions(oriented, centroids)

    rows = []
    for jaw in ("upper", "lower"):
        for tooth in ARCH[jaw]:
            info = positions[tooth]
            xy = info["xy"]
            row = {
                "tooth": tooth,
                "jaw": jaw,
                "site_method": info["method"],
                "site_anchors": info["anchors"],
                "orientation_sign": sign,
                # Voxel coordinates in the ORIENTED volume, so the training
                # patch can be cut without re-reading the mask. Without these
                # every epoch would have to redo the arch fit.
                "site_x": float(xy[0]) if xy else np.nan,
                "site_y": float(xy[1]) if xy else np.nan,
                "spacing_z_mm": float(spacing[2]),
            }

            if not within_volume(info["xy"], oriented.shape):
                # Extrapolating past the teeth that defined the arch can land
                # outside the field of view. That is not a site with no bone; it
                # is a site we never saw.
                row.update({"needs_implant": pd.NA, "feasible": pd.NA,
                            "occupied_by": "", "reason": "outside_volume",
                            "available_height_mm": np.nan, "ridge_width_mm": np.nan,
                            "limiting_structure": "", "crest_mm": np.nan,
                            "n_bone_voxels": 0, "height_ok": pd.NA, "width_ok": pd.NA,
                            "required_height_mm": np.nan, "required_width_mm": np.nan,
                            "site_z": np.nan, "occupying_tooth": 0})
                rows.append(row)
                continue

            centre = (info["xy"][0], info["xy"][1], 0)
            occ = site_is_occupied(oriented, centre, spacing, centroids, tooth=tooth)
            m = measure_site(oriented, centre, jaw, spacing)
            verdict = feasibility(m, **rules)

            row.update(m.as_row())
            row.update(verdict)
            # The patch is cut around the ridge, not around whatever z the arch
            # fit happened to carry, so the box straddles crest and canal.
            row["site_z"] = (float(m.crest_mm) / float(spacing[2])
                             if np.isfinite(m.crest_mm) else np.nan)
            row["needs_implant"] = int(not occ["occupied"])
            row["occupied_by"] = occ["by"]
            row["occupying_tooth"] = occ["tooth_id"]
            # Feasibility is measured everywhere, but it only MEANS anything
            # where an implant is actually wanted. Keeping the measurement on
            # occupied rows costs nothing and lets the rules be re-scored later.
            rows.append(row)
    return rows
