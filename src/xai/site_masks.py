"""Ground-truth anatomy for one implant site, cut to the same patch as the input.

This is what turns the localisation metric from a formality into a real test.

On the previous task the question was "does the explanation point at the
implant?". A dental implant is titanium: the brightest, sharpest, most obvious
object in a CBCT. Any method that responds to edges passes that trivially, which
is exactly how Integrated Gradients scored 86x chance on it while failing the
model-randomisation check outright.

The question here is different:

    the model says this site is NOT feasible
    -> does the explanation point at the INFERIOR ALVEOLAR CANAL?

The canal is a dark, low-contrast tube running inside bone. Nothing about it
stands out to an edge detector, so a method can only find it by having learned
what the model learned. A high enrichment on the canal is therefore evidence in
a way that a high enrichment on metal never was.

The competing structure is the surrounding jawbone. If an explanation for
"not feasible" spreads over bone generally rather than concentrating on the
canal, it is describing where the jaw is, not why the site fails.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from src.data.implant_sites import IAC, INCISIVE_CANAL, LOWER_JAW, UPPER_JAW, superior_sign
from src.data.site_dataset import cut_patch, patch_centre

# The structures worth scoring an explanation against, and what each one means
# if the saliency lands on it.
SITE_STRUCTURES = {
    # The thing a "not feasible" verdict in the mandible is usually about.
    "nerve": IAC + INCISIVE_CANAL,
    # Bone in general -- the competitor. Landing here is not wrong, but it is
    # not specific either.
    "jawbone": (LOWER_JAW, UPPER_JAW),
}


def patch_masks(
    mask_path: str | Path,
    site_row,
    patch_size: int,
    structures: dict | None = None,
) -> dict:
    """Binary masks for each structure, cropped exactly like the model's input.

    The crop must match `CaseSet.load` voxel for voxel, and there are TWO things
    to match, not one. The z flip is the obvious half. The other is the
    quarter-shift `patch_centre` applies to push the box toward the structure
    that decides the answer -- 24 voxels at `patch_size=96`, which is 7.2 mm at
    0.3 mm spacing.

    This function honoured the flip and built its own centre from raw `site_z`,
    so for three releases every mask was cut 7.2 mm from the box the model was
    actually shown, and every number in `results_localization.csv` scored a
    saliency map against anatomy that was not where the model was looking. The
    centre now comes from `patch_centre` itself. Do not reconstruct it here: a
    second implementation of the crop is exactly what went wrong.
    """
    structures = structures or SITE_STRUCTURES
    mask = np.asarray(nib.load(str(mask_path)).dataobj)

    # Same orientation the site coordinates were measured in. Derived from the
    # anatomy, never from the affine -- see src/data/implant_sites.py.
    if superior_sign(mask) == -1:
        mask = mask[:, :, ::-1]

    # The model's own centring function, not a copy of it.
    centre = patch_centre(site_row, mask.shape, int(patch_size))

    out = {}
    for name, labels in structures.items():
        binary = np.isin(mask, labels).astype(np.uint8)
        out[name] = cut_patch(binary, centre, int(patch_size)).astype(bool)
    return out


def describe_coverage(masks: dict) -> dict:
    """What fraction of the patch each structure occupies.

    Reported alongside every enrichment number, because enrichment is a ratio
    against exactly this: a structure filling 30% of the patch cannot reach the
    enrichment a structure filling 0.5% can, and comparing the two without the
    denominator is meaningless.
    """
    return {name: float(m.mean()) for name, m in masks.items()}
