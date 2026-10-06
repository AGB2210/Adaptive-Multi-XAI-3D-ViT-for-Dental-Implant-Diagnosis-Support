"""One scan -> the volume the site model reads: oriented, clipped, z-scored.

ONE IMPLEMENTATION, TWO CALLERS. `scripts/build_site_cache.py` writes the
training cache with this, and the inference app prepares an uploaded scan with
it. They used to be one function inside the cache script; a second copy in the
app would agree with it until the day it did not, and every prediction would
then be made on intensities the model was never trained on while still looking
like a jaw. That is the same shape as the anatomy masks that sat 7.2 mm from
the model's input for three releases.

What it does, and it has to match `src/data/implant_sites.py` exactly:

  * NO canonical reorientation -- the voxel frame stays the label's own
  * the z flip the MASK's anatomy asked for (`superior_sign`), applied here to
    the image, so site_x / site_y / site_z index both the same way
  * a fixed HU window, then a z-score over foreground voxels
"""

from __future__ import annotations

import numpy as np


def normalise(volume: np.ndarray, clip_window, air_threshold: float):
    """Fixed-window clip then z-score on foreground. Same contract as before.

    The window is fixed rather than a per-patient percentile for the reason
    recorded in src/data/preprocess.py: implants are hyperdense and a 99.5th
    percentile clip flattens them onto cortical bone, erasing the very thing the
    model has to see.
    """
    lo, hi = float(clip_window[0]), float(clip_window[1])
    volume = np.clip(volume, lo, hi)

    foreground = volume[volume > air_threshold]
    # Falling back to the whole volume changes what the z-score MEANS for that
    # scan: it is no longer normalised against tissue but against tissue plus
    # air, so its intensities are not comparable with the rest of the cohort.
    # `src/data/preprocess.py` does the same fallback and writes
    # "z-scored on all voxels (foreground empty)" into its manifest. This path
    # did it silently, and this is the live one.
    fell_back = foreground.size < 100
    if fell_back:
        foreground = volume
    mean, std = float(foreground.mean()), float(foreground.std())
    if std < 1e-6:
        raise ValueError("zero-variance volume")
    return ((volume - mean) / std).astype(np.float16), mean, std, fell_back


def prepare_volume(volume: np.ndarray, sign: int, clip_window, air_threshold: float):
    """Raw image array -> (float16 model-frame volume, fg mean, fg std, fell_back).

    `sign` is the superior sign of the scan: -1 means the z index increases
    downward and the volume is flipped so that it increases toward the head, as
    every site coordinate assumes. It is decided by anatomy, never by the
    affine -- see `implant_sites.superior_sign`.

    `volume` may be modified in place: non-finite voxels are set to air first.
    """
    if sign not in (1, -1):
        raise ValueError(f"orientation sign must be +1 or -1, got {sign!r}")
    if volume.ndim != 3:
        raise ValueError(f"expected a 3D volume, got shape {volume.shape}")
    air = float(air_threshold)
    np.nan_to_num(volume, copy=False, nan=air, posinf=air, neginf=air)
    if sign == -1:
        volume = volume[:, :, ::-1]
    return normalise(volume, clip_window, air)
