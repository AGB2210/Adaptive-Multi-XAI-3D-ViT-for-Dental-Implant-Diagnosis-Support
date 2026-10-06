"""An uploaded CBCT (and optional mask) -> the volume the site model reads.

The transform is `src.data.scan.prepare_volume`, the same function that built
the training cache, so an uploaded scan arrives at the model exactly as a
cached one did. Two things the cache build could take for granted have to be
checked here instead, because nobody has looked at an uploaded file:

  SPACING      Every site model is trained on native 0.3 mm voxels, and a
               96-voxel patch is only 28.8 mm because of it. A scan at another
               spacing is resampled to the training spacing -- image linearly,
               mask by nearest neighbour so no class is invented at a boundary
               -- and the result says so.

  ORIENTATION  The z flip is decided by anatomy, never by the affine; see
               `implant_sites.superior_sign`. With a mask that anatomy is read
               directly. Without one the sign comes from the caller: the site
               localiser's orientation head when one is loaded, else the
               configured default, which is MEASURED rather than assumed --
               all 522 ToothFairy3 scans whose anatomy resolves came out at -1
               (`sites_toothfairy3.csv`, column orientation_sign). Either way
               the source of the sign is recorded on the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from src.data.implant_sites import superior_sign
from src.data.scan import prepare_volume


@dataclass
class PreparedScan:
    volume: np.ndarray                 # float16, oriented, z-scored, at `spacing`
    spacing: tuple[float, float, float]
    sign: int
    sign_source: str
    original_shape: tuple[int, ...]
    original_spacing: tuple[float, float, float]
    mask: np.ndarray | None = None     # RAW frame (not flipped), at `spacing`
    fg_mean: float = float("nan")
    fg_std: float = float("nan")
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "shape": list(self.volume.shape),
            "spacing_mm": [round(s, 4) for s in self.spacing],
            "original_shape": list(self.original_shape),
            "original_spacing_mm": [round(s, 4) for s in self.original_spacing],
            "orientation_sign": self.sign,
            "orientation_source": self.sign_source,
            "has_mask": self.mask is not None,
            "fg_mean": self.fg_mean,
            "fg_std": self.fg_std,
        }


def patient_id_from_filename(name: str) -> str:
    """'ToothFairy3F_001_0000.nii.gz' -> 'ToothFairy3F_001' (nnU-Net channel suffix dropped)."""
    stem = Path(name).name
    for suffix in (".nii.gz", ".nii"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    if len(stem) > 5 and stem[-5] == "_" and stem[-4:].isdigit():
        stem = stem[:-5]
    return stem


def _load(path: str | Path, dtype) -> tuple[np.ndarray, tuple[float, float, float]]:
    img = nib.load(str(path))
    shape = img.shape
    if len(shape) == 4 and shape[3] == 1:
        data = np.asarray(img.dataobj, dtype=dtype)[..., 0]
    elif len(shape) == 3:
        data = np.asarray(img.dataobj, dtype=dtype)
    else:
        raise ValueError(f"{Path(path).name}: expected a 3D volume, got shape {shape}")
    spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
    if not all(np.isfinite(s) and s > 0 for s in spacing):
        raise ValueError(f"{Path(path).name}: invalid voxel spacing {spacing}")
    # np.array, not asarray: prepare_volume writes in place, and asarray can hand
    # back a read-only proxy buffer when the on-disk dtype already matches.
    return np.array(data, dtype=dtype, copy=True), spacing


def resample(volume: np.ndarray, spacing, target: float, order: int) -> np.ndarray:
    zoom = np.asarray(spacing, dtype=np.float64) / float(target)
    return ndimage.zoom(volume, zoom, order=order, mode="nearest", prefilter=order > 1)


def prepare_scan(
    image_path: str | Path,
    mask_path: str | Path | None,
    clip_window,
    air_threshold: float,
    target_spacing: float,
    spacing_tolerance: float,
    sign: int | None = None,
    sign_source: str | None = None,
) -> PreparedScan:
    """Load, check, resample if needed, orient and normalise one scan.

    `sign` is used only when there is no mask; with a mask the anatomy decides,
    and a contradicting `sign` raises rather than picking one.
    """
    image, spacing = _load(image_path, np.float32)
    original_shape, original_spacing = tuple(image.shape), spacing
    warnings: list[str] = []

    mask = None
    if mask_path is not None:
        mask, mask_spacing = _load(mask_path, np.int16)
        if mask.shape != image.shape:
            raise ValueError(
                f"image {image.shape} and mask {mask.shape} disagree; site coordinates "
                f"are measured on the mask and read from the image")
        if not np.allclose(mask_spacing, spacing, rtol=1e-3):
            raise ValueError(f"image spacing {spacing} and mask spacing {mask_spacing} disagree")

    off = [abs(s - target_spacing) / target_spacing for s in spacing]
    if max(off) > spacing_tolerance:
        image = resample(image, spacing, target_spacing, order=1)
        if mask is not None:
            mask = resample(mask, spacing, target_spacing, order=0).astype(np.int16)
        warnings.append(
            f"scan spacing {tuple(round(s, 3) for s in spacing)} mm resampled to "
            f"{target_spacing} mm isotropic, the resolution the model was trained at. "
            f"Fine structure below the original resolution cannot be recovered.")
        spacing = (float(target_spacing),) * 3

    if mask is not None:
        anatomical = superior_sign(mask)
        if sign is not None and sign != anatomical:
            raise ValueError(f"orientation {sign} was requested but the mask's anatomy "
                             f"says {anatomical}")
        sign, sign_source = anatomical, "mask anatomy"
    elif sign is None:
        raise ValueError("no mask and no orientation sign: the z direction is unknown")

    volume, mean, std, fell_back = prepare_volume(image, int(sign), clip_window, air_threshold)
    if fell_back:
        warnings.append("no voxels above the air threshold: z-scored over the whole "
                        "volume, so intensities are not comparable with the training cohort")
    return PreparedScan(
        volume=np.ascontiguousarray(volume), spacing=spacing, sign=int(sign),
        sign_source=sign_source or "caller", original_shape=original_shape,
        original_spacing=original_spacing, mask=mask, fg_mean=mean, fg_std=std,
        warnings=warnings,
    )
