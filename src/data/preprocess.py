"""Offline preprocessing: NIfTI -> compact, normalised 128^3 float16 arrays.

At 328-515 MB per volume, decompressing and resampling inside the training loop
is not viable. This module is run once, offline, by scripts/build_cache.py.

Pipeline per volume:
  1. load, reorient to RAS+ canonical
  2. voxel spacing read from the affine (correct under oblique acquisition)
  3. foreground bbox from an air threshold, + margin
  4. resample to the target grid -- see fit_mode below
  5. clip (fixed window or percentiles), z-score on foreground voxels only
  6. save float16 .npy

fit_mode controls step 4, and the choice matters for cross-hospital transfer:

  "isotropic_pad" (default) -- resample to a true `target_spacing` mm isotropic
      grid, then centre pad/crop to out_shape. Every voxel means the same
      physical distance in both cohorts. Cost: out_shape must cover the anatomy
      (128 voxels @ 1 mm = 128 mm), so wider fields of view lose their margins.

  "resize" -- stretch the crop straight onto out_shape, as the original spec
      literally describes. Keeps all anatomy, but effective spacing then varies
      per patient AND per dataset. Measured on the earlier cohort pair:
      1.25 mm/voxel at a 160 mm FOV versus 0.63 mm/voxel at 80 mm, so the
      arrives ~2x larger than anything the model saw in training. It also
      distorts MMDental anisotropically (1.25 in-plane vs 0.78 axial).

Either way `effective_spacing_mm` is recorded per patient in the manifest.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from src.models.geometric import otsu as _otsu

# Both datasets are HU-calibrated with the air peak at exactly -1000.
DEFAULT_AIR_THRESHOLD = -500.0
# Volumes whose minimum sits above this are not HU-like; fall back to Otsu.
HU_SANITY_MIN = -200.0
SUBSAMPLE = 4  # bbox is found on a 4x-decimated copy, then mapped back


@dataclass
class VolumeInfo:
    """One row of artifacts/preprocess_manifest.csv."""

    patient_id: str
    dataset: str
    status: str = "ok"
    warning: str = ""
    orig_shape: str = ""
    orig_spacing_mm: str = ""
    axcodes_in: str = ""
    threshold: float = float("nan")
    threshold_method: str = ""
    fg_fraction: float = float("nan")
    crop_shape: str = ""
    crop_extent_mm: str = ""
    effective_spacing_mm: str = ""
    fit_mode: str = ""
    coverage: float = float("nan")
    clip_mode: str = ""
    clip_lo: float = float("nan")
    clip_hi: float = float("nan")
    fg_mean: float = float("nan")
    fg_std: float = float("nan")
    out_min: float = float("nan")
    out_max: float = float("nan")
    elapsed_s: float = float("nan")
    extras: dict = field(default_factory=dict)


def spacing_from_affine(affine: np.ndarray) -> np.ndarray:
    """Voxel size in mm as the column norms of the direction matrix.

    Correct for oblique acquisitions, where header zooms can disagree.
    """
    return np.sqrt((np.asarray(affine)[:3, :3] ** 2).sum(axis=0))


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu threshold -- delegates to the ONE implementation, in `models.geometric`.

    This file used to carry a second copy. Both used the correct between-class
    variance, but only the other one handles the case where the maximum is a
    PLATEAU: two well-separated peaks leave an empty gap, every threshold inside
    it scores identically, and `argmax` then returns the gap's left edge, hard
    against the darker peak. On the phantom named in that docstring -- peaks at
    -1 and +1, where the answer is 0.0 -- this copy returned -0.8387 while the
    other returned -0.0197.

    So the copy is deleted rather than repaired. Two implementations of one
    thing is what put the anatomy masks 7.2 mm from the model's input (C8k.1),
    and a bug fixed in one of them is not fixed.

    `bins` is kept in the signature for callers; the shared implementation
    defaults lower, and the value is passed through.
    """
    return _otsu(values, bins=bins)


def foreground_bbox(
    small: np.ndarray,
    threshold: float,
    margin: float,
    factor: int,
    full_shape: tuple[int, int, int],
    robust_pct: float = 0.5,
) -> tuple[tuple[slice, slice, slice], float]:
    """Bounding box of the foreground, found on a decimated copy and mapped back.

    Uses percentiles of the foreground coordinates rather than min/max, so a few
    stray bright voxels (scanner table, reconstruction artefacts) cannot blow the
    box out to the full volume.
    """
    mask = small > threshold
    fg_fraction = float(mask.mean())
    if not mask.any():
        return tuple(slice(0, s) for s in full_shape), fg_fraction  # type: ignore[return-value]

    slices = []
    for axis in range(3):
        idx = np.where(mask.any(axis=tuple(a for a in range(3) if a != axis)))[0]
        lo, hi = np.percentile(idx, [robust_pct, 100 - robust_pct])
        lo, hi = float(lo) * factor, float(hi + 1) * factor
        pad = margin * (hi - lo)
        lo = int(max(0, np.floor(lo - pad)))
        hi = int(min(full_shape[axis], np.ceil(hi + pad)))
        if hi <= lo:  # degenerate axis -> keep everything
            lo, hi = 0, full_shape[axis]
        slices.append(slice(lo, hi))

    return (slices[0], slices[1], slices[2]), fg_fraction


def block_mean_decimate(
    vol: np.ndarray, spacing: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Anti-aliased integer decimation toward `target` spacing, per axis.

    Going 0.25 mm -> 1 mm is a 4x reduction. Plain linear interpolation samples
    every 4th voxel and aliases: three quarters of the signal is thrown away
    rather than averaged, which speckles fine trabecular bone. Averaging over
    integer blocks first is the standard anti-aliasing step, and it is much
    faster than interpolating the full-resolution array.
    """
    factors = [max(1, int(np.floor(t / s))) for s, t in zip(spacing, target)]
    if all(f == 1 for f in factors):
        return vol, spacing

    # Trim the tail so each axis divides evenly into blocks.
    trimmed = vol[tuple(slice(0, (n // f) * f) for n, f in zip(vol.shape, factors))]
    if any(s == 0 for s in trimmed.shape):  # volume smaller than one block
        return vol, spacing

    n0, n1, n2 = (n // f for n, f in zip(trimmed.shape, factors))
    f0, f1, f2 = factors
    out = trimmed.reshape(n0, f0, n1, f1, n2, f2).mean(axis=(1, 3, 5), dtype=np.float32)
    return out, spacing * np.array(factors, dtype=np.float64)


def fit_to_shape(
    vol: np.ndarray,
    spacing: np.ndarray,
    out_shape: tuple[int, int, int],
    fit_mode: str,
    target_spacing: float,
    pad_value: float,
    order: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample onto the output grid. Returns (volume, effective spacing in mm)."""
    if fit_mode == "resize":
        eff = np.array(vol.shape) * spacing / np.array(out_shape)
        vol, spacing = block_mean_decimate(vol, spacing, eff)
        zoom = np.array(out_shape, dtype=np.float64) / np.array(vol.shape, dtype=np.float64)
        vol = ndimage.zoom(vol, zoom, order=order, mode="nearest", prefilter=order > 1)

    elif fit_mode == "isotropic_pad":
        eff = np.full(3, float(target_spacing))
        vol, spacing = block_mean_decimate(vol, spacing, eff)
        zoom = spacing / float(target_spacing)
        if not np.allclose(zoom, 1.0):
            vol = ndimage.zoom(vol, zoom, order=order, mode="nearest", prefilter=order > 1)

        # Centre the anatomy: crop the axes that are too long, pad those too short.
        out = np.full(out_shape, pad_value, dtype=np.float32)
        src, dst = [], []
        for axis in range(3):
            n, m = vol.shape[axis], out_shape[axis]
            if n >= m:
                start = (n - m) // 2
                src.append(slice(start, start + m))
                dst.append(slice(0, m))
            else:
                start = (m - n) // 2
                src.append(slice(0, n))
                dst.append(slice(start, start + n))
        out[tuple(dst)] = vol[tuple(src)]
        return out, eff

    else:
        raise ValueError(f"unknown fit_mode {fit_mode!r} (expected 'isotropic_pad' or 'resize')")

    if vol.shape != tuple(out_shape):  # zoom rounding can be off by a voxel
        fixed = np.full(out_shape, pad_value, dtype=np.float32)
        sl = tuple(slice(0, min(a, b)) for a, b in zip(vol.shape, out_shape))
        fixed[sl] = vol[sl]
        vol = fixed
    return vol, eff


def crop_geometry(img, margin: float, air_threshold: float):
    """The foreground crop box for one image, plus the threshold that produced it.

    Split out so a SEGMENTATION MASK can be put through the identical geometry.
    The box is derived from image intensities, so recomputing it from a mask
    would give a different crop and silently misalign the mask against the
    cached volume by a few millimetres -- which is exactly the error that would
    make a localisation metric look plausible and be wrong.
    """
    shape = tuple(int(v) for v in img.shape[:3])
    small = np.asarray(img.dataobj[::SUBSAMPLE, ::SUBSAMPLE, ::SUBSAMPLE], dtype=np.float32)
    if not np.isfinite(small).any():
        raise ValueError("volume contains no finite voxels")

    if float(np.nanmin(small)) <= HU_SANITY_MIN:
        threshold, method = air_threshold, "fixed_hu"
    else:
        threshold, method = otsu_threshold(small), "otsu"

    box, fg_fraction = foreground_bbox(small, threshold, margin, SUBSAMPLE, shape)
    return box, threshold, method, fg_fraction


def preprocess_mask(
    image_path: Path,
    mask_path: Path,
    class_indices: dict[str, int],
    out_shape: tuple[int, int, int] = (128, 128, 128),
    margin: float = 0.05,
    air_threshold: float = DEFAULT_AIR_THRESHOLD,
    fit_mode: str = "isotropic_pad",
    target_spacing: float = 1.0,
) -> dict[str, np.ndarray]:
    """One binary mask per class, on the SAME grid as the cached volume.

    Two things make this correct rather than approximately correct:

      * the crop box comes from the IMAGE (crop_geometry above), never from the
        mask, so the mask lands exactly where the volume did;
      * resampling uses order=0 with pad value 0. Any interpolation of a label
        map invents fractional classes at boundaries, and an implant is only a
        few voxels across at 1 mm -- linear interpolation would smear it into
        the surrounding bone and inflate every overlap score.
    """
    img = nib.as_closest_canonical(nib.load(str(image_path)))
    mask_img = nib.as_closest_canonical(nib.load(str(mask_path)))
    if tuple(mask_img.shape[:3]) != tuple(img.shape[:3]):
        raise ValueError(
            f"mask {mask_img.shape[:3]} does not match image {img.shape[:3]} for {image_path.name}"
        )

    spacing = spacing_from_affine(img.affine)
    box, _, _, _ = crop_geometry(img, margin, air_threshold)
    labels = np.array(mask_img.dataobj[box])

    out = {}
    for name, index in class_indices.items():
        binary = (labels == index).astype(np.float32)
        fitted, _ = fit_to_shape(binary, spacing, tuple(out_shape), fit_mode,
                                 target_spacing, 0.0, order=0)
        out[name] = (fitted > 0.5)
    return out


def preprocess_volume(
    path: Path,
    patient_id: str,
    dataset: str,
    out_shape: tuple[int, int, int] = (128, 128, 128),
    margin: float = 0.05,
    clip_percentiles: tuple[float, float] = (0.5, 99.5),
    clip_mode: str = "percentile",
    clip_window: tuple[float, float] = (-1000.0, 6000.0),
    air_threshold: float = DEFAULT_AIR_THRESHOLD,
    fit_mode: str = "isotropic_pad",
    target_spacing: float = 1.0,
    order: int = 1,
) -> tuple[np.ndarray | None, VolumeInfo]:
    """Run the full pipeline on one volume. Returns (array or None, manifest row).

    On any failure the array is None and info.status carries the reason -- a
    silently-wrong array is never emitted.
    """
    t0 = time.time()
    info = VolumeInfo(patient_id=patient_id, dataset=dataset)

    try:
        img = nib.load(str(path))
        img = nib.as_closest_canonical(img)  # RAS+
        shape = tuple(int(s) for s in img.shape[:3])
        spacing = spacing_from_affine(img.affine)
        info.orig_shape = "x".join(map(str, shape))
        info.orig_spacing_mm = ",".join(f"{v:.4f}" for v in spacing)
        info.axcodes_in = "".join(nib.orientations.aff2axcodes(nib.load(str(path)).affine))

        if len(img.shape) != 3:
            raise ValueError(f"expected a 3D volume, got shape {img.shape}")
        if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
            raise ValueError(f"invalid spacing from affine: {spacing}")

        box, threshold, method, fg_fraction = crop_geometry(img, margin, air_threshold)
        if method == "otsu":
            info.warning = "not HU-calibrated; used Otsu threshold"
        info.threshold, info.threshold_method = float(threshold), method
        info.fg_fraction = fg_fraction
        if fg_fraction < 0.005:
            info.warning = (info.warning + "; " if info.warning else "") + f"tiny foreground ({fg_fraction:.4f})"

        # --- load only the cropped region -------------------------------------
        # np.array, not np.asarray: nan_to_num below writes in place, and asarray
        # returns the proxy's own read-only buffer unchanged when the on-disk dtype
        # is already float32. Both shipped datasets store int16, so the conversion
        # copies and this never fired -- it would fail on the first float32 volume.
        vol = np.array(img.dataobj[box], dtype=np.float32)
        if vol.size == 0:
            raise ValueError("empty crop")
        np.nan_to_num(vol, copy=False, nan=float(threshold), posinf=float(threshold), neginf=float(threshold))

        crop_shape = vol.shape
        extent_mm = np.array(crop_shape) * spacing
        info.crop_shape = "x".join(map(str, crop_shape))
        info.crop_extent_mm = ",".join(f"{v:.1f}" for v in extent_mm)

        # --- resample onto the output grid ------------------------------------
        pad_value = float(np.percentile(vol, 1.0))  # air, so padding looks like background
        vol, eff = fit_to_shape(vol, spacing, tuple(out_shape), fit_mode, target_spacing, pad_value, order)
        info.effective_spacing_mm = ",".join(f"{v:.4f}" for v in eff)
        info.fit_mode = fit_mode
        # How much anatomy the fixed box had to discard (isotropic_pad only).
        needed = np.ceil(extent_mm / float(target_spacing)).astype(int)
        info.coverage = float(np.prod(np.minimum(needed, np.array(out_shape)) / needed)) if fit_mode == "isotropic_pad" else 1.0
        if info.coverage < 0.85:
            info.warning = (info.warning + "; " if info.warning else "") + f"box covers only {info.coverage:.0%} of the crop"

        # --- intensity normalisation ------------------------------------------
        # clip_mode matters more than it looks. A per-patient percentile is
        # scale-free, which is right for anatomy and wrong for metal: dental
        # implants are hyperdense and live in the top 0.01% of the histogram, so
        # a 99.5th-percentile clip flattens them onto dense cortical bone and the
        # implant becomes invisible. Measured on MMDental: voxels above 4000 are
        # 10.5x more common in implant-positive patients, and p99.5 is ~1540.
        # Worse, the percentile is computed per patient, so it normalises away
        # the very difference between an implanted mouth and a healthy one.
        #
        # A fixed window keeps absolute brightness comparable across patients.
        # Both cohorts sit on an HU-like scale (air at -1000), and ToothFairy4 is
        # scanner-saturated at ~3095, so [-1000, 6000] spans MMDental's metal and
        # is a no-op on ToothFairy4.
        if clip_mode == "fixed":
            lo, hi = float(clip_window[0]), float(clip_window[1])
        elif clip_mode == "percentile":
            lo, hi = np.percentile(vol, clip_percentiles)
        else:
            raise ValueError(f"clip_mode must be 'fixed' or 'percentile', got {clip_mode!r}")
        if not np.isfinite([lo, hi]).all() or hi <= lo:
            raise ValueError(f"degenerate intensity range: {lo}..{hi}")
        vol = np.clip(vol, lo, hi)
        info.clip_lo, info.clip_hi = float(lo), float(hi)
        info.clip_mode = clip_mode

        fg = vol[vol > threshold]
        if fg.size < 100:  # threshold sits above the clipped range
            fg = vol
            info.warning = (info.warning + "; " if info.warning else "") + "z-scored on all voxels (foreground empty)"
        mean, std = float(fg.mean()), float(fg.std())
        if std < 1e-6:
            raise ValueError("zero-variance volume")
        vol = (vol - mean) / std
        info.fg_mean, info.fg_std = mean, std
        info.out_min, info.out_max = float(vol.min()), float(vol.max())

        out = vol.astype(np.float16)
        if not np.isfinite(out.astype(np.float32)).all():
            raise ValueError("non-finite values after normalisation")

        info.elapsed_s = time.time() - t0
        return out, info

    except Exception as exc:  # noqa: BLE001 - fail loudly, skip, keep going
        info.status = "failed"
        info.warning = f"{type(exc).__name__}: {exc}"
        info.elapsed_s = time.time() - t0
        return None, info


def info_to_row(info: VolumeInfo) -> dict:
    row = asdict(info)
    row.pop("extras", None)
    return row
