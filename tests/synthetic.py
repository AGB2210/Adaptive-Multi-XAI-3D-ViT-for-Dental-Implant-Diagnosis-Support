"""Synthetic CBCT-ish volumes with a planted, known-location signal.

Two jobs:
  1. Let every module be tested without touching 265 GB of real scans.
  2. Give the XAI phase a case where the ground-truth signal location is known,
     so attribution maps can be scored rather than eyeballed.

Each of the 6 labels owns a fixed site. A positive label plants a bright blob
there; SIGNAL_SITES is the ground truth an attribution map should recover.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

# Normalised (z, y, x) centres in [0, 1] -- one distinct site per label slot.
# These are deliberately NOT the real task's label names. The convergence gate
# tests the machinery, not the dataset: a planted blob at a known location is a
# task the model must be able to learn, so failing it means the code is broken
# rather than the problem being hard. Slots are taken in order, so a model with
# three heads uses the first three; add coordinates here to support wider heads.
SIGNAL_SITES: list[tuple[float, float, float]] = [
    (0.55, 0.35, 0.30),
    (0.60, 0.38, 0.70),
    (0.45, 0.62, 0.28),
    (0.50, 0.65, 0.72),
    (0.40, 0.50, 0.50),
    (0.68, 0.28, 0.50),
    (0.35, 0.40, 0.40),
    (0.62, 0.55, 0.60),
]

#: Names for the synthetic slots, used only for readable test output.
LABELS = [f"synthetic_{i}" for i in range(len(SIGNAL_SITES))]


def n_slots() -> int:
    return len(SIGNAL_SITES)

BLOB_RADIUS = 0.06  # fraction of the volume's smallest side
BLOB_AMPLITUDE = 3.0  # in units of background sigma


def _background(shape: tuple[int, int, int], rng: np.random.Generator) -> np.ndarray:
    """Smooth low-frequency texture plus a jaw-like arch, so montages look plausible."""
    vol = ndimage.gaussian_filter(rng.standard_normal(shape).astype(np.float32), sigma=4.0)
    vol /= vol.std() + 1e-8

    zz, yy, xx = np.meshgrid(*[np.linspace(0, 1, s) for s in shape], indexing="ij")
    # A horseshoe arch in the axial plane, thick in z around the mid-slice.
    arch = np.exp(-(((yy - 0.35 - 1.6 * (xx - 0.5) ** 2) / 0.06) ** 2))
    arch *= np.exp(-(((zz - 0.55) / 0.18) ** 2))
    return vol + 2.5 * arch.astype(np.float32)


def _plant(vol: np.ndarray, centre: tuple[float, float, float], radius: float, amp: float) -> None:
    shape = vol.shape
    zz, yy, xx = np.meshgrid(*[np.linspace(0, 1, s) for s in shape], indexing="ij")
    d2 = (zz - centre[0]) ** 2 + (yy - centre[1]) ** 2 + (xx - centre[2]) ** 2
    vol += amp * np.exp(-d2 / (2 * radius**2)).astype(np.float32)


DEFAULT_PREVALENCE = np.array([0.44, 0.40, 0.37, 0.34, 0.27, 0.25, 0.22, 0.20])


def make_case(
    seed: int,
    shape: tuple[int, int, int] = (128, 128, 128),
    prevalence: float | None = None,
    labels: np.ndarray | None = None,
    n_labels: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """One synthetic volume and its binary labels.

    Pass `labels` to force a label vector, or `prevalence` to draw them. The
    label count follows `labels` when given, so the fixture matches whatever head
    width the model under test has, up to n_slots().
    """
    rng = np.random.default_rng(seed)
    if labels is None:
        k = n_labels
        p = (np.full(k, prevalence, dtype=np.float64) if prevalence is not None
             else DEFAULT_PREVALENCE[:k])
        labels = (rng.random(k) < p).astype(np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size > n_slots():
        raise ValueError(f"{labels.size} labels requested but only {n_slots()} planted "
                         "sites are defined; add coordinates to SIGNAL_SITES")

    vol = _background(shape, rng)
    for i in range(len(labels)):
        if labels[i]:
            jitter = rng.normal(0, 0.015, size=3)
            centre = tuple(float(np.clip(c + j, 0.1, 0.9)) for c, j in zip(SIGNAL_SITES[i], jitter))
            _plant(vol, centre, BLOB_RADIUS, BLOB_AMPLITUDE)

    vol = (vol - vol.mean()) / (vol.std() + 1e-8)
    return vol.astype(np.float32), labels


def make_dataset(
    n: int,
    seed: int = 0,
    shape: tuple[int, int, int] = (32, 32, 32),
    n_labels: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """A small stack of cases. Default shape is tiny -- these are for unit tests.

    `n_labels` must match the task being gated. It used to be fixed at six, so
    running the gate on a two-output config produced (n, 6) targets against a
    two-class model and failed inside the prevalence baseline -- an unhelpful
    place to learn that the gate does not know what it is gating.
    """
    if not 1 <= n_labels <= len(SIGNAL_SITES):
        raise ValueError(
            f"n_labels must be between 1 and {len(SIGNAL_SITES)} "
            f"(one planted site per label), got {n_labels}"
        )
    vols, ys = zip(*(make_case(seed + i, shape=shape, n_labels=n_labels) for i in range(n)))
    return np.stack(vols)[:, None], np.stack(ys)


def signal_mask(labels: np.ndarray, shape: tuple[int, int, int] = (128, 128, 128)) -> np.ndarray:
    """Ground-truth support of the planted signal -- the XAI phase scores against this."""
    mask = np.zeros(shape, dtype=bool)
    zz, yy, xx = np.meshgrid(*[np.linspace(0, 1, s) for s in shape], indexing="ij")
    for i in range(len(labels)):
        if labels[i]:
            c = SIGNAL_SITES[i]
            d2 = (zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2
            mask |= d2 <= (2 * BLOB_RADIUS) ** 2
    return mask


def make_nifti(
    path,
    shape: tuple[int, int, int] = (64, 48, 40),
    spacing: tuple[float, float, float] = (0.4, 0.3, 0.5),
    seed: int = 0,
    axcodes: str = "RAS",
):
    """Write a non-isotropic NIfTI so the preprocessing geometry test has a case
    where correct spacing handling is distinguishable from shape-only handling."""
    import nibabel as nib
    from nibabel.orientations import axcodes2ornt, inv_ornt_aff, ornt_transform

    rng = np.random.default_rng(seed)
    vol = _background(shape, rng)
    # A dense cube of known physical size, so a resampler can be checked in mm.
    vol[shape[0] // 4 : shape[0] // 2, shape[1] // 4 : shape[1] // 2, shape[2] // 4 : shape[2] // 2] += 6.0

    affine = np.diag(list(spacing) + [1.0])
    if axcodes != "RAS":
        ornt = ornt_transform(axcodes2ornt("RAS"), axcodes2ornt(axcodes))
        affine = affine @ inv_ornt_aff(ornt, shape)

    img = nib.Nifti1Image(vol.astype(np.float32), affine)
    img.header.set_zooms(spacing)
    nib.save(img, str(path))
    return path
