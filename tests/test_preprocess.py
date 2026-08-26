"""Preprocessing geometry tests.

The point of these is spacing-correctness: a shape-aware pipeline passes some of
them, a spacing-aware one passes all of them.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from src.data.preprocess import (
    block_mean_decimate,
    fit_to_shape,
    preprocess_volume,
    spacing_from_affine,
)
from tests.synthetic import make_nifti

OUT = (32, 32, 32)


def test_spacing_read_from_affine_not_header():
    affine = np.diag([0.4, 0.3, 0.5, 1.0])
    assert np.allclose(spacing_from_affine(affine), (0.4, 0.3, 0.5))


def test_spacing_correct_under_oblique_affine():
    """Rotate the direction matrix: column norms must still give the true spacing."""
    theta = np.deg2rad(30)
    rot = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    affine = np.eye(4)
    affine[:3, :3] = rot @ np.diag([0.4, 0.3, 0.5])
    assert np.allclose(spacing_from_affine(affine), (0.4, 0.3, 0.5))


def test_block_mean_decimate_reduces_and_updates_spacing():
    vol = np.arange(8 * 8 * 8, dtype=np.float32).reshape(8, 8, 8)
    out, spacing = block_mean_decimate(vol, np.array([0.25, 0.25, 0.25]), np.array([1.0, 1.0, 1.0]))
    assert out.shape == (2, 2, 2)
    assert np.allclose(spacing, 1.0)
    # Block mean, not subsampling: the first cell is the mean of its 4^3 block.
    assert np.isclose(out[0, 0, 0], vol[:4, :4, :4].mean())


def test_decimate_is_a_noop_when_upsampling():
    vol = np.zeros((4, 4, 4), dtype=np.float32)
    out, spacing = block_mean_decimate(vol, np.array([1.0, 1.0, 1.0]), np.array([0.5, 0.5, 0.5]))
    assert out.shape == (4, 4, 4) and np.allclose(spacing, 1.0)


def test_isotropic_pad_gives_constant_effective_spacing():
    """The whole point: two volumes of different FOV land on the same mm/voxel."""
    big = np.zeros((100, 100, 100), dtype=np.float32)
    small = np.zeros((40, 40, 40), dtype=np.float32)
    _, eff_big = fit_to_shape(big, np.array([1.0] * 3), OUT, "isotropic_pad", 1.0, 0.0)
    _, eff_small = fit_to_shape(small, np.array([1.0] * 3), OUT, "isotropic_pad", 1.0, 0.0)
    assert np.allclose(eff_big, eff_small) and np.allclose(eff_big, 1.0)


def test_resize_effective_spacing_varies_with_fov():
    """Documents why resize mode breaks cross-cohort comparability."""
    big = np.zeros((100, 100, 100), dtype=np.float32)
    small = np.zeros((40, 40, 40), dtype=np.float32)
    _, eff_big = fit_to_shape(big, np.array([1.0] * 3), OUT, "resize", 1.0, 0.0)
    _, eff_small = fit_to_shape(small, np.array([1.0] * 3), OUT, "resize", 1.0, 0.0)
    assert not np.allclose(eff_big, eff_small)


def test_isotropic_pad_preserves_physical_size_of_a_structure():
    """A 20 mm cube must occupy ~20 voxels at 1 mm/voxel, whatever the input grid."""
    sizes = {}
    for spacing in (0.25, 0.5, 1.0):
        n = int(60 / spacing)
        vol = np.zeros((n, n, n), dtype=np.float32)
        w = int(20 / spacing)
        start = (n - w) // 2
        vol[start : start + w, start : start + w, start : start + w] = 1.0

        out, eff = fit_to_shape(vol, np.full(3, spacing), (64, 64, 64), "isotropic_pad", 1.0, 0.0)
        assert np.allclose(eff, 1.0)
        sizes[spacing] = (out > 0.5).sum()

    values = list(sizes.values())
    assert min(values) > 0
    # All three should land near 20^3 = 8000 voxels.
    assert max(values) / min(values) < 1.25, sizes


def test_fit_to_shape_always_returns_out_shape():
    for shape in [(10, 10, 10), (100, 40, 77), (33, 33, 33)]:
        vol = np.zeros(shape, dtype=np.float32)
        for mode in ("isotropic_pad", "resize"):
            out, _ = fit_to_shape(vol, np.array([0.6, 0.6, 0.6]), OUT, mode, 1.0, 0.0)
            assert out.shape == OUT, (shape, mode)


def test_unknown_fit_mode_raises():
    with pytest.raises(ValueError, match="unknown fit_mode"):
        fit_to_shape(np.zeros((8, 8, 8), np.float32), np.ones(3), OUT, "bogus", 1.0, 0.0)


def test_end_to_end_on_a_non_isotropic_volume(tmp_path):
    path = make_nifti(tmp_path / "v.nii.gz", shape=(60, 50, 40), spacing=(0.4, 0.3, 0.5), seed=2)
    arr, info = preprocess_volume(path, "t1", "synthetic", out_shape=OUT, fit_mode="isotropic_pad")

    assert arr is not None and info.status == "ok"
    assert arr.shape == OUT and arr.dtype == np.float16
    assert np.isfinite(arr.astype(np.float32)).all()
    assert info.orig_spacing_mm.startswith("0.4000,0.3000,0.5000")
    assert np.allclose([float(v) for v in info.effective_spacing_mm.split(",")], 1.0)


def _hu_volume(tmp_path, shape=(60, 60, 60), spacing=(0.5, 0.5, 0.5), seed=3):
    """An HU-calibrated volume: air at -1000 outside, tissue/bone inside.

    This is the path the real data takes -- both datasets peak at -1000.
    """
    import nibabel as nib

    rng = np.random.default_rng(seed)
    vol = np.full(shape, -1000.0, dtype=np.float32)
    sl = tuple(slice(s // 5, 4 * s // 5) for s in shape)
    vol[sl] = rng.normal(100, 200, size=tuple(s.stop - s.start for s in sl))
    path = tmp_path / "hu.nii.gz"
    nib.save(nib.Nifti1Image(vol, np.diag(list(spacing) + [1.0])), str(path))
    return path


def test_hu_volume_uses_fixed_threshold_and_normalises_foreground(tmp_path):
    path = _hu_volume(tmp_path)
    arr, info = preprocess_volume(path, "t2", "synthetic", out_shape=OUT, fit_mode="resize")

    assert arr is not None and info.threshold_method == "fixed_hu"
    a = arr.astype(np.float32)
    assert np.isfinite(a).all()
    # Foreground stats were used and are sane.
    assert info.fg_std > 0 and -200 < info.fg_mean < 500
    # Foreground voxels land near zero mean / unit std; air sits well below.
    fg = a[a > float(a.mean())]
    assert abs(float(fg.mean())) < 2.0 and 0.1 < float(a.std()) < 5.0


def test_non_hu_volume_falls_back_to_otsu_and_warns(tmp_path):
    path = make_nifti(tmp_path / "v.nii.gz", shape=(50, 50, 50), spacing=(0.5, 0.5, 0.5), seed=3)
    arr, info = preprocess_volume(path, "t3", "synthetic", out_shape=OUT, fit_mode="resize")
    assert arr is not None
    assert info.threshold_method == "otsu" and "HU" in info.warning
    assert np.isfinite(arr.astype(np.float32)).all()


def test_corrupt_volume_fails_loudly_without_emitting_an_array(tmp_path):
    bad = tmp_path / "bad.nii.gz"
    bad.write_bytes(b"this is not a nifti file")
    arr, info = preprocess_volume(bad, "broken", "synthetic", out_shape=OUT)
    assert arr is None
    assert info.status == "failed" and info.warning


def test_constant_volume_is_rejected(tmp_path):
    import nibabel as nib

    path = tmp_path / "flat.nii.gz"
    nib.save(nib.Nifti1Image(np.full((40, 40, 40), 5.0, dtype=np.float32), np.eye(4)), str(path))
    arr, info = preprocess_volume(path, "flat", "synthetic", out_shape=OUT)
    assert arr is None and info.status == "failed"


# --- fixed-window clipping ------------------------------------------------
# The regression these guard against: a per-patient percentile clip erases metal,
# which is the only thing that makes a dental implant visible in CBCT.

def test_fixed_clip_preserves_metal_that_percentile_clip_destroys():
    from src.data.preprocess import preprocess_volume

    rng = np.random.default_rng(0)
    vol = rng.normal(400.0, 150.0, size=(64, 64, 64)).astype(np.float32)
    vol[:8] = -1000.0                      # air
    vol[30:34, 30:34, 30:38] = 6500.0      # a metal implant: hyperdense, tiny

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "v.nii.gz"
        nib.save(nib.Nifti1Image(vol, np.eye(4)), str(path))

        out_pct, _ = preprocess_volume(path, "p", "mmdental", out_shape=(48, 48, 48),
                                       clip_mode="percentile", clip_percentiles=(0.5, 99.5))
        out_fix, _ = preprocess_volume(path, "p", "mmdental", out_shape=(48, 48, 48),
                                       clip_mode="fixed", clip_window=(-1000.0, 6000.0))

    assert out_pct is not None and out_fix is not None
    # The implant is 0.03% of the volume, so a 99.5th-percentile clip flattens it
    # onto the bulk tissue; a fixed window leaves it standing far above.
    assert out_fix.astype("float32").max() > out_pct.astype("float32").max() * 1.5


def test_clip_mode_rejects_unknown_values():
    from src.data.preprocess import preprocess_volume

    vol = np.random.default_rng(0).normal(400.0, 150.0, size=(32, 32, 32)).astype(np.float32)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "v.nii.gz"
        nib.save(nib.Nifti1Image(vol, np.eye(4)), str(path))
        arr, info = preprocess_volume(path, "p", "mmdental", out_shape=(24, 24, 24),
                                      clip_mode="nonsense")
    # Failures are reported, never emitted as a silently-wrong array.
    assert arr is None
    assert info.status == "failed"
    assert "clip_mode" in info.warning
