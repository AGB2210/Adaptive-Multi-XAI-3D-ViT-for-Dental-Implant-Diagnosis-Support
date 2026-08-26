"""The synthetic generator underpins every other test -- check it is sane and deterministic."""

from __future__ import annotations

import numpy as np

from tests.synthetic import LABELS, SIGNAL_SITES, make_case, make_dataset, make_nifti, signal_mask


def test_shapes_and_determinism():
    v1, y1 = make_case(0, shape=(32, 32, 32))
    v2, y2 = make_case(0, shape=(32, 32, 32))
    assert v1.shape == (32, 32, 32) and y1.shape == (6,)   # default head width
    assert np.array_equal(v1, v2) and np.array_equal(y1, y2)

    v3, _ = make_case(1, shape=(32, 32, 32))
    assert not np.array_equal(v1, v3)


def test_normalised_output():
    vol, _ = make_case(3, shape=(32, 32, 32))
    assert abs(float(vol.mean())) < 1e-4
    assert abs(float(vol.std()) - 1.0) < 1e-3
    assert np.isfinite(vol).all()


def test_planted_signal_is_at_the_declared_site():
    """A positive label must brighten its own site relative to the same site when negative."""
    shape = (48, 48, 48)
    n = len(SIGNAL_SITES)
    off = np.zeros(n, dtype=np.int64)
    vol_off, _ = make_case(7, shape=shape, labels=off)
    for i in range(n):
        on = np.zeros(n, dtype=np.int64)
        on[i] = 1
        vol_on, _ = make_case(7, shape=shape, labels=on)

        z, y, x = (int(c * s) for c, s in zip(SIGNAL_SITES[i], shape))
        assert vol_on[z, y, x] > vol_off[z, y, x], LABELS[i]


def test_forced_labels_are_respected():
    want = np.array([1, 0, 1, 0, 1, 0])
    _, got = make_case(5, shape=(16, 16, 16), labels=want)
    assert np.array_equal(got, want)


def test_signal_mask_tracks_labels():
    labels = np.array([1, 0, 0, 0, 0, 0])
    mask = signal_mask(labels, shape=(32, 32, 32))
    assert mask.any() and not signal_mask(np.zeros(6, dtype=np.int64), shape=(32, 32, 32)).any()


def test_make_dataset_batches():
    x, y = make_dataset(4, seed=0, shape=(16, 16, 16))
    assert x.shape == (4, 1, 16, 16, 16) and y.shape == (4, 6)


def test_make_nifti_writes_requested_geometry(tmp_path):
    import nibabel as nib

    path = make_nifti(tmp_path / "t.nii.gz", shape=(20, 16, 12), spacing=(0.4, 0.3, 0.5))
    img = nib.load(str(path))
    assert img.shape == (20, 16, 12)
    assert np.allclose(img.header.get_zooms()[:3], (0.4, 0.3, 0.5), atol=1e-5)
