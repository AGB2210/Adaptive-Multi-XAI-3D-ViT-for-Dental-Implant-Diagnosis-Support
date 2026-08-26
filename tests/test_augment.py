"""Augmentation tests: anatomically sane transforms, reproducible, non-destructive."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.data.augment import Augment3D

BASE = dict(
    translate_voxels=0, translate_prob=0.0, flip_lr_prob=0.0, rotate_deg=0.0, rotate_prob=0.0,
    intensity_shift=0.0, intensity_scale=0.0, gamma_range=[1.0, 1.0], gamma_prob=0.0,
    noise_std=0.0, noise_prob=0.0,
)


def cfg(**over):
    return SimpleNamespace(**{**BASE, **over})


def volume(shape=(16, 16, 16)):
    v = np.zeros(shape, dtype=np.float32)
    v[4:8, 4:12, 4:12] = 1.0  # off-centre block, so shifts and flips are visible
    return v


def test_identity_config_is_a_noop():
    v = volume()
    out = Augment3D(cfg())(v, np.random.default_rng(0))
    assert np.array_equal(out, v)


def test_flip_is_left_right_only():
    v = volume()
    out = Augment3D(cfg(flip_lr_prob=1.0))(v, np.random.default_rng(0))
    assert np.array_equal(out, v[::-1])
    # Anterior-posterior and superior-inferior must be untouched.
    assert not np.array_equal(out, v[:, ::-1])
    assert not np.array_equal(out, v[:, :, ::-1])


def test_translate_shifts_without_wrapping():
    v = volume()
    aug = Augment3D(cfg(translate_voxels=3, translate_prob=1.0))
    out = aug(v, np.random.default_rng(0))
    assert out.shape == v.shape
    # Mass is preserved or reduced (content can leave the box), never duplicated.
    assert out.sum() <= v.sum() + 1e-5
    assert not np.array_equal(out, v)


def test_translate_fills_with_air_not_wraparound():
    v = np.zeros((8, 8, 8), dtype=np.float32)
    v[0] = 5.0  # a slab at the very edge
    aug = Augment3D(cfg(translate_voxels=2, translate_prob=1.0))
    for seed in range(10):
        out = aug(v, np.random.default_rng(seed))
        # If it wrapped, the slab could reappear at the opposite face.
        assert not (out[0].max() > 0 and out[-1].max() > 0)


def test_output_is_finite_and_float32_under_everything():
    aug = Augment3D(cfg(
        translate_voxels=4, translate_prob=1.0, flip_lr_prob=1.0, rotate_deg=10.0, rotate_prob=1.0,
        intensity_shift=0.1, intensity_scale=0.1, gamma_range=[0.7, 1.5], gamma_prob=1.0,
        noise_std=0.05, noise_prob=1.0,
    ))
    out = aug(volume(), np.random.default_rng(0))
    assert out.dtype == np.float32 and np.isfinite(out).all() and out.shape == (16, 16, 16)


def test_reproducible_under_the_same_rng_seed():
    aug = Augment3D(cfg(translate_voxels=4, translate_prob=0.5, flip_lr_prob=0.5,
                        rotate_deg=10.0, rotate_prob=0.5, noise_std=0.05, noise_prob=0.5))
    v = volume()
    a = aug(v, np.random.default_rng(42))
    b = aug(v, np.random.default_rng(42))
    assert np.array_equal(a, b)


def test_different_seeds_give_different_results():
    aug = Augment3D(cfg(translate_voxels=4, translate_prob=1.0))
    v = volume()
    assert not np.array_equal(aug(v, np.random.default_rng(1)), aug(v, np.random.default_rng(7)))


def test_input_is_not_mutated():
    v = volume()
    before = v.copy()
    Augment3D(cfg(translate_voxels=3, translate_prob=1.0, noise_std=0.1, noise_prob=1.0))(
        v, np.random.default_rng(0)
    )
    assert np.array_equal(v, before)
