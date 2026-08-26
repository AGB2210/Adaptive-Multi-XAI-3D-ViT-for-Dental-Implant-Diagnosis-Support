"""Mask -> case-label derivation.

The failure this guards against is silent: a wrong class index, or a threshold
applied to the wrong column, produces a perfectly well-formed CSV that trains a
model against the wrong anatomy.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_labels import class_indices, count_classes  # noqa: E402


def _mask(tmp_path: Path, spec: dict[int, int]) -> Path:
    """A mask where class `k` occupies `spec[k]` voxels."""
    vol = np.zeros(1000, dtype=np.int16)
    at = 0
    for cls, n in spec.items():
        vol[at : at + n] = cls
        at += n
    path = tmp_path / "labels.nii.gz"
    nib.save(nib.Nifti1Image(vol.reshape(10, 10, 10), np.eye(4)), str(path))
    return path


def test_counts_only_the_requested_classes(tmp_path):
    path = _mask(tmp_path, {7: 120, 8: 30, 9: 5})
    got = count_classes(path, {"implant": 7, "crown": 8})
    assert got == {"implant": 120, "crown": 30}


def test_absent_class_counts_zero_rather_than_raising(tmp_path):
    # A cohort where no case has an implant must yield zeros, not a crash --
    # the zero is the finding.
    path = _mask(tmp_path, {7: 50})
    assert count_classes(path, {"implant": 7, "bridge": 42}) == {"implant": 50, "bridge": 0}


def test_class_indices_refuses_to_guess():
    cfg = SimpleNamespace(task=SimpleNamespace(class_indices={"implant": 7}))
    with pytest.raises(SystemExit, match="crown"):
        class_indices(cfg, ["implant", "crown"])


def test_class_indices_reads_config_mapping():
    cfg = SimpleNamespace(task=SimpleNamespace(class_indices={"implant": 7, "crown": 8}))
    assert class_indices(cfg, ["crown", "implant"]) == {"crown": 8, "implant": 7}
