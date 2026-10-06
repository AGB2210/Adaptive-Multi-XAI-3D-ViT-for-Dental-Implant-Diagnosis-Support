"""A toy CBCT and its mask, and a tiny site checkpoint, for the inference tests.

The mask uses ToothFairy3's real class indices, so the label builder's own
`score_one` locates and measures its sites exactly as it would a real scan:
both jaws (for orientation), ten lower teeth on a parabolic arch with four
missing (so some sites need an implant), and an inferior alveolar canal under
the ridge. `flipped=True` stores it the ToothFairy3 way up -- the z index
increasing DOWNWARD -- which is the case every real scan is.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from src.data.dental_arch import LOWER_ARCH
from src.data.implant_sites import IAC, LOWER_JAW, UPPER_JAW
from src.models.vit3d import ViT3D
from src.train.targets import TargetSpec

SHAPE = (100, 100, 70)
SPACING = (0.3, 0.3, 0.3)
MISSING = (34, 35, 36, 37)
NAMES = ["needs_implant", "available_height_mm", "ridge_width_mm"]


def arch_xy(tooth: int) -> tuple[int, int]:
    i = LOWER_ARCH.index(tooth)
    u = (i - 6.5) / 6.5
    return int(round(50 + 34 * u)), int(round(28 + 38 * u * u))


def toy_mask(flipped: bool = True) -> np.ndarray:
    """Oriented (z up) anatomy, then optionally stored upside down."""
    m = np.zeros(SHAPE, dtype=np.int16)
    m[8:92, 14:84, 10:30] = LOWER_JAW            # mandible: crest at z = 29
    m[8:92, 14:84, 16:18] = IAC[0]               # canal roof at z = 17
    m[8:92, 14:84, 52:60] = UPPER_JAW            # maxilla above, for orientation
    for tooth in LOWER_ARCH:
        if tooth in MISSING:
            continue
        x, y = arch_xy(tooth)
        m[x - 2:x + 3, y - 2:y + 3, 26:38] = tooth
    return m[:, :, ::-1].copy() if flipped else m


def toy_image(mask: np.ndarray, seed: int = 0) -> np.ndarray:
    """HU-like intensities consistent with the mask."""
    rng = np.random.default_rng(seed)
    img = np.full(mask.shape, -1000.0, dtype=np.float32)
    img[mask == LOWER_JAW] = 900.0
    img[mask == UPPER_JAW] = 900.0
    img[np.isin(mask, IAC)] = 40.0
    img[(mask >= 31) & (mask <= 47)] = 2200.0
    img += rng.normal(0.0, 30.0, size=mask.shape).astype(np.float32)
    return img


def write_toy_scan(folder: Path, flipped: bool = True, spacing=SPACING,
                   image_name: str = "TOY_001_0000.nii.gz") -> tuple[Path, Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    # Built the right way up and flipped together, so the two storage
    # orientations hold the same anatomy AND the same noise.
    mask = toy_mask(flipped=False)
    image = toy_image(mask)
    if flipped:
        mask, image = mask[:, :, ::-1].copy(), image[:, :, ::-1].copy()
    affine = np.diag(list(spacing) + [1.0])
    image_path, mask_path = folder / image_name, folder / "TOY_001.nii.gz"
    img = nib.Nifti1Image(image, affine)
    img.header.set_zooms(spacing)
    nib.save(img, str(image_path))
    lab = nib.Nifti1Image(mask, affine)
    lab.header.set_zooms(spacing)
    nib.save(lab, str(mask_path))
    return image_path, mask_path


MODEL_CONFIG = {"name": "vit3d", "in_channels": 1, "stem_channels": 4, "embed_dim": 16,
                "patch_size": 2, "depth": 1, "num_heads": 2, "mlp_ratio": 4.0,
                "dropout": 0.0, "attn_dropout": 0.0, "drop_path": 0.0,
                "num_classes": 3, "img_size": 16}


def tiny_model(seed: int = 0) -> ViT3D:
    torch.manual_seed(seed)
    c = MODEL_CONFIG
    return ViT3D(in_channels=1, num_classes=3, img_size=c["img_size"],
                 stem_channels=c["stem_channels"], embed_dim=c["embed_dim"],
                 patch_size=c["patch_size"], depth=c["depth"], num_heads=c["num_heads"],
                 mlp_ratio=c["mlp_ratio"], drop_path=0.0).eval()


def tiny_spec() -> TargetSpec:
    return TargetSpec(binary=["needs_implant"],
                      millimetres=["available_height_mm", "ridge_width_mm"],
                      mean=np.array([14.0, 8.0]), std=np.array([4.0, 2.0]))


def write_tiny_checkpoint(path: Path, embed_config: bool = True, seed: int = 0,
                          **overrides) -> Path:
    payload = {"epoch": 3, "model": tiny_model(seed).state_dict(),
               "label_names": list(NAMES), "target_spec": tiny_spec().state_dict()}
    if embed_config:
        payload["model_config"] = dict(MODEL_CONFIG)
    payload.update(overrides)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path
