"""Find the fourteen lower-jaw tooth sites on a scan that has no segmentation.

`src/data/site_dataset.py` says it plainly: site positions come from the masks,
and a fielded system needs a site-detection step that does not exist here. This
is that step. It is a separate trained model with its own measured error, and
everything downstream of it inherits that error -- a patch cut 3 mm from where
the label builder would have cut it still looks like a jaw.

WHAT IT PREDICTS, per site (FDI 47..41, 31..37, in arch order):

  position   (x, y, z) in the oriented full-resolution voxel frame -- the frame
             `sites_toothfairy3.csv` stores site_x / site_y / site_z in, so a
             prediction can be cut by `patch_centre` exactly like a mask-derived
             one. z is the CREST height, which is what the label build stores.
  spread     the standard deviation of the heatmap, in millimetres: how sure
             the network is of that position. Reported, never hidden.
  in view    whether the site is inside the field of view at all.

and once per scan:

  orientation  whether the input is upside down. Every mask-derived orientation
               is decided by anatomy (`implant_sites.superior_sign`); a scan with
               no mask has no anatomy to read, so this head is trained on
               randomly z-flipped inputs and its accuracy is measured by
               `scripts/eval_localiser.py` like any other number.

THE INPUT is the model-frame volume `src.data.scan.prepare_volume` produces,
block-averaged by `factor` (4 -> 1.2 mm) and centre-padded or cropped to a fixed
grid. Measured over all 532 ToothFairy3 scans: every one is 0.3 mm isotropic and
at most 512 x 512 x 298 voxels, so 128 x 128 x 80 at 1.2 mm holds every scan
whole. `localiser_input` is the one function both the cache build and the app
call, so the two cannot disagree about the grid.

HEATMAPS, NOT DIRECT REGRESSION. Regressing fourteen coordinates from a pooled
feature throws away exactly the spatial information the task needs; a heatmap
per site with a soft-argmax keeps the answer in the image's own coordinates and
gives the spread for free.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.dental_arch import LOWER_ARCH
from src.models.vit3d import ResBlock3d


@dataclass
class LocaliserConfig:
    factor: int = 4
    input_shape: tuple[int, int, int] = (128, 128, 80)
    channels: tuple[int, ...] = (16, 32, 64, 128)
    sites: tuple[int, ...] = tuple(LOWER_ARCH)
    spacing_mm: float = 0.3

    def __post_init__(self):
        self.input_shape = tuple(int(v) for v in self.input_shape)
        self.channels = tuple(int(v) for v in self.channels)
        self.sites = tuple(int(v) for v in self.sites)
        levels = len(self.channels) - 1
        for n in self.input_shape:
            if n % (2 ** levels):
                raise ValueError(f"input_shape {self.input_shape} must be divisible by "
                                 f"{2 ** levels} for {len(self.channels)} levels")

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: list(v) if isinstance(v, tuple) else v for k, v in d.items()}


# ---------------------------------------------------------------- the grid
def localiser_input(volume: np.ndarray, factor: int, shape) -> tuple[np.ndarray, np.ndarray]:
    """Full-resolution model-frame volume -> (fixed-shape low-res input, offset).

    Block mean over `factor`^3 voxels (anti-aliased, unlike striding), then
    centre pad or crop to `shape`. `offset` is added to a low-res index to get
    its position in the fixed grid; it is negative on an axis that was cropped.
    """
    factor = int(factor)
    n = [s // factor for s in volume.shape]
    if min(n) < 1:
        raise ValueError(f"volume {volume.shape} is smaller than one {factor}^3 block")
    trimmed = np.asarray(volume[: n[0] * factor, : n[1] * factor, : n[2] * factor], dtype=np.float32)
    low = trimmed.reshape(n[0], factor, n[1], factor, n[2], factor).mean(axis=(1, 3, 5))

    out = np.full(tuple(shape), float(low.min()), dtype=np.float32)
    offset = np.array([(int(o) - int(m)) // 2 for o, m in zip(shape, low.shape)])
    src, dst = [], []
    for axis in range(3):
        lo = offset[axis]
        s0, s1 = max(0, -lo), min(low.shape[axis], shape[axis] - lo)
        src.append(slice(s0, s1))
        dst.append(slice(s0 + lo, s1 + lo))
    out[tuple(dst)] = low[tuple(src)]
    return out, offset


def to_input_coords(full: np.ndarray, offset, factor: int) -> np.ndarray:
    """Full-resolution voxel coordinates -> fixed-grid coordinates."""
    return (np.asarray(full, dtype=np.float64) - (factor - 1) / 2.0) / factor + np.asarray(offset)


def to_full_coords(grid: np.ndarray, offset, factor: int) -> np.ndarray:
    """Fixed-grid coordinates -> full-resolution voxel coordinates. Inverse of the above."""
    return (np.asarray(grid, dtype=np.float64) - np.asarray(offset)) * factor + (factor - 1) / 2.0


# ---------------------------------------------------------------- the network
def soft_argmax(heatmaps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, S, D, H, W) logits -> (B, S, 3) expected index, (B, S) spread in voxels."""
    b, s, d, h, w = heatmaps.shape
    prob = F.softmax(heatmaps.reshape(b, s, -1).float(), dim=-1).reshape(b, s, d, h, w)
    axes = [torch.arange(n, device=heatmaps.device, dtype=torch.float32) for n in (d, h, w)]
    px = prob.sum(dim=(3, 4))
    py = prob.sum(dim=(2, 4))
    pz = prob.sum(dim=(2, 3))
    mean = torch.stack([(px * axes[0]).sum(-1), (py * axes[1]).sum(-1), (pz * axes[2]).sum(-1)], -1)
    var = ((px * axes[0] ** 2).sum(-1) - mean[..., 0] ** 2
           + (py * axes[1] ** 2).sum(-1) - mean[..., 1] ** 2
           + (pz * axes[2] ** 2).sum(-1) - mean[..., 2] ** 2)
    return mean, var.clamp_min(0).sqrt()


class SiteLocaliser(nn.Module):
    """3D U-Net to half resolution, one heatmap per site, plus two pooled heads."""

    def __init__(self, cfg: LocaliserConfig):
        super().__init__()
        self.cfg = cfg
        ch = cfg.channels
        n_sites = len(cfg.sites)
        self.stem = ResBlock3d(1, ch[0])
        self.down = nn.ModuleList([ResBlock3d(ch[i], ch[i + 1], stride=2) for i in range(len(ch) - 1)])
        # Decoder back to HALF resolution: 2.4 mm voxels at factor 4, finer than a
        # tooth and fine enough for soft-argmax to place a site between voxels.
        self.up = nn.ModuleList([ResBlock3d(ch[i + 1] + ch[i], ch[i])
                                 for i in reversed(range(1, len(ch) - 1))])
        self.head = nn.Conv3d(ch[1], n_sites, kernel_size=1)
        self.valid = nn.Linear(ch[-1], n_sites)
        # Orientation reads a TOP-TO-BOTTOM PROFILE, pooled over x and y only.
        # A global average -- the first version -- discards exactly the
        # up/down arrangement the question is about, and its accuracy sat at
        # 50% on both orientations for 40 epochs while positions trained fine.
        depth = cfg.input_shape[2] // 2 ** (len(ch) - 1)
        self.flip = nn.Linear(ch[-1] * depth, 1)

    def forward(self, x: torch.Tensor) -> dict:
        feats = [self.stem(x)]
        for block in self.down:
            feats.append(block(feats[-1]))
        y = feats[-1]
        for i, block in enumerate(self.up):
            skip = feats[-2 - i]
            y = F.interpolate(y, size=skip.shape[2:], mode="trilinear", align_corners=False)
            y = block(torch.cat([y, skip], dim=1))
        heat = self.head(y)                                # (B, S, D/2, H/2, W/2)
        coords, spread = soft_argmax(heat)
        pooled = feats[-1].mean(dim=(2, 3, 4))
        profile = feats[-1].mean(dim=(2, 3)).flatten(1)     # (B, C * depth)
        # Half-resolution index -> fixed-grid index (centre of a 2-voxel block).
        return {"coords": coords * 2.0 + 0.5, "spread": spread * 2.0,
                "valid_logit": self.valid(pooled), "flip_logit": self.flip(profile)[:, 0]}


def localiser_loss(out: dict, target: torch.Tensor, coord_mask: torch.Tensor,
                   valid: torch.Tensor, flipped: torch.Tensor, voxel_mm: float,
                   spread_weight: float = 0.05) -> dict:
    """Coordinate error in mm (smooth L1, masked per axis) + spread + in-view + orientation.

    `coord_mask` is per axis, not per site: a site whose crest could not be
    measured still has a valid (x, y) from the arch fit, and its z must not be
    trained toward a NaN.

    THE SPREAD TERM is what makes the reported spread mean anything. A
    soft-argmax trained on coordinates alone is free to keep a wide heatmap
    whose MEAN lands in the right place: measured on the smoke run, positions
    within a few millimetres came with spreads of 17-48 mm on every site, so
    every site was flagged uncertain and the flag carried no information.
    Penalising the spread (in mm, on sites that are in view) sharpens the
    heatmaps, as in DSNT's variance regularisation.
    """
    err = (out["coords"] - torch.nan_to_num(target)) * voxel_mm
    per = F.smooth_l1_loss(err, torch.zeros_like(err), reduction="none", beta=1.0)
    m = coord_mask.float()
    coord = (per * m).sum() / m.sum().clamp_min(1.0)
    in_view = valid.float()
    spread = (out["spread"] * voxel_mm * in_view).sum() / in_view.sum().clamp_min(1.0)
    valid_loss = F.binary_cross_entropy_with_logits(out["valid_logit"], valid.float())
    flip_loss = F.binary_cross_entropy_with_logits(out["flip_logit"], flipped.float())
    return {"total": coord + spread_weight * spread + valid_loss + flip_loss,
            "coord_mm": coord, "spread_mm": spread, "valid": valid_loss, "flip": flip_loss}


# ---------------------------------------------------------------- loading
@dataclass
class Localiser:
    """A loaded localiser, ready to `locate` sites on a prepared volume."""

    model: SiteLocaliser
    cfg: LocaliserConfig
    device: torch.device
    meta: dict = field(default_factory=dict)

    def describe(self) -> dict:
        return {"kind": "localiser", "sites": list(self.cfg.sites),
                "factor": self.cfg.factor, "input_shape": list(self.cfg.input_shape),
                "grid_mm": round(self.cfg.factor * self.cfg.spacing_mm, 3),
                "epoch": self.meta.get("epoch"), "fold": self.meta.get("fold"),
                "validation": self.meta.get("val") or {},
                "n_params": int(sum(p.numel() for p in self.model.parameters()))}

    @torch.no_grad()
    def predict(self, volume: np.ndarray) -> dict:
        x, offset = localiser_input(volume, self.cfg.factor, self.cfg.input_shape)
        out = self.model(torch.from_numpy(x)[None, None].to(self.device))
        grid = out["coords"][0].float().cpu().numpy()
        return {
            "full": to_full_coords(grid, offset, self.cfg.factor),
            "spread_mm": out["spread"][0].float().cpu().numpy() * self.cfg.factor * self.cfg.spacing_mm,
            "valid_prob": torch.sigmoid(out["valid_logit"][0]).float().cpu().numpy(),
            "flip_prob": float(torch.sigmoid(out["flip_logit"][0])),
            "cropped": bool((offset < 0).any()),
        }

    def locate(self, volume: np.ndarray, spacing_mm: float, jaws=("lower",)) -> dict:
        """Sites in the same shape `sites_from_mask` returns, without truth."""
        if abs(float(spacing_mm) - self.cfg.spacing_mm) > 1e-3:
            raise ValueError(f"this localiser was trained at {self.cfg.spacing_mm} mm, "
                             f"the volume is at {spacing_mm} mm")
        p = self.predict(volume)
        warnings = []
        if "upper" in jaws:
            warnings.append("the localiser finds lower-jaw sites only")
        if p["cropped"]:
            warnings.append("the scan is larger than the localiser's grid and was cropped "
                            "at its edges; sites near the border may be missed")
        error = (self.meta.get("val") or {}).get("median_error_mm")
        warnings.append(
            "Tooth sites were located by the localiser model, not a segmentation"
            + (f" (median position error {error:.1f} mm on its validation patients)" if error else "")
            + ". A position error moves the patch and every measurement with it.")

        # "Uncertain" is relative to THIS model: a site whose heatmap is more than
        # twice as spread as this localiser's median on its validation patients.
        # A fixed millimetre cutoff would be a guess about a model it has never
        # seen, and on the smoke run flagged every site, which says nothing.
        typical = (self.meta.get("val") or {}).get("median_spread_mm")

        sites = []
        for i, tooth in enumerate(self.cfg.sites):
            x, y, z = (float(v) for v in p["full"][i])
            in_view = float(p["valid_prob"][i])
            spread = float(p["spread_mm"][i])
            inside = all(0 <= v < n for v, n in zip((x, y, z), volume.shape))
            site = {"tooth": int(tooth), "jaw": "lower", "site_x": x, "site_y": y, "site_z": z,
                    "source": "localiser", "method": "localiser", "anchors": None, "truth": None,
                    "position_sd_mm": spread, "in_view_prob": in_view}
            if in_view < 0.5 or not inside:
                site.update({"site_x": None, "site_y": None, "site_z": None,
                             "note": f"the localiser places this site outside the field of view "
                                     f"(P(in view) {in_view:.2f})"})
            elif typical and spread > 2.0 * float(typical):
                site["note"] = (f"uncertain position: heatmap spread ± {spread:.1f} mm, over twice "
                                f"this localiser's typical {float(typical):.1f} mm")
            sites.append(site)
        return {"sites": sites, "flip": p["flip_prob"] > 0.5,
                "flip_prob": p["flip_prob"], "warnings": warnings,
                # Measured on this checkpoint's validation patients, each shown
                # both ways up. None for a checkpoint that never recorded it.
                "orientation_accuracy": (self.meta.get("val") or {}).get("orientation_accuracy")}


def load_localiser(path, device: torch.device, allow_unsafe: bool = False) -> Localiser:
    from src.inference.checkpoint import CheckpointError, load_checkpoint_dict

    ckpt = load_checkpoint_dict(path, allow_unsafe=allow_unsafe)
    if not isinstance(ckpt, dict) or ckpt.get("kind") != "localiser":
        raise CheckpointError("not a localiser checkpoint (no kind='localiser'); a site "
                              "model is added with kind 'Site model'")
    cfg = LocaliserConfig(**ckpt["localiser_config"])
    model = SiteLocaliser(cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return Localiser(model=model, cfg=cfg, device=device,
                     meta={k: ckpt.get(k) for k in ("epoch", "fold", "val")})
