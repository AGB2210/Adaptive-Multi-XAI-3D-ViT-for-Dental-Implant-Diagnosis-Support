"""Load any site-model checkpoint, and refuse one whose architecture is unclear.

A `.pt` file is chosen at run time, so nothing here may assume which model it
holds. Three things have to be known before its outputs mean anything:

  ARCHITECTURE  Most of it is written in the tensor shapes and is read from
                there: embed width, depth, patch size, stem width, MLP ratio,
                head width and the input size the positional embedding fixes.
                `num_heads` is NOT -- the qkv projection is (3*dim, dim) however
                the heads divide it -- so it must be declared, and a wrong
                value would load cleanly and compute a different function.
                Declared values come from the checkpoint itself (written by
                training since `model_config` was added), else a config passed
                with it, else a sidecar YAML, else the app's own config. Every
                declared value that the shapes can check IS checked, and any
                disagreement refuses the load rather than picking a winner.

  UNITS         The standardiser in `target_spec`. Without it the millimetre
                heads report standardised units that look like millimetres.

  CALIBRATION   Temperature and confidence-gate threshold, fitted on that
                checkpoint's own validation split by `run_adaptive.py` and
                written to `calibration.json` beside it. Absent, the app says
                so on every result instead of quietly using T = 1.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.models.vit3d import build_vit3d
from src.train.targets import TargetSpec
from src.utils.config import load_config
from src.utils.log import get_logger

log = get_logger("inference.checkpoint")

# What the app needs a site model to predict. The binary head is optional --
# without it there is no "needs an implant" call -- but feasibility is defined on
# these two lengths, so a checkpoint without them cannot do the job at all.
REQUIRED_MM = ("available_height_mm", "ridge_width_mm")

# Shape-checkable architecture fields. `num_heads` is deliberately absent.
SHAPE_FIELDS = ("in_channels", "stem_channels", "embed_dim", "patch_size",
                "depth", "mlp_ratio", "num_classes", "img_size")

# Inference-time no-ops that `build_vit3d` still wants a value for.
EVAL_DEFAULTS = {"dropout": 0.0, "attn_dropout": 0.0, "drop_path": 0.0}


class CheckpointError(ValueError):
    """The file cannot be used as a site model. The message says why."""


def read_architecture(state: dict) -> dict:
    """Every architecture field the weight tensors fix, read off their shapes."""
    needed = ("cls_token", "pos_embed", "patch_embed.proj.weight",
              "stem.block1.conv1.weight", "head.weight", "blocks.0.mlp.fc1.weight")
    missing = [k for k in needed if k not in state]
    if missing:
        raise CheckpointError(
            f"not a vit3d checkpoint: no {', '.join(missing)} in its weights. The app "
            f"explains predictions with attention rollout, which needs the ViT.")

    embed_dim = int(state["cls_token"].shape[-1])
    n_tokens = int(state["pos_embed"].shape[1]) - 1
    grid = int(round(n_tokens ** (1.0 / 3.0)))
    if grid ** 3 != n_tokens:
        raise CheckpointError(f"positional embedding holds {n_tokens} patch tokens, "
                              f"which is not a cube -- not a 3D ViT this app knows")
    proj = state["patch_embed.proj.weight"]          # (embed, stem, p, p, p)
    patch_size = int(proj.shape[-1])
    depth = len({m.group(1) for k in state
                 if (m := re.match(r"blocks\.(\d+)\.attn\.qkv\.weight$", k))})
    hidden = int(state["blocks.0.mlp.fc1.weight"].shape[0])
    return {
        "in_channels": int(state["stem.block1.conv1.weight"].shape[1]),
        "stem_channels": int(proj.shape[1]),
        "embed_dim": embed_dim,
        "patch_size": patch_size,
        "depth": depth,
        "mlp_ratio": hidden / embed_dim,
        "num_classes": int(state["head.weight"].shape[0]),
        # The conv stem has stride 2, so one token spans 2 * patch_size input
        # voxels. Getting this wrong once put a token at 4.8 mm.
        "img_size": grid * 2 * patch_size,
        "grid": grid,
    }


def _declared_from_config(path: Path) -> dict:
    cfg = load_config(path)
    model = dict(vars(cfg.model))
    if model.get("img_size") is None:
        shape = getattr(getattr(cfg, "preprocess", None), "out_shape", None)
        model["img_size"] = int(shape[0]) if shape else None
    return model


def _sidecar(path: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = path.parent / name
        if candidate.is_file():
            return candidate
    return None


def resolve_declared(ckpt: dict, path: Path, config_path: str | Path | None,
                     fallback_config: str | Path | None) -> tuple[dict, str]:
    """(declared model config, where it came from), most specific first."""
    embedded = ckpt.get("model_config")
    if isinstance(embedded, dict) and embedded:
        return dict(embedded), "embedded in the checkpoint"
    if config_path is not None:
        return _declared_from_config(Path(config_path)), f"config {Path(config_path).name}"
    side = _sidecar(path, (f"{path.stem}.yaml", "config.yaml"))
    if side is not None:
        return _declared_from_config(side), f"sidecar {side.name}"
    if fallback_config is not None:
        return (_declared_from_config(Path(fallback_config)),
                f"app config {Path(fallback_config).name} (the checkpoint does not say)")
    raise CheckpointError("the checkpoint carries no model_config and no config was "
                          "supplied, so num_heads is unknown")


def check_declared(shapes: dict, declared: dict) -> list[str]:
    """Every shape-checkable field where the declaration and the weights disagree."""
    problems = []
    for key in SHAPE_FIELDS:
        want = declared.get(key)
        if want is None:
            continue
        got = shapes[key]
        same = (abs(float(want) - float(got)) < 1e-6) if key == "mlp_ratio" else int(want) == int(got)
        if not same:
            problems.append(f"{key}: declared {want}, weights say {got}")
    heads = declared.get("num_heads")
    if heads is None:
        problems.append("num_heads: not declared, and the weights cannot say")
    elif shapes["embed_dim"] % int(heads):
        problems.append(f"num_heads: {heads} does not divide embed_dim {shapes['embed_dim']}")
    name = declared.get("name", "vit3d")
    if name != "vit3d":
        problems.append(f"name: declared {name!r}; the app supports vit3d only")
    return problems


def read_calibration(path: Path | None) -> dict:
    """Temperature and gate from a `run_adaptive.py` calibration.json, validated."""
    if path is None:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    t = data.get("temperature")
    if t is None or not np.isfinite(float(t)) or float(t) <= 0:
        # A NaN temperature is the exact failure that silently killed the gate
        # once; refusing it here is cheaper than finding it in a figure.
        raise CheckpointError(f"{Path(path).name}: temperature {t!r} is not a finite "
                              f"positive number")
    out = {"temperature": float(t), "source": Path(path).name,
           "fitted_on": data.get("fitted_on")}
    if data.get("gate_threshold") is not None and np.isfinite(float(data["gate_threshold"])):
        out["gate"] = {
            "threshold": float(data["gate_threshold"]),
            "uncertainty": data.get("gate_uncertainty", "margin"),
            "ensemble_fraction": data.get("gate_ensemble_fraction"),
            "cheap_method": data.get("gate_cheap_method", "attention_rollout"),
        }
    return out


def read_validation_metrics(path: Path | None) -> dict:
    """Decision thresholds and per-head MAE from that checkpoint's validation split."""
    if path is None:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    # metrics.json nests the validation block under "val"; best_val_metrics.json
    # is the block itself, with classification and regression side by side.
    block = data.get("val", data)
    cls = block.get("classification", block)
    out = {"source": Path(path).name,
           "thresholds": {k: float(v) for k, v in (cls.get("thresholds") or {}).items()
                          if v is not None},
           "mae": {k: float(v["mae"]) for k, v in (block.get("regression") or {}).items()
                   if isinstance(v, dict) and v.get("mae") is not None}}
    return out


@dataclass
class ModelBundle:
    """A loaded site model and everything needed to read its outputs honestly."""

    model: torch.nn.Module
    device: torch.device
    spec: TargetSpec
    names: list[str]
    architecture: dict
    architecture_source: str
    epoch: int | None = None
    n_params: int = 0
    temperature: float = 1.0
    calibration_source: str | None = None
    gate: dict | None = None
    thresholds: dict = field(default_factory=dict)
    mae: dict = field(default_factory=dict)
    metrics_source: str | None = None
    path: str = ""

    @property
    def n_binary(self) -> int:
        return len(self.spec.binary)

    @property
    def img_size(self) -> int:
        return int(self.architecture["img_size"])

    @property
    def grid_size(self) -> tuple[int, int, int]:
        return tuple(self.model.grid_size)

    def describe(self, spacing_mm: float) -> dict:
        grid = self.grid_size[0]
        return {
            "path": self.path,
            "epoch": self.epoch,
            "n_params": self.n_params,
            "outputs": [{"name": n, "unit": "probability" if i < self.n_binary else "mm"}
                        for i, n in enumerate(self.names)],
            "architecture": {k: self.architecture.get(k) for k in
                             ("embed_dim", "depth", "num_heads", "patch_size",
                              "stem_channels", "mlp_ratio", "img_size")},
            "architecture_source": self.architecture_source,
            "tokens": f"{grid}x{grid}x{grid}",
            "mm_per_token": round(self.img_size / grid * spacing_mm, 3),
            "patch_mm": round(self.img_size * spacing_mm, 2),
            "temperature": self.temperature,
            "calibrated": self.calibration_source is not None,
            "calibration_source": self.calibration_source,
            "gate": self.gate,
            "decision_thresholds": self.thresholds,
            "validation_mae_mm": self.mae,
            "metrics_source": self.metrics_source,
        }


def load_checkpoint_dict(path: str | Path, allow_unsafe: bool = False) -> dict:
    """torch.load, refusing to unpickle arbitrary objects unless told otherwise.

    `src.train.loop.load_checkpoint_file` falls back to the permissive loader
    with a warning, which is right for a file this project wrote on its own
    rented box. A file picked in a browser is not that, and the permissive
    loader runs whatever code the pickle carries.
    """
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - any rejection means "not plain data"
        if not allow_unsafe:
            raise CheckpointError(
                f"{Path(path).name} is not a plain-data checkpoint "
                f"({type(exc).__name__}); refusing to unpickle it. Set "
                f"app.allow_unsafe_checkpoints only for files you produced yourself."
            ) from exc
        log.warning("%s: loading with weights_only=False (allowed by config)", path)
        return torch.load(str(path), map_location="cpu", weights_only=False)


def load_bundle(
    path: str | Path,
    device: torch.device,
    config_path: str | Path | None = None,
    fallback_config: str | Path | None = None,
    calibration_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    allow_unsafe: bool = False,
) -> ModelBundle:
    path = Path(path)
    ckpt = load_checkpoint_dict(path, allow_unsafe=allow_unsafe)
    if not isinstance(ckpt, dict):
        raise CheckpointError("the file does not hold a checkpoint dictionary")
    state = ckpt.get("model", ckpt)
    if not isinstance(state, dict) or not state:
        raise CheckpointError("the checkpoint holds no model weights")

    shapes = read_architecture(state)
    declared, source = resolve_declared(ckpt, path, config_path, fallback_config)
    problems = check_declared(shapes, declared)
    if problems:
        raise CheckpointError(
            "the declared architecture does not match the weights (" + source + "):\n  - "
            + "\n  - ".join(problems))

    arch = {**EVAL_DEFAULTS, **{k: shapes[k] for k in SHAPE_FIELDS},
            "num_heads": int(declared["num_heads"]), "name": "vit3d"}
    arch["patch_size"] = int(arch["patch_size"])
    model = build_vit3d(SimpleNamespace(**arch), img_size=int(arch["img_size"]))
    # strict: a key the model does not have, or lacks, is a different network.
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    for param in model.parameters():
        # Attribution hangs the autograd graph off the input, never the weights.
        param.requires_grad_(False)

    spec = TargetSpec.from_state(ckpt.get("target_spec"))
    names = list(ckpt.get("label_names") or spec.names)
    if len(names) != shapes["num_classes"]:
        raise CheckpointError(f"the head has {shapes['num_classes']} outputs but the "
                              f"checkpoint names {len(names)}: {names}")
    if spec.names and spec.names != names:
        raise CheckpointError(f"label_names {names} disagree with target_spec {spec.names}; "
                              f"one target's value would be reported under another's name")
    missing = [n for n in REQUIRED_MM if n not in spec.millimetres]
    if missing:
        raise CheckpointError(
            f"this checkpoint predicts {names}, not the site task's millimetre heads "
            f"({', '.join(missing)} missing). Feasibility is a rule on those two lengths, "
            f"so the app cannot use it.")
    if spec.mean is None or spec.std is None:
        raise CheckpointError("the checkpoint carries no millimetre standardiser, so its "
                              "outputs cannot be converted back to millimetres")

    calibration = read_calibration(
        Path(calibration_path) if calibration_path else _sidecar(path, ("calibration.json",)))
    metrics = read_validation_metrics(
        Path(metrics_path) if metrics_path
        else _sidecar(path, ("best_val_metrics.json", "metrics.json")))

    return ModelBundle(
        model=model, device=device, spec=spec, names=names,
        architecture=arch, architecture_source=source,
        epoch=ckpt.get("epoch"),
        n_params=int(sum(p.numel() for p in model.parameters())),
        temperature=calibration.get("temperature", 1.0),
        calibration_source=calibration.get("source"),
        gate=calibration.get("gate"),
        thresholds=metrics.get("thresholds", {}),
        mae=metrics.get("mae", {}),
        metrics_source=metrics.get("source"),
        path=str(path),
    )
