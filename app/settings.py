"""App settings, read from a project config that `extends: sites.yaml`."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from src.utils.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "app.yaml"


def _ns_dict(ns) -> dict:
    return dict(vars(ns)) if ns is not None and not isinstance(ns, dict) else dict(ns or {})


@dataclass
class Settings:
    config_path: Path
    data_dir: Path
    host: str
    port: int
    device: torch.device
    spacing_mm: float
    spacing_tolerance: float
    default_orientation_sign: int
    min_orientation_accuracy: float
    allow_unsafe_checkpoints: bool
    window: tuple[float, float]
    clip_window: tuple[float, float]
    air_threshold: float
    rules: dict
    site_jaws: tuple[str, ...]
    xai: dict = field(default_factory=dict)

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def scans_dir(self) -> Path:
        return self.data_dir / "scans"


def resolve_device(name: str) -> torch.device:
    name = (name or "auto").lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("app.device is 'cuda' but no CUDA device is available")
    return torch.device(name)


def load_settings(config_path: str | Path | None = None, **overrides) -> Settings:
    path = Path(config_path or DEFAULT_CONFIG)
    cfg = load_config(path)
    app = getattr(cfg, "app", None)
    if app is None:
        raise SystemExit(f"{path} has no `app:` block -- point --config at configs/app.yaml "
                         f"or a config that extends it")

    rules = {k: float(v) for k, v in _ns_dict(cfg.sites).items()
             if k in ("min_height_mandible_mm", "min_height_maxilla_mm", "min_width_mm")}
    missing = {"min_height_mandible_mm", "min_height_maxilla_mm", "min_width_mm"} - set(rules)
    if missing:
        raise SystemExit(f"sites.{sorted(missing)} not set: clinical thresholds are never defaulted")

    data_dir = Path(overrides.get("data_dir") or app.data_dir)
    if not data_dir.is_absolute():
        data_dir = REPO_ROOT / data_dir
    return Settings(
        config_path=path,
        data_dir=data_dir,
        host=str(overrides.get("host") or app.host),
        port=int(overrides.get("port") or app.port),
        device=resolve_device(overrides.get("device") or getattr(app, "device", "auto")),
        spacing_mm=float(app.spacing_mm),
        spacing_tolerance=float(app.spacing_tolerance),
        default_orientation_sign=int(app.default_orientation_sign),
        min_orientation_accuracy=float(app.min_orientation_accuracy),
        allow_unsafe_checkpoints=bool(getattr(app, "allow_unsafe_checkpoints", False)),
        window=tuple(float(v) for v in getattr(app, "window", (-2.0, 4.0))),
        clip_window=tuple(float(v) for v in cfg.preprocess.clip_window),
        air_threshold=float(getattr(cfg.preprocess, "air_threshold", -500.0)),
        rules=rules,
        site_jaws=tuple(getattr(cfg.task, "site_jaws", ("lower",))),
        xai=_ns_dict(getattr(app, "xai", None)),
    )
