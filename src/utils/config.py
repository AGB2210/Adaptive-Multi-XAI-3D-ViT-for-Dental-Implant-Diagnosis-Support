"""Config loading: YAML -> nested namespace, with env-var overrides and root validation.

Datasets are declared in the config rather than hard-coded, so adding or swapping
a cohort is a YAML change and never a code change:

    data:
      datasets:
        toothfairy3:
          root: ToothFairy3
          cases: "labelsTr/*.nii.gz"           # glob that enumerates cases
          volume: "imagesTr/{pid}_0000.nii.gz" # relative to root, {pid} substituted
          labels: "labelsTr/{pid}.nii.gz"      # optional: voxel-level masks
      artifacts_dir: artifacts
      cache_dir: artifacts/cache

    `cases` handles both common layouts: a flat nnU-Net-style directory pair as
    above, or one directory per case with `cases: "*/"`. Without it, enumeration
    falls back to subdirectories of the root.

Each dataset's root can be overridden by an env var named <NAME>_ROOT (upper-cased,
non-alphanumerics become underscores), which always wins over the file. Roots are
runtime configuration -- they differ per machine and must never be committed.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


def _namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_namespace(v) for v in obj]
    return obj


def env_var_for(name: str) -> str:
    """Dataset name -> the env var that overrides its root. 'toothfairy3' -> TOOTHFAIRY3_ROOT."""
    return re.sub(r"[^A-Za-z0-9]", "_", name).upper() + "_ROOT"


def _deep_merge(base: dict, over: dict) -> dict:
    """Recursive dict merge; `over` wins on conflict, lists replace rather than concatenate.

    Lists replace on purpose: out_shape [256,256,256] must not become
    [128,128,128,256,256,256], and task.labels must not silently gain columns.
    """
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_raw(path: Path, seen: list[Path] | None = None) -> dict:
    """Read one YAML, resolving an `extends:` chain first.

    An experiment config should state only what it changes. Copying default.yaml
    and editing three fields means the copy silently keeps the old task.labels or
    clip_window when those move, which is the exact failure this project has hit
    before -- a config that looks right and trains against the wrong premise.
    """
    path = path.resolve()
    seen = seen or []
    if path in seen:
        chain = " -> ".join(p.name for p in [*seen, path])
        raise ValueError(f"circular extends in config chain: {chain}")
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base_name = raw.pop("extends", None)
    if base_name is None:
        return raw

    # Relative to the extending file's own directory, so a config is movable.
    base = _load_raw(Path(path.parent / base_name), [*seen, path])
    return _deep_merge(base, raw)


def load_config(path: str | Path) -> SimpleNamespace:
    """Load a YAML config, apply env-var root overrides, return a nested namespace.

    Supports `extends: other.yaml` for experiment configs that override a few
    fields of a base config.
    """
    path = Path(path)
    raw = _load_raw(path)

    for name, spec in (raw.get("data", {}).get("datasets") or {}).items():
        value = os.environ.get(env_var_for(name))
        if value:
            spec["root"] = value

    cfg = _namespace(raw)
    cfg.config_path = str(path)
    return cfg


def dataset_names(cfg: SimpleNamespace) -> list[str]:
    return list(vars(cfg.data.datasets))


def dataset_spec(cfg: SimpleNamespace, name: str) -> SimpleNamespace:
    spec = getattr(cfg.data.datasets, name, None)
    if spec is None:
        raise KeyError(f"dataset {name!r} is not declared in {cfg.config_path}; "
                       f"known: {dataset_names(cfg)}")
    return spec


def dataset_root(cfg: SimpleNamespace, name: str) -> Path:
    return Path(dataset_spec(cfg, name).root).expanduser()


def _strip_suffixes(path: Path) -> str:
    """'ToothFairy3F_001.nii.gz' -> 'ToothFairy3F_001'. Path.stem only drops one."""
    stem = path.name
    while True:
        base = Path(stem).stem
        if base == stem:
            return stem
        stem = base


def list_cases(cfg: SimpleNamespace, name: str) -> list[str]:
    """Case ids for a dataset, in sorted order.

    Driven by the `cases` glob so a flat nnU-Net layout and a directory-per-case
    layout are both a config entry rather than a branch in the callers. Ids come
    from the *labels* side when the glob points there, which also means a case
    without a mask is never enumerated and cannot silently become a negative.
    """
    root = dataset_root(cfg, name)
    pattern = getattr(dataset_spec(cfg, name), "cases", None)
    if not pattern:
        return sorted(p.name for p in root.iterdir() if p.is_dir())
    if pattern.endswith("/"):
        return sorted(p.name for p in root.glob(pattern.rstrip("/")) if p.is_dir())
    return sorted(_strip_suffixes(p) for p in root.glob(pattern) if p.is_file())


def volume_path(cfg: SimpleNamespace, name: str, pid: str) -> Path:
    return dataset_root(cfg, name) / dataset_spec(cfg, name).volume.format(pid=pid)


def label_path(cfg: SimpleNamespace, name: str, pid: str) -> Path | None:
    """Voxel-level mask for one case, or None when the dataset ships no masks."""
    pattern = getattr(dataset_spec(cfg, name), "labels", None)
    return None if not pattern else dataset_root(cfg, name) / pattern.format(pid=pid)


def validate_roots(cfg: SimpleNamespace, need: tuple[str, ...] | None = None) -> None:
    """Fail loudly, naming the missing path and the env var that would fix it."""
    problems: list[str] = []
    for name in (need if need is not None else dataset_names(cfg)):
        root = dataset_root(cfg, name)
        if not root.is_dir():
            problems.append(f"{name}: root missing: {root.resolve()}  (set {env_var_for(name)})")
        elif not any(p.is_dir() for p in root.iterdir()):
            problems.append(f"{name}: root has no case directories: {root}")
    if problems:
        raise FileNotFoundError("Dataset layout check failed:\n  - " + "\n  - ".join(problems))


def artifacts_dir(cfg: SimpleNamespace) -> Path:
    d = Path(cfg.data.artifacts_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_dir(cfg: SimpleNamespace, name: str) -> Path:
    return Path(cfg.data.cache_dir) / name
