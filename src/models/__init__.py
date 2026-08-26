"""Model registry.

`build_model` lives here rather than beside any one architecture: every script
needs the factory, and importing it from a module named after a single baseline
made the baseline look load-bearing when it is optional. Adding an architecture
is one entry in BUILDERS.
"""

from __future__ import annotations

from typing import Callable

import torch.nn as nn


def _build_vit3d(cfg, img_size: int) -> nn.Module:
    from src.models.vit3d import build_vit3d

    return build_vit3d(cfg, img_size=img_size)


def _build_cnn3d(cfg, img_size: int) -> nn.Module:
    from src.models.cnn3d import build_cnn3d

    return build_cnn3d(cfg)


#: name -> builder. Imports are deferred so a missing optional architecture
#: cannot break the ones that are present.
BUILDERS: dict[str, Callable[..., nn.Module]] = {
    "vit3d": _build_vit3d,
    "cnn3d": _build_cnn3d,
}


def build_model(cfg, img_size: int = 128) -> nn.Module:
    """Build the architecture named by cfg.name."""
    name = getattr(cfg, "name", "vit3d")
    if name not in BUILDERS:
        raise ValueError(f"unknown model name {name!r} (known: {sorted(BUILDERS)})")
    return BUILDERS[name](cfg, img_size)
