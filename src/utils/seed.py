"""Deterministic seeding across random / numpy / torch."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG we use.

    deterministic=True also pins torch to reproducible kernels. That is right for
    TRAINING, where it costs little: the forward/backward is dominated by AMP
    matmuls and the weight-gradient path.

    It is ruinous for ATTRIBUTION. Integrated Gradients backpropagates to the
    INPUT, so it leans on cuDNN's 3D convolution data-gradient kernels, whose
    deterministic variants are ~100x slower here -- measured 16s vs >25min for
    one 256-step IG map on a 128^3 volume.

    And it does not buy determinism anyway: PyTorch's memory-efficient attention
    backward has no deterministic implementation, so with warn_only=True it warns
    and proceeds non-deterministically regardless. The XAI stage therefore passes
    deterministic=False and relies on RNG seeding, which is what actually makes
    the sampling methods (GradientSHAP, LIME) reproducible.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:  # label/preprocess stages do not need torch
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Required by some cuBLAS kernels before they can run deterministically.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker a distinct but reproducible seed."""
    import torch

    seed = (torch.initial_seed() + worker_id) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
