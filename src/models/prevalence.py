"""The chance floor: what a model that ignores the image scores.

Not a network and not a baseline architecture -- it is the number every other
result has to be read against. AUROC is 0.5 by construction; average precision is
NOT zero but the label's prevalence, which is the value most often misread. A
macro AP of 0.36 against labels averaging 0.32 prevalence is nothing at all.
"""

from __future__ import annotations

import numpy as np


class PrevalenceBaseline:
    """Predicts the training-set prevalence for every case. Nothing to train."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.prevalence: np.ndarray | None = None

    def fit(self, y: np.ndarray) -> "PrevalenceBaseline":
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 2 or y.shape[1] != self.num_classes:
            raise ValueError(f"expected (n, {self.num_classes}) labels, got {y.shape}")
        self.prevalence = y.mean(axis=0)
        return self

    def predict_proba(self, n: int) -> np.ndarray:
        if self.prevalence is None:
            raise RuntimeError("call fit() first")
        return np.tile(self.prevalence, (n, 1))
