"""A small 3D ResNet: the bar the ViT has to clear.

Optional by design. With a few hundred patients a ViT losing to this is a
plausible outcome and a reportable one -- transformers carry no built-in
assumption that nearby voxels are related, so they have to learn it from data
that may not be there. Without this control, a mediocre ViT score cannot be
distinguished from a task that is simply hard.

Deleting it means removing the "cnn3d" entry from src/models/BUILDERS and this
file. Nothing else imports it.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.vit3d import ResBlock3d


class CNN3D(nn.Module):
    """Small 3D ResNet: stem + 4 stages + global average pool + linear head."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        widths: tuple[int, ...] = (16, 32, 64, 128),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.stem = nn.Conv3d(in_channels, widths[0], kernel_size=3, stride=2, padding=1, bias=False)

        stages, prev = [], widths[0]
        for width in widths:
            stages.append(ResBlock3d(prev, width, stride=2))
            stages.append(ResBlock3d(width, width, stride=1))
            prev = width
        self.stages = nn.Sequential(*stages)

        self.norm = nn.GroupNorm(min(8, prev), prev)
        self.act = nn.GELU()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(prev, num_classes)

        self.num_classes = num_classes
        self.features: torch.Tensor | None = None
        self.cache_activations = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stages(self.stem(x))
        x = self.act(self.norm(x))
        if self.cache_activations:  # (B, C, d, h, w) -- Grad-CAM target
            self.features = x
            if x.requires_grad:
                x.retain_grad()
        x = self.pool(x).flatten(1)
        return self.head(self.drop(x))


def build_cnn3d(cfg) -> CNN3D:
    return CNN3D(num_classes=cfg.num_classes, in_channels=cfg.in_channels)
