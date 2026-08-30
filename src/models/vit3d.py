"""3D Vision Transformer, written from scratch. No MONAI, no timm, no pretrained weights.

As configured for the site task, `configs/sites.yaml`:

    input (B, 1, 96, 96, 96)                              96 voxels = 28.8 mm
      -> conv stem: 2 residual blocks, stride 2 overall   -> (B, C, 48, 48, 48)
      -> patch embed Conv3d(kernel=4, stride=4)           -> 12x12x12 = 1,728 tokens
      -> prepend CLS + learnable positional embeddings
      -> N pre-norm encoder blocks (MHSA + MLP)
      -> LayerNorm -> CLS -> Linear -> num_outputs logits

THE STEM'S STRIDE IS PART OF THE TOKEN SIZE. One token spans `2 * patch_size`
INPUT voxels, so patch_size 4 gives 8 voxels = 2.4 mm at 0.3 mm. The defaults in
`__init__` are the superseded 128^3 whole-volume settings, where patch_size 8
gave 4.8 mm per token -- wider than the inferior alveolar canal the explanations
exist to resolve. Read `model.grid_size` rather than recomputing it.

Design notes:
  * Dense global attention, not windowed. Attention rollout has well-defined maths
    on dense attention; rollout across shifted windows does not. The stem's
    stride-2 downsampling is what keeps the token count at a tractable 512.
  * Every block can store its (B, heads, T, T) attention weights behind a flag,
    and caches its own input/output activations for hook-based attribution.
    The XAI stack depends on this; training only checks that it works.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock3d(nn.Module):
    """Pre-activation residual block. GroupNorm, not BatchNorm: batches are small
    (4 volumes) and GroupNorm is batch-independent and deterministic."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(groups, in_ch), in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.act = nn.GELU()

        self.skip: nn.Module = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class ConvStem(nn.Module):
    """Convolutional stem placed *before* patch embedding.

    Adopted from ImplantFormer (arXiv 2210.16467), whose ablation showed a conv
    stem improves performance by extracting local features prior to tokenization.
    We take that finding, not their detection head.
    """

    def __init__(self, in_channels: int = 1, channels: int = 32):
        super().__init__()
        self.block1 = ResBlock3d(in_channels, channels, stride=2)
        self.block2 = ResBlock3d(channels, channels, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block2(self.block1(x))


class PatchEmbed3d(nn.Module):
    """Non-overlapping 3D patches via a strided conv."""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 8):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        x = self.proj(x)
        grid = tuple(x.shape[2:])
        return x.flatten(2).transpose(1, 2), grid  # (B, T, D)


class Attention(nn.Module):
    """Multi-head self-attention that can hand back its own attention matrix."""

    def __init__(self, dim: int, num_heads: int = 8, attn_dropout: float = 0.0, proj_dropout: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"embed_dim {dim} must be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

        self.store_attention = False
        self.attn_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, heads, T, head_dim)

        if self.store_attention:
            # Explicit path: the fused kernel never materialises the attention matrix.
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            # Kept for XAI (rollout). retain_grad() so gradient-weighted variants work.
            self.attn_weights = attn
            if attn.requires_grad:
                attn.retain_grad()
            out = self.attn_drop(attn) @ v
        else:
            self.attn_weights = None
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0
            )

        out = out.transpose(1, 2).reshape(B, T, D)
        return self.proj_drop(self.proj(out))


class DropPath(nn.Module):
    """Stochastic depth on the residual branch."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty(x.shape[0], *([1] * (x.ndim - 1))).bernoulli_(keep)
        return x * mask / keep


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Block(nn.Module):
    """Pre-norm transformer encoder block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, attn_dropout, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), dropout)
        self.drop_path = DropPath(drop_path)

        # Activation cache for attribution methods.
        self.cache_activations = False
        self.activations: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.cache_activations:
            self.activations = x
            if x.requires_grad:
                x.retain_grad()
        return x


class ViT3D(nn.Module):
    """3D ViT trunk with a linear head over the CLS token.

    "Classifier" only on the superseded whole-volume task. On the site task the
    head is hybrid -- one binary output and two millimetre outputs -- and a
    sample is a tooth position rather than a patient. The model itself does not
    know the difference: it emits `num_classes` raw values and the loss decides
    what they mean.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 6,
        img_size: int = 128,
        stem_channels: int = 32,
        embed_dim: int = 512,
        patch_size: int = 8,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.1,
    ):
        super().__init__()
        self.stem = ConvStem(in_channels, stem_channels)
        self.patch_embed = PatchEmbed3d(stem_channels, embed_dim, patch_size)

        grid = img_size // 2 // patch_size  # stem halves the resolution
        self.grid_size = (grid, grid, grid)
        num_tokens = grid**3
        if num_tokens < 1:
            raise ValueError(f"img_size {img_size} too small for patch_size {patch_size}")

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Linearly increasing stochastic depth, the standard ViT schedule.
        rates = torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, dropout, attn_dropout, rates[i]) for i in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    # ---- XAI hooks -------------------------------------------------------
    def set_store_attention(self, flag: bool = True) -> None:
        """Turn on (B, heads, T, T) attention capture in every block."""
        for block in self.blocks:
            block.attn.store_attention = flag
            if not flag:
                block.attn.attn_weights = None

    def set_cache_activations(self, flag: bool = True) -> None:
        for block in self.blocks:
            block.cache_activations = flag
            if not flag:
                block.activations = None

    def get_attention_maps(self) -> list[torch.Tensor]:
        """Per-block attention, each (B, heads, T, T). Requires set_store_attention(True)."""
        return [b.attn.attn_weights for b in self.blocks if b.attn.attn_weights is not None]

    def get_activations(self) -> list[torch.Tensor]:
        return [b.activations for b in self.blocks if b.activations is not None]

    def tokens_to_grid(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, T) patch scores -> (B, gz, gy, gx), for mapping attributions back to voxels."""
        B, T = tokens.shape[0], tokens.shape[1]
        expected = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        if T == expected + 1:  # CLS still attached
            tokens = tokens[:, 1:]
        elif T != expected:
            raise ValueError(f"expected {expected} (+1) tokens, got {T}")
        return tokens.reshape(B, *self.grid_size)

    # ---- forward ---------------------------------------------------------
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x, _ = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits -- loss is BCEWithLogits, so no sigmoid here."""
        return self.head(self.forward_features(x)[:, 0])


def build_vit3d(cfg, img_size: int = 128) -> ViT3D:
    return ViT3D(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        img_size=img_size,
        stem_channels=cfg.stem_channels,
        embed_dim=cfg.embed_dim,
        patch_size=cfg.patch_size,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
        attn_dropout=cfg.attn_dropout,
        drop_path=cfg.drop_path,
    )
