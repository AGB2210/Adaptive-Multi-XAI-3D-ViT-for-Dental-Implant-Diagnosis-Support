"""Model contract tests: output shapes, attention shapes, hooks, determinism.

The attention and hook tests exist because the XAI stack depends entirely on
them. They are checked now so that phase does not discover a missing interface.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.cnn3d import CNN3D
from src.models.prevalence import PrevalenceBaseline
from src.models.vit3d import ConvStem, PatchEmbed3d, ViT3D

IMG = 32


def small_vit(**kwargs) -> ViT3D:
    defaults = dict(
        img_size=IMG, stem_channels=8, embed_dim=32, patch_size=4, depth=2, num_heads=4, drop_path=0.0
    )
    defaults.update(kwargs)
    return ViT3D(**defaults)


def test_conv_stem_halves_resolution():
    stem = ConvStem(1, 8)
    out = stem(torch.randn(2, 1, IMG, IMG, IMG))
    assert out.shape == (2, 8, IMG // 2, IMG // 2, IMG // 2)


def test_patch_embed_token_count():
    embed = PatchEmbed3d(8, 32, patch_size=4)
    tokens, grid = embed(torch.randn(2, 8, 16, 16, 16))
    assert grid == (4, 4, 4)
    assert tokens.shape == (2, 4**3, 32)


def test_forward_output_shape():
    model = small_vit()
    logits = model(torch.randn(2, 1, IMG, IMG, IMG))
    assert logits.shape == (2, 6)
    # Raw logits, not probabilities -- loss is BCEWithLogits.
    assert logits.dtype == torch.float32


def test_token_count_follows_the_stem_stride_and_the_patch_size():
    """One token spans 2 * patch_size INPUT voxels, because the stem has stride 2.

    Both configurations, because getting this wrong is what put a token at
    4.8 mm -- wider than the canal the explanations exist to resolve -- while
    the docs said 2.4 mm.
    """
    # The superseded whole-volume setting: 128 -> stem 64 -> patch 8 -> 512.
    old = ViT3D(img_size=128, patch_size=8, embed_dim=32, depth=1, num_heads=4, stem_channels=8)
    assert old.grid_size == (8, 8, 8)
    assert old.pos_embed.shape == (1, 512 + 1, 32)

    # The site setting actually in use: 96 -> stem 48 -> patch 4 -> 1,728.
    site = ViT3D(img_size=96, patch_size=4, embed_dim=32, depth=1, num_heads=4, stem_channels=8)
    assert site.grid_size == (12, 12, 12)
    assert site.pos_embed.shape == (1, 12**3 + 1, 32)
    assert 96 / 12 * 0.3 == pytest.approx(2.4), "a token must span 2.4 mm at 0.3 mm"


def test_attention_weights_shape_and_normalisation():
    model = small_vit(depth=3)
    model.set_store_attention(True)
    model.eval()
    with torch.no_grad():
        model(torch.randn(2, 1, IMG, IMG, IMG))

    maps = model.get_attention_maps()
    assert len(maps) == 3
    n_tokens = 4**3 + 1  # patches + CLS
    for attn in maps:
        assert attn.shape == (2, 4, n_tokens, n_tokens)
        assert torch.allclose(attn.sum(-1), torch.ones_like(attn.sum(-1)), atol=1e-4)


def test_attention_is_not_stored_unless_requested():
    model = small_vit()
    model.eval()
    with torch.no_grad():
        model(torch.randn(1, 1, IMG, IMG, IMG))
    assert model.get_attention_maps() == []

    model.set_store_attention(True)
    with torch.no_grad():
        model(torch.randn(1, 1, IMG, IMG, IMG))
    assert len(model.get_attention_maps()) == 2

    model.set_store_attention(False)
    with torch.no_grad():
        model(torch.randn(1, 1, IMG, IMG, IMG))
    assert model.get_attention_maps() == []


def test_stored_and_fused_attention_agree():
    """The fast path and the explicit path must compute the same function."""
    model = small_vit()
    model.eval()
    x = torch.randn(2, 1, IMG, IMG, IMG)
    with torch.no_grad():
        fused = model(x)
        model.set_store_attention(True)
        explicit = model(x)
    assert torch.allclose(fused, explicit, atol=1e-4)


def test_activation_cache_and_gradients():
    model = small_vit(depth=2)
    model.set_cache_activations(True)
    logits = model(torch.randn(1, 1, IMG, IMG, IMG))
    logits[0, 0].backward()

    acts = model.get_activations()
    assert len(acts) == 2
    assert all(a.grad is not None for a in acts), "activations must retain grad for attribution"


def test_forward_and_backward_hooks_fire():
    """External hooks on intermediate modules -- the generic XAI entry point."""
    model = small_vit()
    seen = {}

    h1 = model.blocks[0].attn.register_forward_hook(lambda m, i, o: seen.__setitem__("fwd", o.shape))
    h2 = model.blocks[0].attn.register_full_backward_hook(
        lambda m, gi, go: seen.__setitem__("bwd", go[0].shape)
    )
    model(torch.randn(1, 1, IMG, IMG, IMG)).sum().backward()
    h1.remove()
    h2.remove()

    assert "fwd" in seen and "bwd" in seen


def test_tokens_to_grid_roundtrip():
    model = small_vit()
    scores = torch.arange(4**3, dtype=torch.float32).unsqueeze(0)
    grid = model.tokens_to_grid(scores)
    assert grid.shape == (1, 4, 4, 4)
    # With CLS still attached it must also work.
    with_cls = torch.cat([torch.zeros(1, 1), scores], dim=1)
    assert torch.equal(model.tokens_to_grid(with_cls), grid)


def test_tokens_to_grid_rejects_wrong_length():
    model = small_vit()
    with pytest.raises(ValueError, match="tokens"):
        model.tokens_to_grid(torch.zeros(1, 17))


def test_deterministic_under_fixed_seed():
    def run():
        torch.manual_seed(0)
        model = small_vit()
        model.eval()
        torch.manual_seed(1)
        x = torch.randn(2, 1, IMG, IMG, IMG)
        with torch.no_grad():
            return model(x)

    assert torch.equal(run(), run())


def test_embed_dim_must_divide_heads():
    with pytest.raises(ValueError, match="divisible"):
        small_vit(embed_dim=30, num_heads=4)


def test_cnn_baseline_shapes_and_features():
    model = CNN3D(num_classes=6)
    model.cache_activations = True
    logits = model(torch.randn(2, 1, IMG, IMG, IMG))
    assert logits.shape == (2, 6)
    assert model.features is not None and model.features.ndim == 5


def test_prevalence_baseline():
    y = np.array([[1, 0], [1, 1], [0, 1], [1, 1]], dtype=float)
    baseline = PrevalenceBaseline(2).fit(y)
    probs = baseline.predict_proba(3)
    assert probs.shape == (3, 2)
    assert np.allclose(probs[0], [0.75, 0.75])
