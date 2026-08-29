"""Faithfulness metrics against hand-checkable cases."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.vit3d import ViT3D
from src.xai import build_method
from src.xai.faithfulness import (
    agreement_matrix,
    bone_mask,
    bone_mass_fraction,
    deletion_insertion,
    model_randomization_check,
    randomize_progressively,
    spearman,
    ssim3d,
    topk_jaccard,
)

IMG = 32


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = ViT3D(img_size=IMG, stem_channels=8, embed_dim=32, patch_size=4, depth=2,
              num_heads=4, drop_path=0.0, num_classes=6)
    m.eval()
    return m


@pytest.fixture(scope="module")
def volume():
    torch.manual_seed(1)
    return torch.randn(1, 1, IMG, IMG, IMG)


# --------------------------------------------------------------------------
# rank statistics
# --------------------------------------------------------------------------
def test_spearman_perfect_and_inverse():
    x = np.arange(100, dtype=float)
    assert spearman(x, x) == pytest.approx(1.0, abs=1e-6)
    assert spearman(x, -x) == pytest.approx(-1.0, abs=1e-6)


def test_spearman_is_rank_based_not_linear():
    x = np.arange(1, 50, dtype=float)
    assert spearman(x, x**3) == pytest.approx(1.0, abs=1e-6)


def test_spearman_rejects_size_mismatch():
    with pytest.raises(ValueError, match="size mismatch"):
        spearman(np.zeros(10), np.zeros(11))


def test_topk_jaccard_identical_and_disjoint():
    a = np.arange(1000, dtype=float)
    assert topk_jaccard(a, a, 0.1) == pytest.approx(1.0)
    assert topk_jaccard(a, -a, 0.1) == pytest.approx(0.0)


def test_ssim_identity_is_one():
    torch.manual_seed(0)
    x = torch.rand(16, 16, 16)
    assert ssim3d(x, x) == pytest.approx(1.0, abs=1e-3)


def test_ssim_falls_for_unrelated_volumes():
    torch.manual_seed(0)
    a, b = torch.rand(16, 16, 16), torch.rand(16, 16, 16)
    assert ssim3d(a, b) < 0.5


# --------------------------------------------------------------------------
# deletion / insertion
# --------------------------------------------------------------------------
def test_deletion_insertion_shapes_and_ranges(model, volume):
    saliency = torch.rand(IMG, IMG, IMG)
    out = deletion_insertion(model, volume, saliency, 0, steps=10)

    assert len(out["deletion_curve"]) == len(out["insertion_curve"]) == len(out["fractions"])
    assert 0.0 <= out["deletion_auc"] <= 1.0
    assert 0.0 <= out["insertion_auc"] <= 1.0
    assert all(0.0 <= p <= 1.0 for p in out["deletion_curve"])


def test_deletion_starts_at_the_unperturbed_prediction(model, volume):
    saliency = torch.rand(IMG, IMG, IMG)
    out = deletion_insertion(model, volume, saliency, 0, steps=8)
    with torch.no_grad():
        expected = float(torch.sigmoid(model(volume))[0, 0])
    assert out["deletion_curve"][0] == pytest.approx(expected, abs=1e-4)


def test_insertion_ends_at_the_unperturbed_prediction(model, volume):
    saliency = torch.rand(IMG, IMG, IMG)
    out = deletion_insertion(model, volume, saliency, 0, steps=8)
    with torch.no_grad():
        expected = float(torch.sigmoid(model(volume))[0, 0])
    assert out["insertion_curve"][-1] == pytest.approx(expected, abs=1e-4)


# --------------------------------------------------------------------------
# model randomisation
# --------------------------------------------------------------------------
def test_randomization_order_runs_output_end_to_input_end(model):
    """Head first, conv stem last. The cascade has to reach the input end --
    see test_randomization_covers_every_parameter for why."""
    stages = randomize_progressively(model)
    assert stages[0] == "head"
    assert stages.index(f"blocks.{len(model.blocks) - 1}") < stages.index("blocks.0")
    assert stages.index("blocks.0") < stages.index("stem")


def test_randomization_check_produces_a_row_per_stage(model, volume):
    method = build_method("gradcam", model, torch.device("cpu"))
    intact = method.attribute(volume, 0)

    rows = model_randomization_check(
        model, lambda m: build_method("gradcam", m, torch.device("cpu")), volume, 0, intact
    )
    assert len(rows) == len(randomize_progressively(model))
    for row in rows:
        if "spearman_vs_intact" not in row:
            continue
        rho = row["spearman_vs_intact"]
        # nan is a legitimate result: a fully destroyed model can return a flat
        # map, and correlation with a constant is undefined, not zero.
        assert np.isnan(rho) or -1.0 <= rho <= 1.0
        if np.isnan(rho):
            assert row["degraded_is_constant"] is True
        assert -1.0 <= row["ssim_vs_intact"] <= 1.0


def test_randomization_does_not_mutate_the_original_model(model, volume):
    before = {k: v.clone() for k, v in model.state_dict().items()}
    method = build_method("gradcam", model, torch.device("cpu"))
    intact = method.attribute(volume, 0)
    model_randomization_check(
        model, lambda m: build_method("gradcam", m, torch.device("cpu")), volume, 0, intact
    )
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key]), f"{key} was mutated by the sanity check"


# --------------------------------------------------------------------------
# agreement and the bone proxy
# --------------------------------------------------------------------------
def test_agreement_matrix_is_self_consistent():
    torch.manual_seed(0)
    maps = {"a": torch.rand(8, 8, 8), "b": torch.rand(8, 8, 8)}
    out = agreement_matrix(maps)

    assert out["spearman"]["a|a"] == pytest.approx(1.0, abs=1e-6)
    assert "a|b" in out["spearman"]
    assert "top1pct" in out["jaccard"] and "top5pct" in out["jaccard"]


def test_bone_mask_selects_the_bright_tail(volume):
    mask = bone_mask(volume, percentile=95.0)
    assert mask.shape == (IMG, IMG, IMG)
    fraction = float(mask.float().mean())
    assert 0.01 < fraction < 0.12, f"expected roughly the top 5%, got {fraction:.3f}"


def test_bone_mass_fraction_detects_concentration():
    saliency = torch.zeros(8, 8, 8)
    mask = torch.zeros(8, 8, 8, dtype=torch.bool)
    mask[:2] = True
    saliency[:2] = 1.0  # all mass inside the mask

    out = bone_mass_fraction(saliency, mask)
    assert out["bone_mass_fraction"] == pytest.approx(1.0)
    assert out["enrichment"] == pytest.approx(4.0)  # 1.0 / 0.25


def test_bone_mass_fraction_handles_empty_saliency():
    out = bone_mass_fraction(torch.zeros(8, 8, 8), torch.ones(8, 8, 8, dtype=torch.bool))
    assert np.isnan(out["bone_mass_fraction"])


# ---- the randomisation cascade must reach the input end ---------------------

def test_randomization_covers_every_parameter(model):
    """An earlier version stopped after the transformer blocks, leaving the conv
    stem and patch embedding trained -- 51.8% of the real model untouched, and
    exactly the layers that give gradient methods their edge structure. That did
    not weaken the test, it biased it: rollout reads attention inside the blocks
    (destroyed) while IG backpropagates through the stem (spared), so the two
    were never judged on equal terms."""
    import torch.nn as nn

    from src.xai.faithfulness import randomize_progressively

    covered = set()
    for stage in randomize_progressively(model):
        module = model
        for part in stage.split("."):
            module = module[int(part)] if part.isdigit() else getattr(module, part)
        if isinstance(module, nn.Parameter):
            covered.add(id(module))
        else:
            covered |= {id(p) for p in module.parameters()}

    missed = [n for n, p in model.named_parameters() if id(p) not in covered]
    assert not missed, f"never randomised: {missed}"


def test_randomization_reaches_the_stem_last(model):
    """Output end first, input end last -- the standard cascade order."""
    from src.xai.faithfulness import randomize_progressively

    stages = randomize_progressively(model)
    assert stages[0] == "head"
    assert "stem" in stages
    assert stages.index("stem") > stages.index("blocks.0")


# ---- randomisation must RE-INITIALISE, never nullify -------------------------

def test_reinitialize_does_not_zero_a_layernorm_gain():
    """The bug this guards cost a full 73-minute run and produced a conclusion
    that was exactly backwards.

    An earlier version zeroed every 1-D parameter. LayerNorm's gain is 1-D, so
    it went to zero -- and a zeroed gain makes the block emit exactly 0,
    collapsing the whole network into a constant function. Every gradient-based
    attribution then returned exactly 0.0 for the rest of the cascade, which the
    summary read as "the map barely changed". It had not changed because there
    was nothing left to attribute.
    """
    import torch.nn as nn

    from src.xai.faithfulness import reinitialize

    norm = nn.LayerNorm(8)
    with torch.no_grad():
        norm.weight.fill_(3.0)
    reinitialize(norm, torch.Generator().manual_seed(0))

    assert norm.weight.abs().sum() > 0, "LayerNorm gain was nullified, not re-initialised"
    assert torch.allclose(norm.weight, torch.ones(8)), "LayerNorm resets to gain 1"


def test_reinitialize_actually_changes_trained_weights(model):
    """Re-initialising must still DESTROY what was learned -- the check is
    worthless if `reset_parameters` quietly leaves a module as it was."""
    import copy

    from src.xai.faithfulness import reinitialize

    scrambled = copy.deepcopy(model)
    before = scrambled.head.weight.detach().clone()
    reinitialize(scrambled.head, torch.Generator().manual_seed(0))
    assert not torch.allclose(before, scrambled.head.weight)


def test_cascade_leaves_the_network_input_dependent_at_every_stage(model, volume):
    """The real requirement: after each stage the degraded model must still be a
    FUNCTION OF ITS INPUT, so the saliency it produces is a measurement rather
    than a division by nothing. Two different inputs must give different logits.
    """
    import copy

    from src.xai.faithfulness import randomize_progressively, reinitialize

    scrambled = copy.deepcopy(model)
    generator = torch.Generator().manual_seed(0)
    other = torch.randn_like(volume)

    for stage in randomize_progressively(scrambled):
        module = scrambled
        for part in stage.split("."):
            module = module[int(part)] if part.isdigit() else getattr(module, part)
        reinitialize(module, generator)

        with torch.no_grad():
            a, b = scrambled(volume), scrambled(other)
        assert torch.isfinite(a).all(), f"{stage}: logits went non-finite"
        assert not torch.allclose(a, b), (
            f"{stage}: the degraded model returns the same logits for different "
            "inputs -- it is a constant function, so attribution is undefined"
        )


def test_reinitialize_is_reproducible_for_a_given_seed(model):
    """`reset_parameters` draws from the GLOBAL rng, so without seeding it from
    the passed generator the cascade would differ run to run."""
    import copy

    from src.xai.faithfulness import reinitialize

    outs = []
    for _ in range(2):
        scrambled = copy.deepcopy(model)
        reinitialize(scrambled.blocks[0], torch.Generator().manual_seed(1234))
        outs.append(torch.cat([p.detach().reshape(-1) for p in scrambled.blocks[0].parameters()]))
    assert torch.allclose(outs[0], outs[1])


# --------------------------------------------------------------------------
# what the curve is a curve OF
# --------------------------------------------------------------------------
def test_a_millimetre_head_is_not_read_through_a_sigmoid(model, volume):
    """Deletion/insertion on a millimetre target must not squash the response.

    The explained target on the hybrid task is `available_height_mm`, whose
    head emits a standardised length, not a logit. Sigmoid is monotone, so it
    never reverses a single curve -- but an AUC is an integral, and the
    integral of a NONLINEAR monotone transform can rank two methods the
    opposite way round from the untransformed integral. Un-standardising to
    millimetres is affine and provably cannot, which is why the raw output is
    the right thing to integrate.
    """
    torch.manual_seed(3)
    saliency = torch.rand(volume.shape[2:])

    squashed = deletion_insertion(model, volume, saliency, target_label=0, steps=8)
    raw = deletion_insertion(model, volume, saliency, target_label=0, steps=8,
                             target_is_probability=False)

    assert all(0.0 <= v <= 1.0 for v in squashed["deletion_curve"])
    assert raw["deletion_curve"] != squashed["deletion_curve"]
    # The raw curve is the model's own output; sigmoid of it is the squashed one.
    assert np.allclose(1.0 / (1.0 + np.exp(-np.array(raw["deletion_curve"]))),
                       np.array(squashed["deletion_curve"]), atol=1e-5)


def test_the_default_still_reads_a_binary_head_as_a_probability(model, volume):
    torch.manual_seed(4)
    saliency = torch.rand(volume.shape[2:])
    result = deletion_insertion(model, volume, saliency, target_label=1, steps=8)
    assert 0.0 <= result["deletion_auc"] <= 1.0
    assert 0.0 <= result["insertion_auc"] <= 1.0


def test_deviation_scores_distance_from_the_full_input_prediction(model, volume):
    """The reading a millimetre head needs.

    Deletion/insertion assumes a score that rises with evidence for the target.
    A length is not that: deleting voxels moves a millimetre prediction toward
    whatever the baseline implies, which may be larger or smaller, so the raw
    response has no reason to fall. `score="deviation"` integrates minus the
    distance from the full-input prediction instead, which restores the metric's
    own semantics -- it starts at 0 and falls as the prediction is pulled away.
    """
    torch.manual_seed(5)
    saliency = torch.rand(volume.shape[2:])
    raw = deletion_insertion(model, volume, saliency, target_label=0, steps=8,
                             target_is_probability=False)
    dev = deletion_insertion(model, volume, saliency, target_label=0, steps=8,
                             target_is_probability=False, score="deviation")

    curve = np.array(dev["deletion_curve"])
    assert (curve <= 1e-5).all(), "deviation is a negated absolute value; it cannot be positive"
    with torch.no_grad():
        reference = float(model(volume)[0, 0])
    assert np.allclose(curve, -np.abs(np.array(raw["deletion_curve"]) - reference), atol=1e-5)
    # Nothing deleted yet: the prediction is the full-input one, so distance 0.
    assert curve[0] == pytest.approx(0.0, abs=1e-5)


def test_an_unknown_score_is_refused(model, volume):
    torch.manual_seed(6)
    with pytest.raises(ValueError, match="score must be"):
        deletion_insertion(model, volume, torch.rand(volume.shape[2:]),
                           target_label=0, steps=4, score="absolute")


def test_a_blur_baseline_leaves_a_crest_to_canal_distance_readable():
    """Why the baseline is the prime suspect in the flat deletion/insertion result.

    The default baseline is a blur at sigma=4 voxels, which at 0.3 mm is 1.2 mm.
    The explained head is `available_height_mm`, and available height in the
    mandible IS crest-to-canal distance. This builds that geometry -- a bright
    slab of bone with a dark tube inside it -- and checks what each baseline
    does to the measurement, not to the pixels.

    The blur is supposed to be a "deleted" input. If the distance survives it,
    then deleting the most salient voxels hands the model something that still
    answers the question, the response barely moves, and every method scores the
    same constant. That is exactly the shape the fold-0 run reported.
    """
    from src.xai.base import make_baseline

    n = 96
    vol = torch.zeros(1, 1, n, n, n)
    vol[..., 20:80] = 1.0          # bone slab: crest at z = 20
    vol[..., 48:56] = -1.0         # canal: a dark band, roof at z = 48

    def crest_to_canal(v: torch.Tensor) -> int:
        """Top of the bright block down to the darkest point below it."""
        profile = v[0, 0].mean(dim=(0, 1))
        crest = int((profile > 0.5 * float(profile.max())).nonzero()[0])
        canal = int(profile[crest:].argmin()) + crest
        return canal - crest

    sharp = crest_to_canal(vol)
    assert sharp == 28

    # The tolerance is sigma, not a round number: a Gaussian of sigma=4 voxels
    # can move a landmark by about that much and no more. Surviving to within
    # one sigma is the claim -- the blur perturbs the measurement, it does not
    # remove it.
    blurred = crest_to_canal(make_baseline(vol, "blur"))
    assert abs(blurred - sharp) <= 4, (
        f"crest-to-canal went {sharp} -> {blurred} voxels under the blur "
        f"baseline. It was expected to survive to within one sigma; if it no "
        f"longer does, the blur is harsher than the argument in "
        f"deletion_insertion assumes and that argument needs revisiting"
    )

    flat = make_baseline(vol, "mean", mean_volume=torch.full_like(vol, float(vol.mean())))
    assert float(flat.std()) == pytest.approx(0.0, abs=1e-6), (
        "the mean baseline must carry no geometry at all -- that is the point "
        "of re-running the diagnosis with it"
    )
