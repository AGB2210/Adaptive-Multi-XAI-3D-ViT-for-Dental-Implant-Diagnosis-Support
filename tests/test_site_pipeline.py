"""End-to-end: labels -> cache -> cases -> attribution -> localisation.

Every unit in this pipeline had passing tests while the pipeline itself was
broken. The XAI scripts failed at their first file read on the site task, the
synthetic gate produced six targets for a two-output model, checkpoints were
written into another task's directory, and `feasible=False` was used for sites
that could not be measured at all. None of those are unit-level faults; they
are seams. This file tests the seams.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.site_dataset import SitePatchDataset, load_sites, target_matrix
from src.models.vit3d import ViT3D
from src.xai import build_method
from src.xai.localization import enrichment, localization_scores
from src.xai.runner import CaseSet, site_case_ids

PATCH = 16


@pytest.fixture
def world(tmp_path):
    """Two scans, a cache, and a sites CSV that agree with one another."""
    cache = tmp_path / "cache"
    cache.mkdir()
    rows = []
    for pid in ("P1", "P2"):
        vol = np.random.default_rng(abs(hash(pid)) % 1000).normal(size=(48, 48, 48))
        vol[22:26, 22:26, 22:26] += 6.0            # something to find
        np.save(cache / f"{pid}.npy", vol.astype(np.float32))
        for tooth, feasible, reason in ((36, 0.0, "height_nerve"), (37, 1.0, "ok")):
            rows.append({
                "patient_id": pid, "tooth": tooth, "jaw": "lower",
                "site_method": "teeth", "site_x": 24.0, "site_y": 24.0, "site_z": 24.0,
                "needs_implant": 1.0, "feasible": feasible, "reason": reason,
                "available_height_mm": 8.0 if feasible == 0.0 else 18.0,
                "ridge_width_mm": 9.0,
            })
    # one row that must never reach training
    rows.append({"patient_id": "P1", "tooth": 35, "jaw": "lower", "site_method": "teeth",
                 "site_x": 24.0, "site_y": 24.0, "site_z": 24.0,
                 "needs_implant": 1.0, "feasible": 0.0, "reason": "unmeasurable",
                 "available_height_mm": 8.0, "ridge_width_mm": 9.0})
    csv = tmp_path / "sites.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return cache, csv


# One binary head plus two millimetre heads. `feasible` is no longer a target:
# it is a rule applied to the millimetres at inference, so a threshold revision
# is a re-score rather than five folds of retraining.
TARGETS = ["needs_implant", "available_height_mm", "ridge_width_mm"]


def make_cases(cache, csv):
    sites = load_sites(csv, targets=TARGETS)
    return CaseSet(ids=site_case_ids(sites), y=target_matrix(sites, TARGETS),
                   cache=cache, labels=TARGETS, sites=sites, patch_size=PATCH)


class TestLabelsToCases:
    def test_the_unmeasurable_row_never_reaches_a_case(self, world):
        """It carries feasible=False, not NaN, so only `reason` catches it."""
        cases = make_cases(*world)
        assert "P1#35" not in cases.ids
        assert len(cases.ids) == 4

    def test_every_case_can_load_its_input(self, world):
        cases = make_cases(*world)
        for cid in cases.ids:
            x = cases.load(cid, torch.device("cpu"))
            assert tuple(x.shape) == (1, 1, PATCH, PATCH, PATCH)
            assert torch.isfinite(x).all()

    def test_labels_line_up_with_ids(self, world):
        cases = make_cases(*world)
        by_id = dict(zip(cases.ids, cases.y))
        assert by_id["P1#36"][TARGETS.index("available_height_mm")] == 8.0
        assert by_id["P1#37"][TARGETS.index("available_height_mm")] == 18.0

    def test_the_dataset_and_the_case_set_agree(self, world):
        """Training and explanation must see the same voxels, or an explanation
        describes an input the model never got."""
        cache, csv = world
        sites = load_sites(csv, targets=TARGETS)
        ds = SitePatchDataset(cache, sites, targets=TARGETS, patch_size=PATCH)
        cases = make_cases(cache, csv)
        for i, cid in enumerate(cases.ids):
            a = ds[i][0]
            b = cases.load(cid, torch.device("cpu"))[0]
            assert torch.allclose(a, b), f"{cid}: dataset and CaseSet disagree"


class TestCasesToAttribution:
    def model(self):
        torch.manual_seed(0)
        return ViT3D(img_size=PATCH, stem_channels=4, embed_dim=16, patch_size=2,
                     depth=1, num_heads=2, drop_path=0.0, num_classes=len(TARGETS)).eval()

    @pytest.mark.parametrize("name", ["attention_rollout", "gradcam", "integrated_gradients"])
    def test_every_method_runs_on_a_site_patch(self, world, name):
        cases = make_cases(*world)
        model = self.model()
        for p in model.parameters():
            p.requires_grad_(False)
        method = build_method(name, model, torch.device("cpu"),
                              **({"steps": 4, "batch_size": 1}
                                 if name == "integrated_gradients" else {}))
        volume = cases.load(cases.ids[0], torch.device("cpu"))
        saliency = method.attribute(volume, TARGETS.index("available_height_mm"))
        assert tuple(saliency.shape) == (PATCH, PATCH, PATCH)
        assert torch.isfinite(saliency).all()

    def test_the_token_grid_is_finer_than_the_structure_being_explained(self):
        """The conv stem's stride 2 doubles the effective token size, and this
        was got wrong once: patch_size 8 was documented as 2.4 mm per token and
        is really 4.8 mm -- wider than the 2-3 mm nerve canal. A token must be
        no coarser than what the explanation has to point at."""
        model = ViT3D(img_size=96, stem_channels=8, embed_dim=32, patch_size=4,
                      depth=1, num_heads=2, drop_path=0.0, num_classes=2)
        mm_per_token = (96 / model.grid_size[0]) * 0.3
        assert model.grid_size == (12, 12, 12)
        assert mm_per_token == pytest.approx(2.4)
        assert mm_per_token < 3.0, "a token is wider than the nerve canal"


class TestAttributionToScore:
    def test_a_map_on_the_structure_beats_one_beside_it(self, world):
        """The localisation metric has to be able to tell the two apart, or no
        result it produces means anything."""
        canal = np.zeros((PATCH, PATCH, PATCH), dtype=bool)
        canal[7:9, 7:9, :] = True

        on_target = canal.astype(np.float32)
        elsewhere = np.zeros_like(on_target)
        elsewhere[1:3, 1:3, :] = 1.0

        assert enrichment(on_target, canal) > 10.0
        assert enrichment(elsewhere, canal) == pytest.approx(0.0)

    def test_scores_are_finite_for_a_realistic_diffuse_map(self, world):
        canal = np.zeros((PATCH, PATCH, PATCH), dtype=bool)
        canal[7:9, 7:9, :] = True
        rng = np.random.default_rng(0)
        out = localization_scores(rng.random((PATCH, PATCH, PATCH)).astype(np.float32), canal)
        for key in ("enrichment", "mass_inside", "iou", "dice"):
            assert np.isfinite(out[key])


class TestConfigsAgreeWithTheCode:
    def test_the_site_config_resolves_end_to_end(self):
        """Catches the class of bug where a config overrides one path and
        inherits another that no longer matches it."""
        from src.data.taskdef import all_target_names
        from src.utils.config import load_config
        from src.xai.runner import model_img_size

        cfg = load_config("configs/sites.yaml")
        assert model_img_size(cfg) == 96
        assert cfg.preprocess.out_shape is None      # nothing is resampled
        assert str(cfg.train.out_dir).startswith(str(cfg.data.artifacts_dir))
        assert getattr(cfg.task, "sites_csv", None)
        assert tuple(all_target_names(cfg)) == tuple(TARGETS)

    def test_the_site_model_builds_at_the_configured_size(self):
        from src.data.taskdef import all_target_names
        from src.models import build_model
        from src.utils.config import load_config
        from src.xai.runner import model_img_size

        cfg = load_config("configs/sites.yaml")
        model = build_model(cfg.model, img_size=model_img_size(cfg))
        out = model(torch.zeros(2, 1, 96, 96, 96))
        assert tuple(out.shape) == (2, len(all_target_names(cfg)))
