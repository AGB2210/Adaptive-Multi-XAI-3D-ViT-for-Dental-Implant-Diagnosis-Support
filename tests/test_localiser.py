"""The site localiser: its grid, its geometry, and its seam with the app.

A localiser position becomes a patch through `patch_centre`, exactly as a
mask-derived one does, so every coordinate transform here has to be exact --
a grid that is off by half a block moves every patch by 0.6 mm and nothing
downstream would say so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.dental_arch import LOWER_ARCH
from src.data.localiser_dataset import LocaliserDataset, site_targets
from src.models.localiser import (
    Localiser,
    LocaliserConfig,
    SiteLocaliser,
    load_localiser,
    localiser_input,
    localiser_loss,
    soft_argmax,
    to_full_coords,
    to_input_coords,
)
from tests.toy_scan import SMALL_LOCALISER as SMALL
from tests.toy_scan import write_tiny_localiser

CPU = torch.device("cpu")


class TestTheGrid:
    def test_full_and_grid_coordinates_invert_each_other(self):
        full = np.array([[123.4, 56.7, 89.0], [0.0, 0.0, 0.0], [511.0, 400.0, 297.0]])
        for offset in ([0, 0, 0], [5, -3, 12]):
            back = to_full_coords(to_input_coords(full, offset, 4), offset, 4)
            np.testing.assert_allclose(back, full, atol=1e-9)

    def test_a_voxel_lands_in_its_own_block(self):
        """Full voxel 4k..4k+3 is block k; its centre (4k + 1.5) maps to exactly k."""
        np.testing.assert_allclose(to_input_coords(np.array([9.5, 1.5, 5.5]), [0, 0, 0], 4),
                                   [2.0, 0.0, 1.0])

    def test_a_bright_voxel_is_found_where_the_mapping_says(self):
        vol = np.zeros((60, 52, 40), dtype=np.float32)
        vol[37, 18, 22] = 64.0                      # one voxel, block (9, 4, 5)
        grid, offset = localiser_input(vol, 4, (32, 32, 24))
        peak = np.array(np.unravel_index(np.argmax(grid), grid.shape))
        expected = to_input_coords(np.array([37, 18, 22]), offset, 4)
        assert np.all(np.abs(peak - expected) <= 0.5 + 1e-9)

    def test_a_larger_scan_is_cropped_and_says_so(self):
        _, offset = localiser_input(np.zeros((200, 40, 40), np.float32), 4, (32, 32, 24))
        assert offset[0] < 0 and offset[1] >= 0

    def test_the_cache_builder_and_the_app_share_the_grid_function(self):
        import scripts.build_localiser_cache as build
        assert build.localiser_input is localiser_input


class TestTheNetwork:
    def test_soft_argmax_finds_a_single_peak(self):
        heat = torch.full((1, 2, 8, 9, 10), -50.0)
        heat[0, 0, 3, 7, 2] = 50.0
        heat[0, 1, 6, 1, 9] = 50.0
        coords, spread = soft_argmax(heat)
        torch.testing.assert_close(coords[0], torch.tensor([[3.0, 7.0, 2.0], [6.0, 1.0, 9.0]]))
        assert float(spread.max()) < 1e-3

    def test_a_flat_heatmap_reports_a_wide_spread(self):
        _, spread = soft_argmax(torch.zeros(1, 1, 8, 8, 8))
        assert float(spread) > 3.0, "an undecided heatmap must not look certain"

    def test_outputs_have_one_row_per_site(self):
        model = SiteLocaliser(LocaliserConfig(**SMALL)).eval()
        out = model(torch.randn(2, 1, 32, 32, 24))
        assert out["coords"].shape == (2, 14, 3)
        assert out["spread"].shape == out["valid_logit"].shape == (2, 14)
        assert out["flip_logit"].shape == (2,)

    def test_a_grid_the_network_cannot_halve_is_refused(self):
        with pytest.raises(ValueError, match="divisible"):
            LocaliserConfig(factor=4, input_shape=(30, 32, 24), channels=(4, 8, 8, 8))

    def test_an_unmeasured_crest_trains_x_and_y_only(self):
        out = {"coords": torch.zeros(1, 1, 3, requires_grad=True), "valid_logit": torch.zeros(1, 1),
               "flip_logit": torch.zeros(1), "spread": torch.zeros(1, 1)}
        target = torch.tensor([[[1.0, 1.0, 50.0]]])
        mask = torch.tensor([[[True, True, False]]])
        loss = localiser_loss(out, target, mask, torch.ones(1, 1), torch.zeros(1), 1.2)
        loss["total"].backward()
        assert out["coords"].grad[0, 0, 2] == 0, "a NaN crest must not pull z anywhere"
        assert out["coords"].grad[0, 0, 0] != 0


class TestTargets:
    def sites(self):
        rows = []
        for tooth in LOWER_ARCH:
            rows.append({"patient_id": "P", "tooth": tooth, "jaw": "lower", "site_method": "teeth",
                         "site_x": 40.0 + tooth % 10, "site_y": 30.0, "site_z": 20.0, "reason": "ok"})
        rows[0].update(site_z=np.nan)                         # crest not measured
        rows[1].update(reason="outside_volume")               # never in view
        rows[2].update(site_method="sparse")
        return pd.DataFrame(rows)

    def test_masks_follow_what_was_measured(self):
        coords, mask, valid = site_targets(self.sites(), "P", LOWER_ARCH, [0, 0, 0], 4)
        assert mask[0].tolist() == [True, True, False] and valid[0] == 1
        assert not mask[1].any() and valid[1] == 0
        assert mask[3].all()

    def test_a_tier_filter_drops_the_others(self):
        _, mask, valid = site_targets(self.sites(), "P", LOWER_ARCH, [0, 0, 0], 4, methods=("teeth",))
        assert valid[2] == 0 and not mask[2].any()

    def test_a_flipped_sample_still_points_at_its_site(self, tmp_path):
        """The orientation augmentation must move the target with the image."""
        sites = self.sites()
        vol = np.zeros((32, 32, 24), dtype=np.float32)
        coords, mask, _ = site_targets(sites, "P", LOWER_ARCH, [0, 0, 0], 4)
        i = 5
        x, y, z = np.round(coords[i]).astype(int)
        vol[x, y, z] = 9.0
        np.save(tmp_path / "P.npy", vol.astype(np.float16))
        manifest = pd.DataFrame([{"patient_id": "P", "offset_x": 0, "offset_y": 0, "offset_z": 0}])
        ds = LocaliserDataset(tmp_path, manifest, sites, ["P"], LOWER_ARCH, 4,
                              augment=True, flip_prob=1.0, translate=0)
        xb, cb, mb, _, flipped = ds[0]
        assert float(flipped) == 1.0
        px, py, pz = np.round(cb[i].numpy()).astype(int)
        assert float(xb[0, px, py, pz]) > 5.0, "the flipped target no longer sits on its voxel"


class TestItLearns:
    def test_one_scan_is_fitted_far_below_its_starting_error(self):
        """The localiser gate: a site position the network cannot fit on one
        scan it will not find on unseen ones."""
        from src.data.scan import normalise
        from src.data.site_scoring import score_one
        from tests.toy_scan import toy_image, toy_mask

        mask = toy_mask(flipped=False)
        volume, *_ = normalise(toy_image(mask), (-1000.0, 6000.0), -500.0)
        rules = {"min_height_mandible_mm": 12.0, "min_height_maxilla_mm": 10.0, "min_width_mm": 6.0}
        rows = pd.DataFrame([{**r, "patient_id": "P"} for r in score_one(mask, (0.3,) * 3, rules)])
        grid, offset = localiser_input(volume, 4, SMALL["input_shape"])
        coords, cmask, valid = site_targets(rows, "P", LOWER_ARCH, offset, 4)

        torch.manual_seed(0)
        model = SiteLocaliser(LocaliserConfig(**SMALL))
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        x = torch.from_numpy(grid)[None, None]
        t, m = torch.from_numpy(np.nan_to_num(coords)).float()[None], torch.from_numpy(cmask)[None]
        v = torch.from_numpy(valid)[None]
        first = None
        for _ in range(150):
            loss = localiser_loss(model(x), t, m, v, torch.zeros(1), 1.2)
            opt.zero_grad()
            loss["total"].backward()
            opt.step()
            first = first if first is not None else loss["coord_mm"].item()
        assert loss["coord_mm"].item() < first / 3, (first, loss["coord_mm"].item())


class TestLoading:
    def test_a_localiser_checkpoint_loads_and_describes_itself(self, tmp_path):
        loc = load_localiser(write_tiny_localiser(tmp_path / "loc.pt"), CPU)
        d = loc.describe()
        assert d["kind"] == "localiser" and d["sites"] == list(LOWER_ARCH)
        assert d["grid_mm"] == pytest.approx(1.2)

    def test_a_site_model_is_not_a_localiser(self, tmp_path):
        from src.inference.checkpoint import CheckpointError
        from tests.toy_scan import write_tiny_checkpoint
        with pytest.raises(CheckpointError, match="localiser"):
            load_localiser(write_tiny_checkpoint(tmp_path / "site.pt"), CPU)

    def test_locate_returns_sites_in_the_mask_paths_shape(self, tmp_path):
        loc: Localiser = load_localiser(write_tiny_localiser(tmp_path / "loc.pt"), CPU)
        got = loc.locate(np.random.default_rng(0).normal(size=(100, 100, 70)).astype(np.float16),
                         0.3)
        assert [s["tooth"] for s in got["sites"]] == list(LOWER_ARCH)
        for s in got["sites"]:
            assert s["source"] == "localiser" and s["truth"] is None
            assert {"site_x", "site_y", "site_z", "jaw", "method"} <= set(s)
        assert isinstance(got["flip"], bool)
        assert any("localiser" in w for w in got["warnings"])

    def test_a_different_spacing_is_refused(self, tmp_path):
        loc = load_localiser(write_tiny_localiser(tmp_path / "loc.pt"), CPU)
        with pytest.raises(ValueError, match="0.3"):
            loc.locate(np.zeros((40, 40, 40), np.float16), 0.5)
