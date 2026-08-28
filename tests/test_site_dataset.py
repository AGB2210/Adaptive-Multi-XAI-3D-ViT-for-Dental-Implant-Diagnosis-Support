"""Site-level samples: patch cutting, filtering, and the split that must not leak."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.site_dataset import (
    SITE_TARGETS,
    SitePatchDataset,
    cut_patch,
    group_ids,
    load_sites,
    target_matrix,
)


def sites_frame(rows):
    return pd.DataFrame(rows)


# `feasible` is kept in the fixture because the CSV still carries it -- the label
# builder writes it, and the threshold-sensitivity work reads it. It is simply no
# longer a training target; the millimetres are.
BASE = dict(site_method="teeth", jaw="lower", site_x=20.0, site_y=20.0, site_z=20.0,
            needs_implant=1.0, feasible=1.0, reason="ok",
            available_height_mm=18.0, ridge_width_mm=9.0)


@pytest.fixture
def csv(tmp_path):
    df = sites_frame([
        {**BASE, "patient_id": "A", "tooth": 36},
        {**BASE, "patient_id": "A", "tooth": 37, "feasible": 0.0, "available_height_mm": 8.0},
        {**BASE, "patient_id": "B", "tooth": 36, "site_method": "sparse"},
        {**BASE, "patient_id": "B", "tooth": 37, "site_method": "opposite_jaw"},
        {**BASE, "patient_id": "C", "tooth": 36, "feasible": np.nan, "available_height_mm": np.nan},
        {**BASE, "patient_id": "C", "tooth": 37, "site_x": np.nan},
        {**BASE, "patient_id": "D", "tooth": 16, "jaw": "upper"},
        {**BASE, "patient_id": "E", "tooth": 35, "feasible": 0.0,
         "reason": "unmeasurable"},
    ])
    path = tmp_path / "sites.csv"
    df.to_csv(path, index=False)
    return path


class TestCutPatch:
    def test_returns_the_requested_size(self):
        v = np.arange(60 ** 3, dtype=np.float32).reshape(60, 60, 60)
        assert cut_patch(v, (30, 30, 30), 16).shape == (16, 16, 16)

    def test_is_centred_on_the_point(self):
        v = np.zeros((60, 60, 60), dtype=np.float32)
        v[30, 30, 30] = 9.0
        assert cut_patch(v, (30, 30, 30), 16)[8, 8, 8] == 9.0

    def test_pads_at_the_edge_rather_than_sliding(self):
        """A site near the edge of the field of view genuinely has less context.
        Shifting the window would centre it on different anatomy than the label
        describes, and nothing downstream would notice."""
        v = np.ones((60, 60, 60), dtype=np.float32)
        patch = cut_patch(v, (1, 30, 30), 16)
        assert patch.shape == (16, 16, 16)
        assert patch[0, 8, 8] == 0.0     # padded
        assert patch[8, 8, 8] == 1.0     # real voxel still at the centre

    def test_a_box_entirely_outside_is_all_padding(self):
        v = np.ones((60, 60, 60), dtype=np.float32)
        assert cut_patch(v, (-100, 30, 30), 16).sum() == 0.0

    def test_does_not_modify_the_source_volume(self):
        v = np.ones((40, 40, 40), dtype=np.float32)
        cut_patch(v, (20, 20, 20), 8)[:] = 5.0
        assert v.max() == 1.0


class TestLoadSites:
    def test_default_keeps_teeth_only(self, csv):
        df = load_sites(csv)
        assert set(df.site_method) == {"teeth"}
        assert len(df) == 2

    def test_weaker_tiers_can_be_opted_into(self, csv):
        """The edentulous patients are 30.7% of missing sites and only reachable
        through the weaker tiers, so including them must be possible."""
        df = load_sites(csv, methods=("teeth", "sparse", "opposite_jaw"))
        assert set(df.site_method) == {"teeth", "sparse", "opposite_jaw"}

    def test_rows_without_a_position_are_dropped(self, csv):
        df = load_sites(csv, methods=None)
        assert df.site_x.notna().all()

    def test_unmeasurable_rows_are_dropped_not_treated_as_negative(self, csv):
        """Training a nan as 'not feasible' would teach the model to reproduce
        our own measurement failures as clinical findings."""
        df = load_sites(csv, methods=None)
        assert "C" not in set(df.patient_id) or df[df.patient_id == "C"].feasible.notna().all()

    def test_the_maxilla_is_excluded_by_default(self, csv):
        """Measured, not preferred: of the sites that need an implant, 91.8% are
        measurable in the mandible and 4.3% in the maxilla. After an upper tooth
        is lost the ridge resorbs and ToothFairy3's UpperJaw mask does not cover
        the remnant, so 98% of those sites have no bone at all. Training on them
        would reproduce an annotation gap as a clinical verdict."""
        df = load_sites(csv, methods=None)
        assert set(df.jaw) == {"lower"}

    def test_the_maxilla_can_be_opted_back_in(self, csv):
        df = load_sites(csv, methods=None, jaws=("lower", "upper"))
        assert set(df.jaw) == {"lower", "upper"}

    def test_an_unmeasurable_site_is_dropped_even_though_its_target_is_not_nan(self, csv):
        """The trap: feasibility() returns feasible=False when a site cannot be
        measured -- it cannot return True -- so a notna() test sails past it and
        the model learns our measurement failures as clinical verdicts. Only the
        `reason` column tells the two apart."""
        df = load_sites(csv, methods=None)
        assert "E" not in set(df.patient_id)
        assert (df.reason != "unmeasurable").all()

    def test_unmeasurable_rows_survive_when_the_caller_opts_out(self, csv):
        df = load_sites(csv, methods=None, drop_unmeasurable=False)
        assert "E" in set(df.patient_id)

    def test_a_missing_column_is_named(self, tmp_path):
        path = tmp_path / "bad.csv"
        pd.DataFrame([{"patient_id": "A", "tooth": 36}]).to_csv(path, index=False)
        with pytest.raises(ValueError, match="site_x"):
            load_sites(path)


class TestTargets:
    def test_matrix_follows_the_requested_order(self):
        df = sites_frame([{**BASE, "patient_id": "A", "tooth": 36, "feasible": 0.0}])
        assert target_matrix(df, ("needs_implant", "feasible")).tolist() == [[1.0, 0.0]]
        assert target_matrix(df, ("feasible", "needs_implant")).tolist() == [[0.0, 1.0]]


class TestGrouping:
    def test_group_is_the_patient_not_the_site(self):
        """28 sites from one scan share anatomy, field of view and annotator.
        Splitting by site puts a patient's left molars in train and their right
        molars in test, which reads as generalisation and is not."""
        df = sites_frame([
            {**BASE, "patient_id": "A", "tooth": 36},
            {**BASE, "patient_id": "A", "tooth": 37},
            {**BASE, "patient_id": "B", "tooth": 36},
        ])
        assert group_ids(df).tolist() == ["A", "A", "B"]
        assert len(set(group_ids(df))) == 2


class TestSitePatchDataset:
    def build(self, tmp_path, n_patients=2):
        cache = tmp_path / "cache"
        cache.mkdir()
        rows = []
        for p in [chr(ord("A") + i) for i in range(n_patients)]:
            np.save(cache / f"{p}.npy", np.full((60, 60, 60), ord(p), dtype=np.float32))
            rows.append({**BASE, "patient_id": p, "tooth": 36, "site_x": 30.0,
                         "site_y": 30.0, "site_z": 30.0})
        return cache, sites_frame(rows)

    def test_yields_a_patch_and_one_target_per_head(self, tmp_path):
        cache, sites = self.build(tmp_path)
        ds = SitePatchDataset(cache, sites, patch_size=16)
        x, y = ds[0]
        assert tuple(x.shape) == (1, 16, 16, 16)
        assert tuple(y.shape) == (3,)      # 1 binary + 2 millimetre heads

    def test_targets_do_not_share_memory_with_the_frame(self, tmp_path):
        """A sample's labels must be a copy, not a window into `sites`.

        Read this as a guard, not a reproduction. Whether `to_numpy` returns a
        view depends on the pandas build, and on 2.3.3 it always materialises,
        so this test passes with or without the `copy=True` that fixes it. It
        only fails on the builds that have the bug -- which is precisely the
        machine where it needs to fail, since CI hit the torch warning while
        the same test passed here.

        The properties asserted are the ones that matter either way: the array
        must be writable (torch calls in-place ops on a read-only tensor
        undefined behaviour) and must not alias the frame (or one in-place op
        downstream rewrites the labels for every later epoch).
        """
        cache, sites = self.build(tmp_path)
        ds = SitePatchDataset(cache, sites, patch_size=16)

        assert ds.labels.flags.writeable, (
            "torch.from_numpy on a read-only array gives a tensor whose "
            "in-place ops are undefined behaviour"
        )
        assert not np.shares_memory(ds.labels, sites[list(SITE_TARGETS)].to_numpy())

        _, y = ds[0]
        before = float(sites.iloc[0][SITE_TARGETS[1]])
        y[1] = -999.0                        # the thing a collate or a
        after = float(sites.iloc[0][SITE_TARGETS[1]])   # standardisation does
        assert after == before, "writing to a sample rewrote the source frame"

    def test_one_sample_per_site_not_per_scan(self, tmp_path):
        cache, sites = self.build(tmp_path, n_patients=3)
        assert len(SitePatchDataset(cache, sites, patch_size=16)) == 3

    def test_each_sample_reads_its_own_patient(self, tmp_path):
        cache, sites = self.build(tmp_path, n_patients=2)
        ds = SitePatchDataset(cache, sites, patch_size=8)
        assert ds[0][0].mean().item() == float(ord("A"))
        assert ds[1][0].mean().item() == float(ord("B"))

    def test_a_missing_cached_volume_is_named(self, tmp_path):
        cache, sites = self.build(tmp_path)
        sites.loc[0, "patient_id"] = "ZZZ"
        with pytest.raises(FileNotFoundError, match="ZZZ"):
            SitePatchDataset(cache, sites, patch_size=16)

    def test_volumes_are_memory_mapped(self, tmp_path):
        """A 50 GB native-resolution cache must not be pulled into RAM."""
        cache, sites = self.build(tmp_path)
        ds = SitePatchDataset(cache, sites, patch_size=16)
        assert isinstance(ds.volume("A"), np.memmap)
