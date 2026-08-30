"""The XAI stage's view of a case, on both tasks.

Before this existed the XAI scripts assumed a case was a whole scan, so on the
site task they failed at the first file read -- after a GPU had been paid for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.xai.runner import SITE_SEP, CaseSet, site_case_ids

SPACING = (0.3, 0.3, 0.3)


@pytest.fixture
def cache(tmp_path):
    d = tmp_path / "cache"
    d.mkdir()
    for pid in ("P1", "P2"):
        vol = np.zeros((60, 60, 60), dtype=np.float32)
        vol[30, 30, 30] = 7.0            # a marker at the site centre
        vol[10, 10, 10] = -3.0           # something elsewhere, to catch bad crops
        np.save(d / f"{pid}.npy", vol)
    return d


def frame(rows):
    return pd.DataFrame(rows)


SITE = dict(site_x=30.0, site_y=30.0, site_z=30.0, jaw="lower", site_method="teeth")


class TestWholeVolumeTask:
    def test_load_returns_the_whole_volume(self, cache):
        cases = CaseSet(ids=["P1"], y=np.ones((1, 2), np.float32), cache=cache,
                        labels=["a", "b"])
        assert not cases.is_sites
        assert tuple(cases.load("P1", torch.device("cpu")).shape) == (1, 1, 60, 60, 60)

    def test_patient_of_is_the_id_itself(self, cache):
        cases = CaseSet(ids=["P1"], y=np.zeros((1, 2), np.float32), cache=cache, labels=["a", "b"])
        assert cases.patient_of("P1") == "P1"

    def test_there_is_no_site_row(self, cache):
        cases = CaseSet(ids=["P1"], y=np.zeros((1, 2), np.float32), cache=cache, labels=["a", "b"])
        assert cases.row("P1") is None


class TestSiteTask:
    def build(self, cache, patch=16):
        sites = frame([
            {**SITE, "patient_id": "P1", "tooth": 36, "needs_implant": 1.0, "feasible": 0.0},
            {**SITE, "patient_id": "P2", "tooth": 46, "needs_implant": 1.0, "feasible": 1.0},
        ])
        return CaseSet(
            ids=site_case_ids(sites),
            y=sites[["needs_implant", "feasible"]].to_numpy(np.float32),
            cache=cache, labels=["needs_implant", "feasible"],
            sites=sites, patch_size=patch,
        )

    def test_ids_name_the_patient_and_the_tooth(self, cache):
        cases = self.build(cache)
        assert cases.ids == [f"P1{SITE_SEP}36", f"P2{SITE_SEP}46"]

    def test_load_returns_a_patch_not_the_volume(self, cache):
        cases = self.build(cache, patch=16)
        x = cases.load(f"P1{SITE_SEP}36", torch.device("cpu"))
        assert tuple(x.shape) == (1, 1, 16, 16, 16)

    def test_the_patch_is_pushed_toward_the_structure_that_decides(self, cache):
        """Deliberately NOT centred on the crest. A centred 96^3 box at 0.3 mm
        reached 14.4 mm below the crest against a 12.0 mm threshold, so 26.2% of
        sites needing an implant had their limiting structure outside the input,
        while half the box sat above the crest on air and opposing crowns.

        Quarter-shifted (patch // 4), the crest sits high in the box: here z=30
        with patch 16 shifts the centre to 26, so the crest lands at index 12 of
        16 rather than 8."""
        cases = self.build(cache, patch=16)
        x = cases.load(f"P1{SITE_SEP}36", torch.device("cpu"))
        assert x[0, 0, 8, 8, 12].item() == pytest.approx(7.0), "crest not where the shift puts it"
        assert x[0, 0, 8, 8, 8].item() != pytest.approx(7.0), "patch is still centred"

    def test_x_and_y_are_still_centred(self, cache):
        """Only z moves. Shifting in-plane would centre the box on different
        anatomy than the label describes."""
        cases = self.build(cache, patch=16)
        x = cases.load(f"P1{SITE_SEP}36", torch.device("cpu"))
        col = x[0, 0, :, :, 12]
        assert col[8, 8].item() == pytest.approx(7.0)

    def test_patient_of_strips_the_tooth(self, cache):
        cases = self.build(cache)
        assert cases.patient_of(f"P1{SITE_SEP}36") == "P1"

    def test_the_site_row_is_recoverable(self, cache):
        """run_localization needs it to crop the anatomy mask identically."""
        cases = self.build(cache)
        row = cases.row(f"P2{SITE_SEP}46")
        assert row["patient_id"] == "P2" and row["tooth"] == 46

    def test_an_unknown_case_raises_rather_than_returning_noise(self, cache):
        cases = self.build(cache)
        with pytest.raises(KeyError, match="NOPE"):
            cases.load("NOPE#1", torch.device("cpu"))

    def test_volumes_are_memory_mapped(self, cache):
        """The native-resolution cache is 26.4 GB over 522 scans."""
        cases = self.build(cache)
        cases.load(f"P1{SITE_SEP}36", torch.device("cpu"))
        assert isinstance(cases._volume("P1"), np.memmap)

    def test_two_sites_on_one_scan_reuse_one_open_handle(self, cache):
        sites = frame([
            {**SITE, "patient_id": "P1", "tooth": 36, "needs_implant": 1.0, "feasible": 0.0},
            {**SITE, "patient_id": "P1", "tooth": 37, "needs_implant": 1.0, "feasible": 1.0},
        ])
        cases = CaseSet(ids=site_case_ids(sites),
                        y=sites[["needs_implant", "feasible"]].to_numpy(np.float32),
                        cache=cache, labels=["needs_implant", "feasible"],
                        sites=sites, patch_size=16)
        for i in cases.ids:
            cases.load(i, torch.device("cpu"))
        assert list(cases._volumes) == ["P1"]

    def test_a_site_at_the_edge_is_padded_not_shifted(self, cache):
        sites = frame([{**SITE, "patient_id": "P1", "tooth": 31, "site_x": 1.0,
                        "needs_implant": 1.0, "feasible": 0.0}])
        cases = CaseSet(ids=site_case_ids(sites),
                        y=sites[["needs_implant", "feasible"]].to_numpy(np.float32),
                        cache=cache, labels=["needs_implant", "feasible"],
                        sites=sites, patch_size=16)
        x = cases.load(f"P1{SITE_SEP}31", torch.device("cpu"))
        assert tuple(x.shape) == (1, 1, 16, 16, 16)
