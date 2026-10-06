"""The inference path the app takes, held to the training path it must equal.

The app runs on scans nobody cached, through checkpoints nobody configured.
Each seam below is one where it could quietly diverge from what the model was
trained and measured on -- and a divergence there does not crash, it produces a
plausible-looking jaw with the wrong numbers on it.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.implant_sites import superior_sign
from src.data.scan import normalise, prepare_volume
from src.inference.checkpoint import (
    CheckpointError,
    check_declared,
    load_bundle,
    read_architecture,
)
from src.inference.predict import predict_sites, score_sites, site_patch
from src.inference.scan import patient_id_from_filename, prepare_scan
from src.inference.sites import sites_from_mask
from src.xai.runner import CaseSet, site_case_ids
from tests.toy_scan import (
    MISSING,
    MODEL_CONFIG,
    NAMES,
    SPACING,
    tiny_model,
    tiny_spec,
    toy_image,
    toy_mask,
    write_tiny_checkpoint,
    write_toy_scan,
)

CPU = torch.device("cpu")
RULES = {"min_height_mandible_mm": 12.0, "min_height_maxilla_mm": 10.0, "min_width_mm": 6.0}
CLIP, AIR = (-1000.0, 6000.0), -500.0


def prepared(tmp_path, flipped=True, **kw):
    image, mask = write_toy_scan(tmp_path, flipped=flipped)
    return prepare_scan(image, mask, CLIP, AIR, 0.3, 0.02, **kw)


# --------------------------------------------------------------------------
# the volume: one transform, shared with the cache build
# --------------------------------------------------------------------------
class TestTheVolumeIsTheCachedVolume:
    def test_the_cache_builder_and_the_app_share_one_function(self):
        """A second copy of the normalisation is exactly what this guards."""
        import scripts.build_site_cache as cache_script
        assert cache_script.prepare_volume is prepare_volume
        assert cache_script.normalise is normalise

    def test_prepare_scan_equals_the_cache_build_by_hand(self, tmp_path):
        """flip by the mask's anatomy, then clip and z-score: nothing else."""
        got = prepared(tmp_path)
        mask = toy_mask(flipped=True)
        sign = superior_sign(mask)
        assert sign == -1, "the toy is stored the ToothFairy3 way up"
        # Stored upside down, flipped back by anatomy: the right-way-up image.
        want, *_ = normalise(toy_image(toy_mask(flipped=False)), CLIP, AIR)
        assert got.sign == -1 and got.sign_source == "mask anatomy"
        assert np.array_equal(got.volume, want)
        assert got.volume.dtype == np.float16

    def test_both_storage_orientations_give_the_same_volume(self, tmp_path):
        a = prepared(tmp_path / "a", flipped=True)
        b = prepared(tmp_path / "b", flipped=False)
        assert a.sign == -1 and b.sign == 1
        assert np.array_equal(a.volume, b.volume)

    def test_a_contradicting_sign_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="anatomy"):
            prepared(tmp_path, sign=1)

    def test_without_a_mask_the_sign_must_be_given(self, tmp_path):
        image, _ = write_toy_scan(tmp_path)
        with pytest.raises(ValueError, match="orientation"):
            prepare_scan(image, None, CLIP, AIR, 0.3, 0.02)
        got = prepare_scan(image, None, CLIP, AIR, 0.3, 0.02, sign=-1, sign_source="default")
        assert got.sign_source == "default" and got.mask is None

    def test_an_off_spacing_scan_is_resampled_and_says_so(self, tmp_path):
        image, mask = write_toy_scan(tmp_path, spacing=(0.4, 0.4, 0.4))
        got = prepare_scan(image, mask, CLIP, AIR, 0.3, 0.02)
        assert got.spacing == (0.3, 0.3, 0.3)
        assert got.volume.shape == tuple(round(s * 0.4 / 0.3) for s in toy_mask().shape)
        assert got.mask.shape == got.volume.shape
        assert any("resampled" in w for w in got.warnings)
        # Nearest-neighbour: no class index is invented at a boundary.
        assert set(np.unique(got.mask)) <= set(np.unique(toy_mask()))

    def test_a_mask_that_does_not_fit_the_image_is_refused(self, tmp_path):
        import nibabel as nib
        image, _ = write_toy_scan(tmp_path)
        bad = tmp_path / "bad.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((10, 10, 10), np.int16), np.eye(4)), str(bad))
        with pytest.raises(ValueError, match="disagree"):
            prepare_scan(image, bad, CLIP, AIR, 0.3, 0.02)


def test_patient_id_drops_the_channel_suffix():
    assert patient_id_from_filename("ToothFairy3F_001_0000.nii.gz") == "ToothFairy3F_001"
    assert patient_id_from_filename("scan.nii") == "scan"


# --------------------------------------------------------------------------
# sites and patches: the label builder's positions, the training crop
# --------------------------------------------------------------------------
class TestSitesAndPatches:
    def test_sites_are_the_label_builders(self, tmp_path):
        """Positions come from score_one -- the function that wrote the CSV."""
        from src.data.site_scoring import score_one
        mask = toy_mask()
        rows = [r for r in score_one(mask, SPACING, RULES) if r["jaw"] == "lower"]
        sites = sites_from_mask(mask, SPACING, RULES)
        assert len(sites) == 14 == len(rows)
        by_tooth = {r["tooth"]: r for r in rows}
        for s in sites:
            r = by_tooth[s["tooth"]]
            assert (s["site_x"], s["site_y"], s["site_z"]) == (r["site_x"], r["site_y"], r["site_z"])

    def test_missing_teeth_are_the_sites_that_need_an_implant(self):
        sites = sites_from_mask(toy_mask(), SPACING, RULES)
        need = {s["tooth"] for s in sites if s["truth"]["needs_implant"] == 1}
        assert need == set(MISSING)

    def test_the_patch_is_the_one_casset_loads(self, tmp_path):
        """The app's input must be the input training and explanation used."""
        scan = prepared(tmp_path)
        cache = tmp_path / "cache"
        cache.mkdir()
        np.save(cache / "P.npy", scan.volume)
        sites = sites_from_mask(scan.mask, scan.spacing, RULES)
        frame = pd.DataFrame([{**s, "patient_id": "P"} for s in sites])
        cases = CaseSet(ids=site_case_ids(frame), y=np.zeros((len(frame), 3), np.float32),
                        cache=cache, labels=NAMES, sites=frame, patch_size=16)
        for case_id, site in zip(cases.ids, sites):
            want = cases.load(case_id, CPU)[0, 0].numpy()
            assert np.array_equal(site_patch(scan.volume, site, 16), want), case_id


# --------------------------------------------------------------------------
# checkpoints: read the architecture, refuse what does not add up
# --------------------------------------------------------------------------
class TestCheckpoints:
    def test_the_shapes_give_every_field_but_the_head_count(self):
        shapes = read_architecture(tiny_model().state_dict())
        for key in ("in_channels", "stem_channels", "embed_dim", "patch_size", "depth",
                    "mlp_ratio", "num_classes", "img_size"):
            assert shapes[key] == pytest.approx(MODEL_CONFIG[key]), key
        assert "num_heads" not in shapes

    def test_an_embedded_config_needs_nothing_else(self, tmp_path):
        bundle = load_bundle(write_tiny_checkpoint(tmp_path / "m.pt"), CPU)
        assert bundle.architecture_source == "embedded in the checkpoint"
        assert bundle.names == NAMES and bundle.img_size == 16
        assert bundle.temperature == 1.0 and bundle.calibration_source is None

    def test_a_contradicting_declaration_is_refused_not_resolved(self, tmp_path):
        """patch 4 against weights that say 2 is two different networks."""
        path = write_tiny_checkpoint(tmp_path / "m.pt",
                                     model_config={**MODEL_CONFIG, "patch_size": 4})
        with pytest.raises(CheckpointError, match="patch_size"):
            load_bundle(path, CPU)

    def test_heads_must_be_declared_and_must_divide(self):
        shapes = read_architecture(tiny_model().state_dict())
        assert any("num_heads" in p for p in check_declared(shapes, {}))
        assert any("divide" in p for p in check_declared(shapes, {"num_heads": 3}))
        assert not check_declared(shapes, {"num_heads": 4})

    def test_a_legacy_checkpoint_borrows_the_config_but_is_still_checked(self, tmp_path):
        """No model_config: the fallback is consulted, and its shape-checkable
        fields must still match. configs/sites.yaml describes a 256-wide ViT."""
        path = write_tiny_checkpoint(tmp_path / "m.pt", embed_config=False)
        with pytest.raises(CheckpointError, match="embed_dim"):
            load_bundle(path, CPU, fallback_config="configs/sites.yaml")

    def test_a_sidecar_config_is_found_beside_the_weights(self, tmp_path):
        import yaml
        path = write_tiny_checkpoint(tmp_path / "m.pt", embed_config=False)
        (tmp_path / "config.yaml").write_text(yaml.safe_dump({"model": MODEL_CONFIG}))
        bundle = load_bundle(path, CPU)
        assert bundle.architecture_source == "sidecar config.yaml"

    def test_a_checkpoint_without_the_millimetre_heads_is_refused(self, tmp_path):
        from src.train.targets import TargetSpec
        spec = TargetSpec(binary=["implant", "crown", "bridge"]).state_dict()
        path = write_tiny_checkpoint(tmp_path / "m.pt", target_spec=spec,
                                     label_names=["implant", "crown", "bridge"])
        with pytest.raises(CheckpointError, match="millimetre"):
            load_bundle(path, CPU)

    def test_a_pickled_object_is_not_unpickled(self, tmp_path):
        """A browser-picked file must not run code on load."""
        path = write_tiny_checkpoint(tmp_path / "m.pt", extra=SimpleNamespace(x=1))
        with pytest.raises(CheckpointError, match="plain-data"):
            load_bundle(path, CPU)

    def test_calibration_and_metrics_are_read_from_beside_the_weights(self, tmp_path):
        path = write_tiny_checkpoint(tmp_path / "m.pt")
        (tmp_path / "calibration.json").write_text(json.dumps(
            {"temperature": 1.25, "gate_threshold": -0.3, "gate_uncertainty": "margin"}))
        (tmp_path / "best_val_metrics.json").write_text(json.dumps(
            {"thresholds": {"needs_implant": 0.62},
             "regression": {"available_height_mm": {"mae": 3.7}, "ridge_width_mm": {"mae": 2.6}}}))
        bundle = load_bundle(path, CPU)
        assert bundle.temperature == 1.25 and bundle.gate["threshold"] == -0.3
        assert bundle.thresholds == {"needs_implant": 0.62}
        assert bundle.mae == {"available_height_mm": 3.7, "ridge_width_mm": 2.6}

    def test_a_nan_temperature_is_refused(self, tmp_path):
        path = write_tiny_checkpoint(tmp_path / "m.pt")
        (tmp_path / "calibration.json").write_text('{"temperature": NaN}')
        with pytest.raises(CheckpointError, match="temperature"):
            load_bundle(path, CPU)

    def test_training_now_writes_the_architecture_into_the_checkpoint(self, tmp_path):
        """The fix for num_heads being unreadable: the Trainer records it."""
        from src.train.loop import Trainer
        cfg = SimpleNamespace(
            model=SimpleNamespace(**MODEL_CONFIG),
            task=SimpleNamespace(mm_weight=1.0),
            train=SimpleNamespace(lr=1e-3, weight_decay=0.0, amp=False, out_dir=str(tmp_path)),
        )
        trainer = Trainer(tiny_model(), cfg, NAMES, device=CPU, out_dir=tmp_path,
                          spec=tiny_spec())
        trainer.save_checkpoint(0, "best.pt")
        ckpt = torch.load(tmp_path / "best.pt", weights_only=True)
        assert ckpt["model_config"]["num_heads"] == 2
        assert load_bundle(tmp_path / "best.pt", CPU).architecture_source == "embedded in the checkpoint"


# --------------------------------------------------------------------------
# predictions and verdicts
# --------------------------------------------------------------------------
class TestPredictionsAndVerdicts:
    def bundle(self, tmp_path):
        return load_bundle(write_tiny_checkpoint(tmp_path / "m.pt"), CPU)

    def test_every_positioned_site_gets_outputs_in_report_units(self, tmp_path):
        scan = prepared(tmp_path / "scan")
        sites = predict_sites(self.bundle(tmp_path), scan.volume,
                              sites_from_mask(scan.mask, scan.spacing, RULES))
        for s in sites:
            out = s["prediction"]["outputs"]
            assert 0.0 <= out["needs_implant"] <= 1.0
            # Millimetres, not a sigmoid: the toy spec centres them at 14 and 8.
            assert out["available_height_mm"] > 1.0 and out["ridge_width_mm"] > 1.0

    def test_predictions_match_a_direct_forward_pass(self, tmp_path):
        from src.train.targets import to_report_units
        scan = prepared(tmp_path / "scan")
        bundle = self.bundle(tmp_path)
        sites = sites_from_mask(scan.mask, scan.spacing, RULES)
        got = predict_sites(bundle, scan.volume, sites)
        x = torch.from_numpy(site_patch(scan.volume, sites[0], 16))[None, None]
        with torch.no_grad():
            want = to_report_units(bundle.model(x).numpy(), bundle.spec)[0]
        np.testing.assert_allclose(list(got[0]["prediction"]["outputs"].values()), want,
                                   rtol=1e-5)

    def site(self, p, h, w):
        return {"tooth": 36, "jaw": "lower", "prediction": {"outputs": {
            "needs_implant": p, "available_height_mm": h, "ridge_width_mm": w}}}

    INFO = {"thresholds": {"needs_implant": 0.5},
            "mae": {"available_height_mm": 1.0, "ridge_width_mm": 0.5}}

    def status(self, *a, rules=RULES, info=None):
        return score_sites([self.site(*a)], info or self.INFO, rules)[0]["verdict"]["status"]

    def test_the_verdict_is_the_rule_on_the_millimetres(self):
        assert self.status(0.9, 20.0, 9.0) == "feasible"
        assert self.status(0.9, 8.0, 9.0) == "not_feasible"
        assert self.status(0.9, 20.0, 3.0) == "not_feasible"
        assert self.status(0.1, 8.0, 3.0) == "not_needed"

    def test_a_threshold_change_is_a_rescore(self):
        """Lowering 12 mm to 9 mm flips an 11.5 mm site without running anything."""
        assert self.status(0.9, 11.5, 9.0) == "borderline"
        assert self.status(0.9, 11.5, 9.0, info={**self.INFO, "mae": {}}) == "not_feasible"
        relaxed = {**RULES, "min_height_mandible_mm": 9.0}
        assert self.status(0.9, 11.5, 9.0, rules=relaxed, info={**self.INFO, "mae": {}}) == "feasible"

    def test_borderline_is_within_the_models_own_error(self):
        assert self.status(0.9, 12.5, 9.0) == "borderline"      # 0.5 mm from 12, MAE 1.0
        assert self.status(0.9, 13.5, 9.0) == "feasible"        # 1.5 mm away

    def test_without_a_measured_error_nothing_is_called_borderline(self):
        """No MAE, no band: the app does not invent a precision it was not given."""
        assert self.status(0.9, 12.1, 9.0, info={"thresholds": {}, "mae": {}}) == "feasible"

    def test_the_decision_threshold_comes_from_validation(self):
        info = {"thresholds": {"needs_implant": 0.8}, "mae": {}}
        assert self.status(0.7, 20.0, 9.0, info=info) == "not_needed"

    def test_a_site_without_a_position_is_said_to_have_none(self):
        v = score_sites([{"tooth": 36, "jaw": "lower", "prediction": None}], self.INFO, RULES)
        assert v[0]["verdict"]["status"] == "no_position"
