"""The app end to end, through HTTP: add a model, upload a scan, read the result.

These drive the real server -- registry, store, queue, analysis -- against the
toy scan and a tiny checkpoint, so they run in seconds on a CPU and need no
cohort. What they pin is behaviour a user relies on: a model is validated
before it is kept, a threshold change is a re-score, a patch is the model's
own input, and failures arrive as messages rather than as a broken page.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from app.server import create_app  # noqa: E402
from app.settings import load_settings  # noqa: E402
from tests.toy_scan import (  # noqa: E402
    MISSING,
    write_tiny_checkpoint,
    write_tiny_localiser,
    write_toy_scan,
)

TINY_XAI = {"ig_steps": 4, "ig_batch": 2, "shap_samples": 2, "shap_batch": 2,
            "fusion_steps": 4, "default_gate_threshold": -0.25}


@pytest.fixture
def client(tmp_path):
    settings = load_settings("configs/app.yaml", data_dir=str(tmp_path / "data"), device="cpu")
    settings.xai = dict(TINY_XAI)
    app = create_app(settings)
    with TestClient(app) as c:
        c.app_state = app.state
        yield c


@pytest.fixture
def files(tmp_path):
    ckpt = write_tiny_checkpoint(tmp_path / "upload" / "cv_fold0_best.pt")
    (tmp_path / "upload" / "calibration.json").write_text(json.dumps(
        {"temperature": 1.3, "gate_threshold": -0.2, "gate_uncertainty": "margin",
         "fitted_on": "validation split only"}))
    # Decision threshold 0: every site "needs an implant", so the feasibility
    # verdicts below are exercised on all fourteen rather than on whichever an
    # untrained toy model happens to call.
    (tmp_path / "upload" / "best_val_metrics.json").write_text(json.dumps(
        {"thresholds": {"needs_implant": 0.0},
         "regression": {"available_height_mm": {"mae": 3.7}, "ridge_width_mm": {"mae": 2.6}}}))
    image, mask = write_toy_scan(tmp_path / "scan")
    return {"ckpt": ckpt, "cal": tmp_path / "upload" / "calibration.json",
            "metrics": tmp_path / "upload" / "best_val_metrics.json",
            "image": image, "mask": mask}


def add_model(client, *paths, **form):
    handles = [("files", (p.name, p.read_bytes())) for p in paths]
    return client.post("/api/models", files=handles, data={"kind": "site", **form})


def upload(client, image, mask=None):
    handles = {"image": (image.name, image.read_bytes())}
    if mask is not None:
        handles["mask"] = (mask.name, mask.read_bytes())
    return client.post("/api/scans", files=handles)


def analysed(client, files):
    assert add_model(client, files["ckpt"], files["cal"], files["metrics"]).status_code == 201
    res = upload(client, files["image"], files["mask"])
    assert res.status_code == 202, res.text
    client.app_state.jobs.join()
    scan_id = res.json()["scan"]["id"]
    job = client.get(f"/api/jobs/{res.json()['job']['id']}").json()
    assert job["status"] == "done", job
    return scan_id


class TestModels:
    def test_status_reports_the_configured_rules(self, client):
        body = client.get("/api/status").json()
        assert body["rules"]["min_height_mandible_mm"] == 12.0
        assert body["rules"]["min_width_mm"] == 6.0
        assert body["device"] == "cpu"

    def test_a_model_and_its_companions_are_kept_and_activated(self, client, files):
        res = add_model(client, files["ckpt"], files["cal"], files["metrics"])
        assert res.status_code == 201, res.text
        meta = res.json()
        assert meta["active"] and meta["fold"] == 0, "fold is read off cv_fold0 in the name"
        assert set(meta["files"]) == {"model.pt", "calibration.json", "best_val_metrics.json"}
        d = meta["description"]
        assert d["calibrated"] and d["temperature"] == 1.3 and d["gate"]["threshold"] == -0.2
        assert d["validation_mae_mm"]["available_height_mm"] == 3.7
        assert [m["id"] for m in client.get("/api/models").json()] == [meta["id"]]

    def test_a_file_that_is_not_a_site_model_is_refused_and_not_kept(self, client, tmp_path):
        bad = tmp_path / "bad.pt"
        torch.save({"model": {"weight": torch.zeros(3)}}, bad)
        res = add_model(client, bad)
        assert res.status_code == 400 and "vit3d" in res.json()["detail"]
        assert client.get("/api/models").json() == []

    def test_an_unrecognised_companion_is_refused(self, client, files, tmp_path):
        other = tmp_path / "notes.txt"
        other.write_text("hello")
        res = add_model(client, files["ckpt"], other)
        assert res.status_code == 400 and "not recognised" in res.json()["detail"]

    def test_a_scan_needs_a_model_first(self, client, files):
        res = upload(client, files["image"], files["mask"])
        assert res.status_code == 400 and "model" in res.json()["detail"]


class TestAnalysis:
    def test_every_lower_site_is_located_predicted_and_judged(self, client, files):
        scan_id = analysed(client, files)
        result = client.get(f"/api/scans/{scan_id}").json()["result"]
        assert result["site_source"] == "mask"
        assert result["scan"]["orientation_source"] == "mask anatomy"
        teeth = [s["tooth"] for s in result["sites"]]
        assert teeth == [47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37]
        for s in result["sites"]:
            assert s["prediction"] is not None and s["verdict"]["status"]
        truth_need = {s["tooth"] for s in result["sites"] if s["truth"]["needs_implant"] == 1}
        assert truth_need == set(MISSING)

    def test_a_threshold_change_rescores_without_running_the_model(self, client, files):
        """The point of predicting millimetres, through the API."""
        scan_id = analysed(client, files)
        lenient = client.post(f"/api/scans/{scan_id}/score",
                              json={"min_height_mandible_mm": 0.5, "min_width_mm": 0.5}).json()
        strict = client.post(f"/api/scans/{scan_id}/score",
                             json={"min_height_mandible_mm": 90, "min_width_mm": 90}).json()
        needs = [s for s in lenient["sites"] if s["verdict"]["needs_implant"]]
        assert len(needs) == 14
        for a in needs:
            b = next(s for s in strict["sites"] if s["tooth"] == a["tooth"])
            assert a["verdict"]["status"] == "feasible"
            assert b["verdict"]["status"] == "not_feasible"
            assert a["prediction"] == b["prediction"], "the model must not have run again"

    def test_an_implausible_threshold_is_refused(self, client, files):
        scan_id = analysed(client, files)
        res = client.post(f"/api/scans/{scan_id}/score", json={"min_height_mandible_mm": -3})
        assert res.status_code == 400

    def test_the_patch_is_the_models_input(self, client, files):
        scan_id = analysed(client, files)
        res = client.get(f"/api/scans/{scan_id}/sites/36/patch")
        assert res.status_code == 200
        assert res.headers["x-shape"] == "16,16,16"
        assert len(res.content) == 16 ** 3

    def test_the_overview_is_served(self, client, files):
        scan_id = analysed(client, files)
        res = client.get(f"/api/scans/{scan_id}/overview")
        shape = [int(v) for v in res.headers["x-shape"].split(",")]
        assert len(res.content) == int(np.prod(shape))

    def test_the_report_carries_the_rules_it_was_scored_at(self, client, files):
        scan_id = analysed(client, files)
        text = client.get(f"/api/scans/{scan_id}/report.csv",
                          params={"min_height_mandible_mm": 10, "min_width_mm": 5}).text
        lines = text.strip().splitlines()
        assert lines[0].startswith("patient_id,tooth,status")
        assert len(lines) == 15
        assert all(",10.0,5.0," in line for line in lines[1:])
        assert all(line.endswith(",yes") for line in lines[1:]), "screening-aid flag on every row"

    def test_a_scan_without_mask_or_localiser_fails_with_a_reason(self, client, files):
        assert add_model(client, files["ckpt"]).status_code == 201
        res = upload(client, files["image"])
        client.app_state.jobs.join()
        job = client.get(f"/api/jobs/{res.json()['job']['id']}").json()
        assert job["status"] == "failed"
        assert "mask" in job["error"] and "localiser" in job["error"]

    def test_without_a_mask_the_localiser_finds_the_sites(self, client, files, tmp_path):
        """The image-only path: a localiser model, no segmentation."""
        assert add_model(client, files["ckpt"]).status_code == 201
        loc = write_tiny_localiser(tmp_path / "localiser" / "localiser_best.pt")
        res = add_model(client, loc, kind="localiser")
        assert res.status_code == 201, res.text
        assert res.json()["kind"] == "localiser" and res.json()["active"]

        res = upload(client, files["image"])
        client.app_state.jobs.join()
        job = client.get(f"/api/jobs/{res.json()['job']['id']}").json()
        assert job["status"] == "done", job
        result = client.get(f"/api/scans/{res.json()['scan']['id']}").json()["result"]
        assert result["site_source"] == "localiser"
        assert result["scan"]["orientation_source"] in ("configured default",
                                                        "localiser orientation head")
        assert [s["tooth"] for s in result["sites"]] == [47, 46, 45, 44, 43, 42, 41,
                                                         31, 32, 33, 34, 35, 36, 37]
        for s in result["sites"]:
            assert s["source"] == "localiser" and s["truth"] is None
            assert s["verdict"]["status"]
        assert any("localiser" in w for w in result["warnings"]),             "an image-only result must say its sites were not measured from a mask"

    def test_a_site_model_is_not_accepted_as_a_localiser(self, client, files):
        res = add_model(client, files["ckpt"], kind="localiser")
        assert res.status_code == 400 and "localiser" in res.json()["detail"]

    def test_a_non_nifti_upload_is_refused(self, client, files, tmp_path):
        assert add_model(client, files["ckpt"]).status_code == 201
        other = tmp_path / "scan.png"
        other.write_bytes(b"not a scan")
        assert upload(client, other).status_code == 400


class TestExplanations:
    def test_the_full_ensemble_produces_every_map_and_its_weights(self, client, files):
        scan_id = analysed(client, files)
        res = client.post(f"/api/scans/{scan_id}/sites/36/explain",
                          json={"target": "available_height_mm", "force": True})
        assert res.status_code == 202
        client.app_state.jobs.join()
        meta = client.get(f"/api/jobs/{res.json()['job']['id']}").json()["result"]
        assert set(meta["methods"]) == {"attention_rollout", "gradcam", "integrated_gradients",
                                        "gradient_shap", "fused"}
        assert meta["routing"]["decision"] == "ensemble" and meta["routing"]["forced"]
        assert sum(meta["fusion"]["weights"].values()) == pytest.approx(1.0)
        assert meta["fusion"]["weight_metric"] != meta["fusion"]["eval_metric"]
        assert meta["fusion"]["score"] == "deviation", "a millimetre head is read as deviation"
        m = client.get(f"/api/scans/{scan_id}/explain/{meta['key']}/fused")
        assert m.headers["x-shape"] == "16,16,16" and len(m.content) == 16 ** 3

    def test_a_second_request_is_served_from_cache(self, client, files):
        scan_id = analysed(client, files)
        body = {"target": "needs_implant", "force": True}
        client.post(f"/api/scans/{scan_id}/sites/36/explain", json=body)
        client.app_state.jobs.join()
        again = client.post(f"/api/scans/{scan_id}/sites/36/explain", json=body)
        assert again.status_code == 200 and "explanation" in again.json()

    def test_the_gate_routes_by_the_fitted_threshold(self, client, files):
        """calibration.json says -0.2: a site is escalated only at or above it."""
        scan_id = analysed(client, files)
        res = client.post(f"/api/scans/{scan_id}/sites/36/explain",
                          json={"target": "needs_implant", "force": False})
        client.app_state.jobs.join()
        r = client.get(f"/api/jobs/{res.json()['job']['id']}").json()["result"]["routing"]
        assert r["threshold"] == -0.2 and "fitted on validation" in r["source"]
        assert r["decision"] == ("ensemble" if r["uncertainty"] >= -0.2 else "cheap")

    def test_an_unknown_output_is_refused(self, client, files):
        scan_id = analysed(client, files)
        res = client.post(f"/api/scans/{scan_id}/sites/36/explain",
                          json={"target": "feasible", "force": True})
        client.app_state.jobs.join()
        job = client.get(f"/api/jobs/{res.json()['job']['id']}").json()
        assert job["status"] == "failed" and "feasible" in job["error"]
