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
    # Addressed as the app is in a browser: it refuses names it does not have.
    with TestClient(app, base_url=f"http://{settings.host}:{settings.port}") as c:
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
        assert body["app"] == "implant-site-screening", "the launcher recognises the app by this"
        assert body["rules"]["min_height_mandible_mm"] == 12.0
        assert body["rules"]["min_width_mm"] == 6.0
        assert body["device"] == "cpu"

    def test_the_page_is_checked_for_changes_before_it_is_reused(self, client):
        """After an update the browser must not pair the new HTML with a cached script."""
        for path in ("/", "/static/app.js", "/static/app.css"):
            res = client.get(path)
            assert res.status_code == 200 and res.headers["cache-control"] == "no-cache", path

    def test_another_sites_page_cannot_change_anything(self, client, files):
        """A form POST to 127.0.0.1 needs nobody's permission, so the server has to
        look at who the browser says it is acting for."""
        handles = [("files", (files["ckpt"].name, files["ckpt"].read_bytes()))]
        foreign = client.post("/api/models", files=handles, data={"kind": "site"},
                              headers={"Origin": "http://elsewhere.example"})
        assert foreign.status_code == 403
        assert client.get("/api/models").json() == []
        own = client.post("/api/models", files=handles, data={"kind": "site"},
                          headers={"Origin": str(client.base_url).rstrip("/")})
        assert own.status_code == 201, own.text
        model_id = own.json()["id"]
        assert client.delete(f"/api/models/{model_id}",
                             headers={"Origin": "http://elsewhere.example"}).status_code == 403
        assert client.get("/api/status", headers={"Origin": "http://elsewhere.example"}).status_code == 200

    def test_the_app_does_not_answer_to_a_name_it_does_not_have(self, client, files):
        """A site that points its own name at 127.0.0.1 shares an origin with the
        app, so the Origin check passes and it could read scans too."""
        as_other = {"Host": "elsewhere.example:8000"}
        assert client.get("/api/scans", headers=as_other).status_code == 403
        assert client.get("/", headers=as_other).status_code == 403
        handles = [("files", (files["ckpt"].name, files["ckpt"].read_bytes()))]
        res = client.post("/api/models", files=handles, data={"kind": "site"},
                          headers={**as_other, "Origin": "http://elsewhere.example:8000"})
        assert res.status_code == 403 and client.get("/api/models").json() == []
        for name in ("localhost:8000", "127.0.0.1:8000", "[::1]:8000", "LOCALHOST"):
            assert client.get("/api/status", headers={"Host": name}).status_code == 200, name

    def test_an_app_put_on_a_network_answers_to_any_name(self, tmp_path):
        """Bound beyond this machine it was exposed on purpose, and its names
        there cannot be known from here."""
        settings = load_settings("configs/app.yaml", data_dir=str(tmp_path / "lan"),
                                 device="cpu", host="0.0.0.0")
        with TestClient(create_app(settings), base_url="http://scanner.lan:8000") as c:
            assert c.get("/api/status").status_code == 200

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

    def test_a_fold_that_is_not_a_number_is_refused_with_a_reason(self, client, files):
        """It used to reach int() outside the error handling and return a bare 500."""
        res = add_model(client, files["ckpt"], fold="first")
        assert res.status_code == 400 and "fold" in res.json()["detail"]
        assert client.get("/api/models").json() == []


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
        assert any("localiser" in w for w in result["warnings"]), \
            "an image-only result must say its sites were not measured from a mask"

    @pytest.mark.parametrize("accuracy, obeyed", [(1.0, True), (0.5, False), (None, False)])
    def test_the_orientation_head_is_obeyed_only_when_it_has_earned_it(
            self, client, files, tmp_path, monkeypatch, accuracy, obeyed):
        """A three-epoch smoke localiser, at chance on orientation, turned a
        correctly stored real scan upside down, and the result looked normal."""
        from src.models.localiser import Localiser

        real = Localiser.predict

        def says_upside_down(self, volume):
            return {**real(self, volume), "flip_prob": 0.93}

        monkeypatch.setattr(Localiser, "predict", says_upside_down)
        assert add_model(client, files["ckpt"]).status_code == 201
        val = {} if accuracy is None else {"orientation_accuracy": accuracy}
        loc = write_tiny_localiser(tmp_path / "localiser" / "localiser_best.pt", val=val)
        assert add_model(client, loc, kind="localiser").status_code == 201

        res = upload(client, files["image"])
        client.app_state.jobs.join()
        result = client.get(f"/api/scans/{res.json()['scan']['id']}").json()["result"]
        default = client.app_state.settings.default_orientation_sign
        said = [w for w in result["warnings"] if "orientation head" in w]
        assert len(said) == 1 and "P = 0.93" in said[0], result["warnings"]
        if obeyed:
            assert result["scan"]["orientation_sign"] == -default
            assert result["scan"]["orientation_source"] == "localiser orientation head"
            assert "100% correct" in said[0] and "turned over" in said[0]
        else:
            assert result["scan"]["orientation_sign"] == default
            assert result["scan"]["orientation_source"] == "configured default"
            assert "default orientation was kept" in said[0]
            assert ("50% correct" if accuracy is not None else "no measured accuracy") in said[0]

    def test_a_site_model_is_not_accepted_as_a_localiser(self, client, files):
        res = add_model(client, files["ckpt"], kind="localiser")
        assert res.status_code == 400 and "localiser" in res.json()["detail"]

    def test_a_non_nifti_upload_is_refused(self, client, files, tmp_path):
        assert add_model(client, files["ckpt"]).status_code == 201
        other = tmp_path / "scan.png"
        other.write_bytes(b"not a scan")
        assert upload(client, other).status_code == 400

    def test_a_failed_analysis_says_why_and_can_be_run_again(self, client, files, tmp_path):
        """A scan whose analysis failed used to be stuck: no result, no reason
        once the progress bar was gone, and no way forward but uploading again."""
        assert add_model(client, files["ckpt"]).status_code == 201
        scan_id = upload(client, files["image"]).json()["scan"]["id"]
        client.app_state.jobs.join()
        body = client.get(f"/api/scans/{scan_id}").json()
        assert "result" not in body
        assert body["failed"]["status"] == "failed" and "localiser" in body["failed"]["error"]

        # What it was missing arrives; the same upload is analysed again.
        loc = write_tiny_localiser(tmp_path / "localiser" / "localiser_best.pt")
        assert add_model(client, loc, kind="localiser").status_code == 201
        res = client.post(f"/api/scans/{scan_id}/analyse")
        assert res.status_code == 202, res.text
        client.app_state.jobs.join()
        body = client.get(f"/api/scans/{scan_id}").json()
        assert "failed" not in body
        assert len(body["result"]["sites"]) == 14 and body["result"]["model"]

    def test_a_removed_scan_is_gone(self, client, files):
        scan_id = analysed(client, files)
        folder = client.app_state.store.dir(scan_id)
        assert client.delete(f"/api/scans/{scan_id}").status_code == 200
        assert not folder.exists()
        assert client.get(f"/api/scans/{scan_id}").status_code == 404
        assert client.get("/api/scans").json() == []


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

    def test_an_unknown_output_is_refused_before_any_job_starts(self, client, files):
        scan_id = analysed(client, files)
        res = client.post(f"/api/scans/{scan_id}/sites/36/explain",
                          json={"target": "feasible", "force": True})
        assert res.status_code == 400 and "feasible" in res.json()["detail"]

    def test_a_tooth_that_is_not_a_site_is_refused(self, client, files):
        scan_id = analysed(client, files)
        res = client.post(f"/api/scans/{scan_id}/sites/99/explain",
                          json={"target": "needs_implant"})
        assert res.status_code == 404


class TestNamesFromTheUrl:
    """An id or key in a URL becomes a directory name. On Windows a backslash is
    a path separator inside ONE url segment, so `..\\models\\<id>` passed as a
    scan id resolved to a model's folder -- and DELETE removed it."""

    def test_a_scan_id_cannot_reach_a_models_folder(self, client, files):
        model_id = add_model(client, files["ckpt"]).json()["id"]
        folder = client.app_state.settings.models_dir / model_id
        escaped = f"..%5Cmodels%5C{model_id}"
        assert client.get(f"/api/scans/{escaped}").status_code == 404
        assert client.delete(f"/api/scans/{escaped}").status_code == 404
        assert (folder / "model.pt").is_file(), "the scan route removed a model"
        assert [m["id"] for m in client.get("/api/models").json()] == [model_id]

    def test_a_map_request_stays_inside_its_explanation(self, client, files):
        """`explain/../volume` used to serve the prepared volume as if it were a map."""
        scan_id = analysed(client, files)
        assert client.get(f"/api/scans/{scan_id}/explain/%2E%2E/volume").status_code == 404
        assert client.get(f"/api/scans/{scan_id}/explain/x/..%5C..%5Cvolume").status_code == 404

    def test_a_model_id_is_a_plain_name(self, client, files):
        scan_id = analysed(client, files)
        escaped = f"..%5Cscans%5C{scan_id}"
        assert client.delete(f"/api/models/{escaped}").status_code == 404
        assert client.post(f"/api/models/{escaped}/activate").status_code == 404
        assert client.get(f"/api/scans/{scan_id}").status_code == 200


class TestJobsRunTogether:
    """Jobs are concurrent. What still runs one at a time does so for
    correctness -- one scan's files, one model's hooks -- never for load."""

    def test_two_jobs_are_inside_the_pool_at_the_same_time(self):
        """A barrier only two threads can pass together: one worker would time out."""
        import threading

        from app.jobs import JobQueue

        queue, barrier = JobQueue(), threading.Barrier(2, timeout=10)
        jobs = [queue.submit("wait", f"scan-{i}", lambda progress: barrier.wait() or {})
                for i in range(2)]
        queue.join()
        assert [queue.get(j.id).status for j in jobs] == ["done", "done"], \
            [queue.get(j.id).error for j in jobs]

    def test_jobs_on_one_scan_do_not_overlap(self):
        """They write the same files: a re-prediction clears explanations."""
        import threading
        import time

        from app.jobs import JobQueue

        queue, inside, overlaps = JobQueue(), threading.Semaphore(1), []

        def work(progress):
            overlaps.append(not inside.acquire(blocking=False))
            time.sleep(0.05)
            inside.release()
            return {}

        for _ in range(4):
            queue.submit("write", "same-scan", work)
        queue.join()
        assert overlaps == [False] * 4

    def test_shutdown_drops_what_has_not_started_and_counts_what_is_running(self):
        """What the launcher asks on Ctrl+C: is anything still computing?"""
        import threading

        from app.jobs import JobQueue

        queue, started, release = JobQueue(workers=1), threading.Event(), threading.Event()

        def work(progress):
            started.set()
            release.wait(10)
            return {}

        first = queue.submit("explain", "a", work)
        waiting = queue.submit("explain", "b", work)
        assert started.wait(10)
        assert queue.shutdown() == 1
        release.set()
        queue.join()
        assert queue.get(first.id).status == "done"
        assert queue.get(waiting.id).status == "queued", "a job that had not started must not run"

    def test_two_scans_are_analysed_side_by_side(self, client, files):
        assert add_model(client, files["ckpt"], files["metrics"]).status_code == 201
        first = upload(client, files["image"], files["mask"]).json()
        second = upload(client, files["image"], files["mask"]).json()
        client.app_state.jobs.join()
        for res in (first, second):
            assert client.get(f"/api/jobs/{res['job']['id']}").json()["status"] == "done"
            sites = client.get(f"/api/scans/{res['scan']['id']}").json()["result"]["sites"]
            assert len(sites) == 14 and all(s["prediction"] for s in sites)

    def test_every_added_model_stays_loaded(self, client, files, tmp_path):
        """Switching models, or explaining a scan an earlier model analysed,
        must not reload one -- and adding a model already loads it."""
        a = add_model(client, files["ckpt"], name="a").json()["id"]
        b = add_model(client, write_tiny_checkpoint(tmp_path / "b" / "b.pt", seed=1), name="b").json()["id"]
        registry = client.app_state.registry
        assert {a, b} <= set(registry._loaded)
        first = registry.load(a)
        client.post(f"/api/models/{b}/activate")
        registry.load(b)
        assert registry.load(a) is first

    def test_a_model_has_one_lock_and_models_do_not_share_it(self, client, files, tmp_path):
        a = add_model(client, files["ckpt"], name="a").json()["id"]
        b = add_model(client, write_tiny_checkpoint(tmp_path / "b" / "b.pt", seed=1), name="b").json()["id"]
        registry = client.app_state.registry
        assert registry.lock(a) is registry.lock(a)
        assert registry.lock(a) is not registry.lock(b)
