"""HTTP API and static page.

Routes are thin: they parse a request, call the registry, the store or the
queue, and return what those produce. Arrays go out as raw uint8 bytes with
their shape in a header, so a 96^3 patch is 0.9 MB rather than a JSON list
five times that size.
"""

from __future__ import annotations

import csv
import io
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import analysis
from app.jobs import JobQueue
from app.registry import ModelRegistry
from app.settings import REPO_ROOT, Settings
from app.store import ScanStore, clean
from src.inference.checkpoint import CheckpointError
from src.inference.predict import site_patch

STATIC = Path(__file__).resolve().parent / "static"


def version() -> str:
    path = REPO_ROOT / "VERSION"
    return path.read_text(encoding="utf-8").strip() if path.is_file() else "unknown"


class Rules(BaseModel):
    min_height_mandible_mm: float | None = None
    min_height_maxilla_mm: float | None = None
    min_width_mm: float | None = None


class ExplainRequest(BaseModel):
    target: str
    force: bool = False


def json(data, status: int = 200) -> JSONResponse:
    return JSONResponse(clean(data), status_code=status)


def array_response(array: np.ndarray, **headers) -> Response:
    array = np.ascontiguousarray(array, dtype=np.uint8)
    h = {"X-Shape": ",".join(str(int(s)) for s in array.shape),
         "Cache-Control": "no-store"}
    h.update({f"X-{k.replace('_', '-').title()}": str(v) for k, v in headers.items()})
    return Response(array.tobytes(), media_type="application/octet-stream", headers=h)


@contextmanager
def errors():
    """Domain errors -> HTTP status codes with the domain's own message."""
    try:
        yield
    except HTTPException:
        raise
    except KeyError as exc:
        raise HTTPException(404, f"not found: {exc.args[0] if exc.args else exc}") from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, f"not available yet: {exc}") from exc
    except (CheckpointError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


def create_app(settings: Settings) -> FastAPI:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = settings.data_dir / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    registry = ModelRegistry(settings)
    store = ScanStore(settings.scans_dir)
    jobs = JobQueue()

    app = FastAPI(title="Implant site screening", version=version())
    app.state.settings, app.state.registry = settings, registry
    app.state.store, app.state.jobs = store, jobs

    def save_upload(upload: UploadFile) -> Path:
        fd, name = tempfile.mkstemp(dir=tmp_root)
        with open(fd, "wb") as out:
            shutil.copyfileobj(upload.file, out, length=4 * 1024 * 1024)
        return Path(name)

    def rules_from(body: Rules | None) -> dict:
        rules = dict(settings.rules)
        if body is not None:
            for key, value in body.model_dump().items():
                if value is not None:
                    if not (0.0 < float(value) < 100.0):
                        raise HTTPException(400, f"{key}={value} is not a plausible length in mm")
                    rules[key] = float(value)
        return rules

    # ---- status ---------------------------------------------------------
    @app.get("/api/status")
    def status():
        return json({
            "version": version(),
            "device": str(settings.device),
            "config": settings.config_path.name,
            "rules": settings.rules,
            "window": settings.window,
            "spacing_mm": settings.spacing_mm,
            "xai": settings.xai,
            "active": {k: (registry.active(k) or {}).get("id") for k in ("site", "localiser")},
        })

    # ---- models ---------------------------------------------------------
    @app.get("/api/models")
    def list_models():
        return json(registry.list())

    @app.post("/api/models")
    def add_model(files: list[UploadFile] = File(...), kind: str = Form("site"),
                  name: str | None = Form(None), fold: str | None = Form(None)):
        saved = []
        try:
            for f in files:
                saved.append((f.filename or "upload", save_upload(f)))
            fold_value = int(fold) if fold not in (None, "") else None
            with errors():
                meta = registry.add(saved, kind=kind, name=name or None, fold=fold_value)
        finally:
            for _, path in saved:
                path.unlink(missing_ok=True)
        return json(meta, 201)

    @app.post("/api/models/{model_id}/activate")
    def activate_model(model_id: str):
        with errors():
            return json(registry.activate(model_id))

    @app.delete("/api/models/{model_id}")
    def delete_model(model_id: str):
        with errors():
            registry.remove(model_id)
        return json({"deleted": model_id})

    # ---- scans ----------------------------------------------------------
    @app.get("/api/scans")
    def list_scans():
        return json(store.list())

    @app.post("/api/scans")
    def add_scan(image: UploadFile = File(...), mask: UploadFile | None = File(None)):
        if registry.active("site") is None:
            raise HTTPException(400, "add a site model (.pt) before uploading a scan")
        image_tmp = save_upload(image)
        mask_tmp = save_upload(mask) if mask is not None and mask.filename else None
        try:
            with errors():
                meta = store.create((image.filename or "scan.nii.gz", image_tmp),
                                    (mask.filename, mask_tmp) if mask_tmp else None)
        finally:
            image_tmp.unlink(missing_ok=True)
            if mask_tmp:
                mask_tmp.unlink(missing_ok=True)
        job = jobs.submit("analyse", meta["id"], lambda progress: analysis.analyse(
            settings, registry, store, meta["id"], progress))
        return json({"scan": meta, "job": job.public()}, 202)

    @app.get("/api/scans/{scan_id}")
    def get_scan(scan_id: str):
        with errors():
            meta = store.meta(scan_id)
            result = store.result(scan_id)
        body = {"meta": meta, "jobs": [j.public() for j in jobs.active_for(scan_id)]}
        if result is not None and "model" in result:
            body["result"] = analysis.scored(result, settings.rules)
        elif result is not None:
            body["result"] = result
        return json(body)

    @app.post("/api/scans/{scan_id}/score")
    def score_scan(scan_id: str, body: Rules):
        with errors():
            result = store.result(scan_id)
        if result is None or "model" not in result:
            raise HTTPException(409, "this scan has no predictions yet")
        # Re-scoring is the whole point of predicting millimetres: the
        # thresholds move, the model does not run.
        return json(analysis.scored(result, rules_from(body)))

    @app.post("/api/scans/{scan_id}/predict")
    def repredict(scan_id: str):
        with errors():
            store.meta(scan_id)
        job = jobs.submit("predict", scan_id, lambda progress: analysis.predict(
            settings, registry, store, scan_id, progress))
        return json({"job": job.public()}, 202)

    @app.delete("/api/scans/{scan_id}")
    def delete_scan(scan_id: str):
        if jobs.active_for(scan_id):
            raise HTTPException(409, "this scan is still being processed")
        with errors():
            store.delete(scan_id)
        return json({"deleted": scan_id})

    @app.get("/api/scans/{scan_id}/overview")
    def get_overview(scan_id: str):
        with errors():
            image = store.load_array(scan_id, "overview.npy", mmap=False)
        return array_response(image)

    def find_site(result: dict, tooth: int) -> dict:
        site = next((s for s in result["sites"] if int(s["tooth"]) == int(tooth)), None)
        if site is None:
            raise KeyError(f"site {tooth}")
        return site

    @app.get("/api/scans/{scan_id}/sites/{tooth}/patch")
    def get_patch(scan_id: str, tooth: int):
        with errors():
            result = store.result(scan_id)
            if result is None or "model" not in result:
                raise FileNotFoundError("predictions")
            site = find_site(result, tooth)
            if site.get("prediction") is None:
                raise HTTPException(409, f"site {tooth} has no position")
            size = int(result["model"]["description"]["architecture"]["img_size"])
            volume = store.load_array(scan_id, "volume.npy")
            # The exact input the model saw: same patch_centre, same cut_patch.
            patch = site_patch(volume, site, size)
        lo, hi = settings.window
        return array_response(analysis.to_uint8(patch, lo, hi), window=f"{lo},{hi}")

    @app.post("/api/scans/{scan_id}/sites/{tooth}/explain")
    def start_explain(scan_id: str, tooth: int, body: ExplainRequest):
        with errors():
            store.meta(scan_id)
            key = analysis.explain_key(tooth, body.target, body.force)
            cached = store.explain_dir(scan_id, key) / "meta.json"
            if cached.is_file():
                return json({"explanation": store.load_json(scan_id, f"explain/{key}/meta.json")})
        job = jobs.submit("explain", scan_id, lambda progress: analysis.explain(
            settings, registry, store, scan_id, tooth, body.target, body.force, progress))
        return json({"job": job.public()}, 202)

    @app.get("/api/scans/{scan_id}/explain/{key}/{method}")
    def get_map(scan_id: str, key: str, method: str):
        with errors():
            folder = store.explain_dir(scan_id, key)
            path = folder / f"{method}.npy"
            if not path.is_file() or path.parent != folder:
                raise KeyError(method)
            array = np.load(path)
        return array_response(array)

    @app.get("/api/scans/{scan_id}/report.csv")
    def report(scan_id: str, min_height_mandible_mm: float | None = None,
               min_width_mm: float | None = None):
        with errors():
            result = store.result(scan_id)
            meta = store.meta(scan_id)
        if result is None or "model" not in result:
            raise HTTPException(409, "this scan has no predictions yet")
        rules = rules_from(Rules(min_height_mandible_mm=min_height_mandible_mm,
                                 min_width_mm=min_width_mm))
        scored = analysis.scored(result, rules)
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["patient_id", "tooth", "status", "p_needs_implant",
                         "available_height_mm", "ridge_width_mm", "height_rule_mm",
                         "width_rule_mm", "reasons", "site_source", "site_method",
                         "true_needs_implant", "true_available_height_mm",
                         "true_ridge_width_mm", "model", "screening_aid_only"])
        for s in scored["sites"]:
            out = (s.get("prediction") or {}).get("outputs", {})
            truth = s.get("truth") or {}
            v = s["verdict"]
            writer.writerow([
                meta["patient_id"], s["tooth"], v["status"], out.get("needs_implant"),
                out.get("available_height_mm"), out.get("ridge_width_mm"),
                rules["min_height_mandible_mm"], rules["min_width_mm"],
                "; ".join(v.get("reasons", [])), s.get("source"), s.get("method"),
                truth.get("needs_implant"), truth.get("available_height_mm"),
                truth.get("ridge_width_mm"), result["model"]["name"], "yes"])
        name = f"{meta['patient_id']}_implant_sites.csv"
        return Response(buf.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="{name}"'})

    # ---- jobs -----------------------------------------------------------
    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        return json(job.public())

    # ---- page -----------------------------------------------------------
    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
