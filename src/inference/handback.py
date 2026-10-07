"""Check what a GPU run produced against what the app needs, and pack it.

The August run trained five folds and ran every analysis, and what came back
was a report: the checkpoints and the result files stayed on the rented box.
This module is the step that was missing. It answers one question before
anything is transferred -- WILL THE APP ACCEPT THESE FILES -- by loading every
checkpoint with the app's own loader, on the machine that still has them, where
a problem costs a command and not another rental.

Nothing here is a second implementation: site models go through
`load_bundle`, localisers through `load_localiser`, both with the app's default
refusal to unpickle arbitrary objects. A file that passes here is a file the
Models dialog will take.

Three kinds of finding, and only the first stops a hand-back:

  PROBLEM  a file is present and unusable. Fix it before sending.
  MISSING  a file the run list produces is not there. Stated with what its
           absence costs, because a partial run is a legitimate thing to send.
  ok       present and accepted.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import tarfile
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path

import torch

from src.data.splits import check_folds_disjoint, load_folds
from src.inference.checkpoint import load_bundle
from src.models.localiser import load_localiser

MANIFEST = "handback_manifest.json"
OK, MISSING, PROBLEM = "ok", "MISSING", "PROBLEM"

# Result files directly under the artifacts directory: (glob, what it is, what
# is lost without it). Globs, because several are one-per-fold or one-per-run.
RESULT_FILES = (
    ("cv_folds.json", "the fold partition",
     "the app cannot say whether a scan was in a model's training data, and no "
     "later fold can be trained against the same split"),
    ("sites_toothfairy3.csv", "the label table the run trained on",
     "no site-level count can be checked against the run"),
    ("cv_pooled_metrics.json", "pooled AUROC, millimetre errors, feasibility agreement",
     "the headline result exists only as text"),
    ("cv_predictions.csv", "one row per site, predicted once by a model that never saw it",
     "nothing pooled can be recomputed without a GPU"),
    ("results_faithfulness*.csv", "deletion and insertion AUC per method per case", "no faithfulness intervals"),
    ("results_randomization*.csv", "the randomisation cascade", "no randomisation intervals"),
    ("results_agreement*.csv", "inter-method agreement", "no agreement table"),
    ("results_localization*.csv", "enrichment against the canal", "no localisation intervals"),
    ("results_ablations*.csv", "adaptive fusion per case", "the three adaptive claims cannot be re-read"),
    ("results_pareto*.csv", "compute against faithfulness", "no Pareto curve"),
    ("results_geometric_baseline_fold*.json", "the threshold-and-measure baseline", "no baseline rows"),
    ("calibration/calibration.json", "temperature and gate of the last fold calibrated", "none, if a copy sits beside its checkpoint"),
    ("xai_*.csv", "runtime, IG completeness, planted-signal sanity", "the Pareto costs cannot be checked"),
)
# Directories taken whole when present.
RESULT_DIRS = ("figures", "calibration")
# Beside each checkpoint. (name, why the app or the report wants it)
SITE_COMPANIONS = (
    ("best_val_metrics.json", "decision threshold and validation MAE: without it the app has no "
                              "threshold for 'needs an implant' and cannot mark a site borderline"),
    ("calibration.json", "temperature and gate from run_adaptive.py: without it every result in the "
                         "app says it is uncalibrated and gated by a default"),
    ("metrics.json", "the fold's validation table; pool_cv.py reads its thresholds"),
    ("history.csv", "the training curve"),
)
LOCALISER_COMPANIONS = (
    ("eval_test.json", "measured position error and coverage on held-out patients: without it the "
                       "image-only path has no error to quote"),
    ("end_to_end_test.csv", "the per-site cost of running without a mask"),
    ("history.csv", "the training curve"),
    ("summary.json", "best validation error"),
)


@dataclass
class Finding:
    status: str
    path: str
    detail: str


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    files: list[Path] = field(default_factory=list)      # everything to pack
    site_models: list[dict] = field(default_factory=list)
    localisers: list[dict] = field(default_factory=list)

    def add(self, status: str, path, detail: str) -> None:
        self.findings.append(Finding(status, str(path), detail))

    @property
    def problems(self) -> list[Finding]:
        return [f for f in self.findings if f.status == PROBLEM]

    @property
    def missing(self) -> list[Finding]:
        return [f for f in self.findings if f.status == MISSING]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment(repo: Path) -> dict:
    """What produced the files: enough to explain a difference after the fact."""
    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, timeout=20)
            return out.stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None

    def installed(name: str) -> str | None:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    version = repo / "VERSION"
    return {
        "code_version": version.read_text(encoding="utf-8").strip() if version.is_file() else None,
        "git_describe": git("describe", "--tags", "--always", "--dirty"),
        "git_commit": git("rev-parse", "HEAD"),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "packages": {name: installed(name) for name in
                     ("numpy", "scipy", "pandas", "nibabel", "PyYAML", "matplotlib")},
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "platform": platform.platform(),
    }


def _fold_dirs(root: Path) -> list[Path]:
    dirs = [d for d in root.glob("cv_fold*") if d.is_dir()] if root.is_dir() else []
    return sorted(dirs, key=lambda d: d.name)


def _companions(report: Report, directory: Path, wanted) -> list[str]:
    present = []
    for name, why in wanted:
        path = directory / name
        if path.is_file():
            report.files.append(path)
            present.append(name)
        else:
            report.add(MISSING, path, why)
    return present


def check_site_model(report: Report, directory: Path, app_config: Path, device: torch.device,
                     pack: bool = True) -> None:
    """One site model, loaded exactly as the app's Models dialog loads it.

    `pack=False` for a checkpoint that is not a cross-validation fold -- the
    smoke gate's, say. It is loaded and reported, which is the whole use of
    running this before paying for a GPU, and it is not sent.
    """
    ckpt = directory / "best.pt"
    if not ckpt.is_file():
        report.add(MISSING, ckpt, "this fold has no checkpoint, so it cannot be pooled or used in the app")
        return
    try:
        bundle = load_bundle(ckpt, device, fallback_config=app_config, allow_unsafe=False)
        with torch.no_grad():
            size = bundle.img_size
            out = bundle.model(torch.zeros(1, 1, size, size, size, device=device))
        if not bool(torch.isfinite(out).all()) or out.shape[1] != len(bundle.names):
            raise ValueError(f"a forward pass returned {tuple(out.shape)}, finite={bool(torch.isfinite(out).all())}")
    except Exception as exc:  # noqa: BLE001 - whatever stops the app from loading it is the finding
        report.add(PROBLEM, ckpt, f"the app would refuse this file: {exc}")
        return
    if not pack:
        report.add(OK, ckpt, f"the app accepts it ({bundle.names}, architecture "
                             f"{bundle.architecture_source}). Not a cross-validation fold: checked, not packed")
        return
    report.files.append(ckpt)
    present = _companions(report, directory, SITE_COMPANIONS)
    info = bundle.describe(0.3)
    notes = [f"epoch {bundle.epoch}", f"outputs {bundle.names}", f"{bundle.n_params / 1e6:.2f}M parameters",
             f"architecture {bundle.architecture_source}"]
    if "embedded" not in bundle.architecture_source:
        notes.append("NO model_config inside: written before v3.5.0. The app reads num_heads from "
                     "its own config for this file, which is right only for the default architecture")
    if "best_val_metrics.json" in present and not bundle.thresholds:
        report.add(PROBLEM, directory / "best_val_metrics.json", "holds no decision thresholds the app can read")
    if "calibration.json" in present and bundle.gate is None:
        notes.append("calibration.json has a temperature and NO gate threshold: written before v3.5.0, "
                     "the app will route by its default. Re-run run_adaptive.py")
    report.add(OK, ckpt, "; ".join(notes))
    report.site_models.append({
        "fold": directory.name, "checkpoint": str(ckpt), "epoch": bundle.epoch,
        "outputs": bundle.names, "architecture_source": bundle.architecture_source,
        "calibrated": info["calibrated"], "temperature": bundle.temperature,
        "gate": bundle.gate, "decision_thresholds": bundle.thresholds,
        "validation_mae_mm": bundle.mae, "companions": present})


def check_localiser(report: Report, directory: Path, device: torch.device, min_orientation: float) -> None:
    ckpt = directory / "best.pt"
    if not ckpt.is_file():
        report.add(MISSING, ckpt, "no localiser for this fold")
        return
    try:
        localiser = load_localiser(ckpt, device, allow_unsafe=False)
    except Exception as exc:  # noqa: BLE001
        report.add(PROBLEM, ckpt, f"the app would refuse this file: {exc}")
        return
    report.files.append(ckpt)
    present = _companions(report, directory, LOCALISER_COMPANIONS)
    val = localiser.meta.get("val") or {}
    accuracy, error = val.get("orientation_accuracy"), val.get("median_error_mm")
    notes = [f"epoch {localiser.meta.get('epoch')}", f"fold {localiser.meta.get('fold')}",
             "validation median error " + (f"{float(error):.2f} mm" if error is not None else "not recorded")]
    if accuracy is None:
        notes.append("records NO orientation accuracy, so the app will never turn a scan over on its word")
    elif float(accuracy) < min_orientation:
        notes.append(f"orientation accuracy {float(accuracy):.3f} is under the {min_orientation:.2f} the app "
                     f"requires: it will keep the default orientation and say so")
    else:
        notes.append(f"orientation accuracy {float(accuracy):.3f}")
    if val.get("median_spread_mm") is None:
        notes.append("no median_spread_mm: the app cannot flag an uncertain position")
    report.add(OK, ckpt, "; ".join(notes))
    report.localisers.append({"fold": directory.name, "checkpoint": str(ckpt),
                              "validation": val, "companions": present})


def check_results(report: Report, artifacts: Path) -> None:
    for pattern, what, cost in RESULT_FILES:
        found = sorted(p for p in artifacts.glob(pattern) if p.is_file())
        if not found:
            report.add(MISSING, artifacts / pattern, f"{what} -- {cost}")
            continue
        for path in found:
            report.files.append(path)
            report.add(OK, path, what)
    for name in RESULT_DIRS:
        folder = artifacts / name
        if folder.is_dir():
            report.files.extend(p for p in sorted(folder.rglob("*")) if p.is_file())

    folds = artifacts / "cv_folds.json"
    if folds.is_file():
        try:
            check_folds_disjoint(load_folds(folds))
        except Exception as exc:  # noqa: BLE001
            report.add(PROBLEM, folds, f"the partition is broken: {exc}")
    pooled = artifacts / "cv_pooled_metrics.json"
    if pooled.is_file() and "feasibility" not in json.loads(pooled.read_text(encoding="utf-8")):
        report.add(PROBLEM, pooled, "has no 'feasibility' block: written by pool_cv.py before v3.8.0. "
                                    "Re-run pool_cv.py with this version; it needs the five checkpoints, not a retrain")
    for csv in artifacts.glob("results_*.csv"):
        head = csv.read_text(encoding="utf-8", errors="replace").splitlines()[:2]
        if len(head) == 2 and "patient_id" in head[0].split(","):
            value = head[1].split(",")[head[0].split(",").index("patient_id")]
            if "#" in value:
                report.add(PROBLEM, csv, f"patient_id holds a case id ({value}): written before v3.3.0, "
                                         f"so a patient-clustered interval over it is a row bootstrap")


def inspect(artifacts: Path, runs: Path, localiser_runs: Path, app_config: Path,
            min_orientation: float = 0.95, device: torch.device | None = None) -> Report:
    """Everything a run should have produced, checked the way the app will check it."""
    device = device or torch.device("cpu")
    report = Report()
    folds = _fold_dirs(runs)
    if not folds:
        report.add(MISSING, runs / "cv_fold*", "no trained fold at all: there is nothing for the app to load")
    for directory in folds:
        check_site_model(report, directory, app_config, device)
    others = [d for d in sorted(runs.glob("*")) if d.is_dir() and d not in folds and (d / "best.pt").is_file()]
    for directory in others:
        check_site_model(report, directory, app_config, device, pack=False)
    loc_folds = _fold_dirs(localiser_runs)
    if not loc_folds:
        report.add(MISSING, localiser_runs / "cv_fold*",
                   "no localiser: the app can analyse a scan only when its mask is uploaded with it")
    for directory in loc_folds:
        check_localiser(report, directory, device, min_orientation)
    check_results(report, artifacts)
    # Order kept, duplicates dropped: `calibration/` is listed as a file and as a directory.
    report.files = list(dict.fromkeys(report.files))
    return report


def write_archive(report: Report, base: Path, archive: Path, env: dict) -> dict:
    """Pack every accepted file with a manifest of checksums. Returns the manifest."""
    def inside(path) -> str:
        try:
            return Path(path).resolve().relative_to(base.resolve()).as_posix()
        except ValueError:
            raise ValueError(f"{path} is not under {base}: the archive keeps the layout the app "
                             f"and the scripts expect, so every file has to sit below one folder") from None

    entries = [{"path": inside(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
               for path in report.files]
    manifest = {
        "environment": env,
        "site_models": [{**m, "checkpoint": inside(m["checkpoint"])} for m in report.site_models],
        "localisers": [{**m, "checkpoint": inside(m["checkpoint"])} for m in report.localisers],
        "missing": [{"path": f.path, "cost": f.detail} for f in report.missing],
        "files": entries,
    }
    archive.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = archive.with_name(MANIFEST)
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    with tarfile.open(archive, "w:gz", compresslevel=1) as tar:
        tar.add(manifest_path, arcname=MANIFEST)
        for path, entry in zip(report.files, entries):
            tar.add(path, arcname=entry["path"])
    manifest_path.unlink()
    return manifest


def extract_and_verify(archive: Path, out_dir: Path) -> tuple[dict, list[str]]:
    """Unpack a hand-back and recompute every checksum. Returns (manifest, faults)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tar:
        tar.extractall(out_dir, filter="data")
    manifest_path = out_dir / MANIFEST
    if not manifest_path.is_file():
        return {}, [f"{archive.name} holds no {MANIFEST}: it was not written by pack_handback.py"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    faults = []
    for entry in manifest["files"]:
        path = out_dir / entry["path"]
        if not path.is_file():
            faults.append(f"{entry['path']}: listed in the manifest and not in the archive")
        elif path.stat().st_size != entry["bytes"]:
            faults.append(f"{entry['path']}: {path.stat().st_size} bytes, the manifest says {entry['bytes']} "
                          f"-- truncated in transfer")
        elif sha256(path) != entry["sha256"]:
            faults.append(f"{entry['path']}: same size, different content -- corrupted in transfer")
    return manifest, faults
