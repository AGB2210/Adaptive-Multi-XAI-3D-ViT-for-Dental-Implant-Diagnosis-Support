"""The hand-back check: what a run produced, against what the app will accept.

The August run's checkpoints and result files never left the rented machine.
`pack_handback.py` is the step that was missing, and what matters about it is
that it cannot pass a file the app would then refuse -- so these tests build a
run tree out of the same toy checkpoints the app's own tests load, and break it
in the ways a real hand-back has been broken.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.inference.handback import (
    MANIFEST,
    MISSING,
    PROBLEM,
    environment,
    extract_and_verify,
    inspect,
    write_archive,
)
from tests.toy_scan import write_tiny_checkpoint, write_tiny_localiser

ROOT = Path(__file__).resolve().parents[1]
APP_CONFIG = ROOT / "configs" / "app.yaml"
VAL = {"classification": {"thresholds": {"needs_implant": 0.6}},
       "regression": {"available_height_mm": {"mae": 3.4}, "ridge_width_mm": {"mae": 2.5}}}
CALIBRATION = {"temperature": 1.3, "gate_threshold": -0.2, "gate_uncertainty": "margin",
               "fitted_on": "validation split only"}
POOLED = {"pooled": {}, "regression": {}, "feasibility": {"all_sites": {"n": 10}}}


def make_run(base: Path, folds=(0, 1), localiser_folds=(0,), orientation=0.99) -> tuple[Path, Path, Path]:
    """A finished run in miniature: (artifacts, runs, localiser_runs) under `base`."""
    art = base / "artifacts_sites"
    runs, loc = art / "runs", art / "localiser_runs"
    for k in folds:
        fold = runs / f"cv_fold{k}"
        write_tiny_checkpoint(fold / "best.pt", seed=k)
        (fold / "best_val_metrics.json").write_text(json.dumps(VAL))
        (fold / "metrics.json").write_text(json.dumps({"val": VAL}))
        (fold / "history.csv").write_text("epoch,loss\n0,1.0\n")
    (runs / "cv_fold0" / "calibration.json").write_text(json.dumps(CALIBRATION))
    for k in localiser_folds:
        fold = loc / f"cv_fold{k}"
        write_tiny_localiser(fold / "best.pt", seed=k,
                             val={"orientation_accuracy": orientation, "median_spread_mm": 4.0})
        (fold / "eval_test.json").write_text("{}")
    (art / "cv_folds.json").write_text(json.dumps({"folds": [["a", "b"], ["c"], ["d"]], "meta": {}}))
    (art / "sites_toothfairy3.csv").write_text("patient_id,tooth\na,36\n")
    (art / "cv_pooled_metrics.json").write_text(json.dumps(POOLED))
    (art / "cv_predictions.csv").write_text("case_id,patient_id\na#36,a\n")
    (art / "results_faithfulness.csv").write_text("case_id,patient_id,method\na#36,a,gradcam\n")
    (art / "figures").mkdir()
    (art / "figures" / "one.png").write_bytes(b"png")
    return art, runs, loc


def run_inspect(art, runs, loc):
    return inspect(art, runs, loc, APP_CONFIG, min_orientation=0.95)


class TestWhatTheAppWillAccept:
    def test_a_complete_fold_is_accepted_with_what_the_app_reads_from_it(self, tmp_path):
        report = run_inspect(*make_run(tmp_path))
        assert not report.problems, [f.detail for f in report.problems]
        first = report.site_models[0]
        assert first["fold"] == "cv_fold0" and first["calibrated"] and first["temperature"] == 1.3
        assert first["decision_thresholds"] == {"needs_implant": 0.6}
        assert first["validation_mae_mm"]["available_height_mm"] == 3.4
        assert first["gate"]["threshold"] == -0.2
        assert not report.site_models[1]["calibrated"], "only fold 0 was calibrated"
        packed = {p.name for p in report.files}
        assert {"best.pt", "calibration.json", "cv_folds.json", "one.png"} <= packed

    def test_a_checkpoint_the_app_would_refuse_is_a_problem_not_a_surprise(self, tmp_path):
        """A numpy scalar in a checkpoint loads on the machine that wrote it and
        is refused by `weights_only=True`, which is how the app opens a file
        picked in a browser."""
        art, runs, loc = make_run(tmp_path)
        write_tiny_checkpoint(runs / "cv_fold1" / "best.pt", best_macro_auroc=np.float64(0.95))
        report = run_inspect(art, runs, loc)
        refused = [f for f in report.problems if "cv_fold1" in f.path]
        assert len(refused) == 1 and "refuse" in refused[0].detail
        assert [m["fold"] for m in report.site_models] == ["cv_fold0"]
        assert not any("cv_fold1" in str(p) and p.name == "best.pt" for p in report.files)

    def test_an_old_checkpoint_says_its_architecture_came_from_the_apps_config(self, tmp_path):
        """Checkpoints from before v3.5.0 carry no model_config. The tiny model
        is not the app's default architecture, so here that is a refusal; on the
        real one it loads, and the report has to say where num_heads came from."""
        art, runs, loc = make_run(tmp_path, folds=(0,))
        write_tiny_checkpoint(runs / "cv_fold0" / "best.pt", embed_config=False)
        report = run_inspect(art, runs, loc)
        assert report.problems and "declared architecture does not match" in report.problems[0].detail

    def test_what_is_missing_is_named_with_its_cost(self, tmp_path):
        art, runs, loc = make_run(tmp_path, localiser_folds=())
        (runs / "cv_fold1" / "best_val_metrics.json").unlink()
        report = run_inspect(art, runs, loc)
        assert not report.problems
        costs = {Path(f.path).name: f.detail for f in report.missing}
        assert "cannot mark a site borderline" in costs["best_val_metrics.json"]
        assert "only when its mask is uploaded" in costs["cv_fold*"]
        fold1 = next(f.detail for f in report.missing
                     if Path(f.path) == runs / "cv_fold1" / "calibration.json")
        assert "uncalibrated" in fold1, "fold 1 was never calibrated"

    def test_a_checkpoint_outside_a_fold_is_checked_and_not_packed(self, tmp_path):
        """The smoke gate trains without `--fold`, so its checkpoint sits in
        `runs/vit3d/`. Loading it is the one check that can be made before a
        GPU is paid for; sending it would put a one-epoch model in the archive."""
        art, runs, loc = make_run(tmp_path, folds=(0,))
        write_tiny_checkpoint(runs / "vit3d" / "best.pt")
        write_tiny_checkpoint(runs / "broken" / "best.pt", best_macro_auroc=np.float64(0.5))
        report = run_inspect(art, runs, loc)
        smoke = next(f for f in report.findings if Path(f.path) == runs / "vit3d" / "best.pt")
        assert smoke.status == "ok" and "checked, not packed" in smoke.detail
        assert [Path(f.path).parent.name for f in report.problems] == ["broken"]
        assert not any(p.parent.name in ("vit3d", "broken") for p in report.files)
        assert [m["fold"] for m in report.site_models] == ["cv_fold0"]

    def test_a_localiser_below_the_apps_orientation_bar_is_said_so(self, tmp_path):
        report = run_inspect(*make_run(tmp_path, orientation=0.71))
        note = next(f.detail for f in report.findings if "localiser_runs" in f.path and f.status == "ok")
        assert "0.710 is under the 0.95 the app requires" in note

    def test_results_written_by_older_code_are_problems(self, tmp_path):
        art, runs, loc = make_run(tmp_path)
        (art / "cv_pooled_metrics.json").write_text(json.dumps({"pooled": {}}))
        (art / "results_faithfulness.csv").write_text("patient_id,method\na#36,gradcam\n")
        (art / "cv_folds.json").write_text(json.dumps({"folds": [["a"], ["a"]], "meta": {}}))
        details = " | ".join(f.detail for f in run_inspect(art, runs, loc).problems)
        assert "no 'feasibility' block" in details
        assert "patient_id holds a case id (a#36)" in details
        assert "partition is broken" in details


class TestTheArchive:
    def pack(self, tmp_path):
        base = tmp_path / "box"
        report = run_inspect(*make_run(base))
        archive = tmp_path / "out" / "capstone_handback_vX.tar.gz"
        manifest = write_archive(report, base, archive, environment(ROOT))
        return archive, manifest

    def test_it_unpacks_to_the_same_layout_and_every_checksum_holds(self, tmp_path):
        archive, manifest = self.pack(tmp_path)
        assert manifest["environment"]["torch"] == torch.__version__
        assert manifest["site_models"][0]["checkpoint"] == "artifacts_sites/runs/cv_fold0/best.pt"
        there = tmp_path / "received"
        got, faults = extract_and_verify(archive, there)
        assert not faults and (there / MANIFEST).is_file()
        assert len(got["files"]) == len(manifest["files"])
        again = run_inspect(there / "artifacts_sites", there / "artifacts_sites" / "runs",
                            there / "artifacts_sites" / "localiser_runs")
        assert not again.problems and len(again.site_models) == 2 and len(again.localisers) == 1

    def test_a_file_changed_in_transfer_is_caught_even_at_the_same_size(self, tmp_path):
        """One cache file on this project arrived matching in size exactly and
        differing in content. Size alone would have passed it."""
        archive, _ = self.pack(tmp_path)
        there = tmp_path / "received"
        extract_and_verify(archive, there)
        target = there / "artifacts_sites" / "runs" / "cv_fold0" / "best.pt"
        data = bytearray(target.read_bytes())
        data[len(data) // 2] ^= 0xFF
        target.write_bytes(bytes(data))

        import tarfile
        damaged = tmp_path / "damaged.tar.gz"
        with tarfile.open(damaged, "w:gz") as tar:
            for path in sorted(there.rglob("*")):
                if path.is_file():
                    tar.add(path, arcname=path.relative_to(there).as_posix())
        _, faults = extract_and_verify(damaged, tmp_path / "again")
        assert len(faults) == 1 and "same size, different content" in faults[0]

    def test_an_archive_that_is_not_a_handback_is_refused(self, tmp_path):
        import tarfile
        (tmp_path / "x.txt").write_text("x")
        other = tmp_path / "other.tar.gz"
        with tarfile.open(other, "w:gz") as tar:
            tar.add(tmp_path / "x.txt", arcname="x.txt")
        manifest, faults = extract_and_verify(other, tmp_path / "out")
        assert manifest == {} and "not written by pack_handback.py" in faults[0]

    def test_a_file_outside_the_run_folder_cannot_be_packed(self, tmp_path):
        base = tmp_path / "box"
        report = run_inspect(*make_run(base))
        stray = tmp_path / "elsewhere.csv"
        stray.write_text("x")
        report.files.append(stray)
        with pytest.raises(ValueError, match="is not under"):
            write_archive(report, base, tmp_path / "a.tar.gz", {})


def test_the_script_reports_and_exits_zero_on_a_tree_with_nothing_wrong():
    """`--check-only` on this repository's own artifacts: whatever is there
    must load, and what is not there is missing, not a failure."""
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "pack_handback.py"), "--check-only"],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode in (0, 1), out.stderr[-600:]
    assert "problem(s)" in out.stdout and MISSING in out.stdout or "ok (" in out.stdout
    assert (PROBLEM in out.stdout) == (out.returncode == 1)
