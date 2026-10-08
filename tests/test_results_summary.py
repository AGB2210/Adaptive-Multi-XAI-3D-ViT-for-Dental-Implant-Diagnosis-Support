"""The tables are recomputed from rows by the functions the run scripts print with.

Twice a figure in this project's documents failed when it was recomputed from
the CSV it was said to come from: once because a "patient-clustered" interval
had been clustered on case ids, once because a recomputation used the mean
where the published statistic was the median. `summarise_results.py` is that
recomputation done once, and what is pinned here is that it cannot repeat
either: its numbers are `ci_table`'s, a file with case ids in the patient column
gets no interval, and a file that is not there is named.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.inference.results_summary import SUMMARY_JSON, SUMMARY_MD, how_to_read, summarise, write
from src.xai.runner import ci_table

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("attention_rollout", "gradcam", "integrated_gradients")


def cases(n_patients: int = 6, per_patient: int = 3) -> list[tuple[str, str]]:
    return [(f"P{p:02d}#{36 + s}", f"P{p:02d}") for p in range(n_patients) for s in range(per_patient)]


def faithfulness_frame(seed: int = 0, baseline: str = "blur", score: str = "response",
                       unit: str = "mm", methods=METHODS, **kw) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for case_id, patient in cases(**kw):
        for i, method in enumerate(methods):
            rows.append({"case_id": case_id, "patient_id": patient, "method": method,
                         "target_label": "available_height_mm", "target_unit": unit,
                         "baseline_kind": baseline, "score": score,
                         "deletion_auc": rng.normal(0.2 * i, 0.05),
                         "insertion_auc": rng.normal(0.5 - 0.1 * i, 0.05)})
    return pd.DataFrame(rows)


def randomisation_frame(seed: int = 1, **kw) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for case_id, patient in cases(**kw):
        for i, method in enumerate(METHODS):
            for stage in ("head", "blocks.3", "cls_token"):
                rows.append({"case_id": case_id, "patient_id": patient, "method": method,
                             "stage": stage, "spearman_vs_intact": rng.normal(0.4 * i, 0.03)})
    return pd.DataFrame(rows)


@pytest.fixture
def folder(tmp_path):
    art = tmp_path / "artifacts_sites"
    art.mkdir()
    faithfulness_frame().to_csv(art / "results_faithfulness.csv", index=False)
    randomisation_frame().to_csv(art / "results_randomization.csv", index=False)
    return art


def strict(text: str):
    """Parse as JSON proper: a bare NaN, which Python writes and reads happily, is an error."""
    def refuse(token):
        raise ValueError(f"{token} is not JSON")
    return json.loads(text, parse_constant=refuse)


class TestTheNumbersAreTheScriptsNumbers:
    def test_a_faithfulness_table_is_ci_table_of_the_same_rows(self, folder):
        frame = pd.read_csv(folder / "results_faithfulness.csv")
        expected = ci_table(frame, "deletion_auc")
        got = pd.DataFrame(summarise(folder).data["faithfulness"]["results_faithfulness.csv"]["deletion_auc"])
        got = got.set_index("method").loc[expected.index]
        for column in ("deletion_auc", "ci_lo", "ci_hi"):
            np.testing.assert_allclose(got[column], expected[column], rtol=0, atol=1e-12)

    def test_the_statistic_is_the_median_and_the_mean_is_labelled_as_the_mean(self, folder):
        frame = pd.read_csv(folder / "results_faithfulness.csv")
        frame.loc[frame.index[0], "deletion_auc"] = 50.0   # one wild row moves a mean, not a median
        frame.to_csv(folder / "results_faithfulness.csv", index=False)
        rows = summarise(folder).data["faithfulness"]["results_faithfulness.csv"]["deletion_auc"]
        row = next(r for r in rows if r["method"] == frame.loc[frame.index[0], "method"])
        mine = frame[frame.method == row["method"]]["deletion_auc"]
        assert row["deletion_auc"] == pytest.approx(mine.median())
        assert row["mean"] == pytest.approx(mine.mean())
        assert abs(row["mean"] - row["deletion_auc"]) > 1

    def test_randomisation_reads_the_last_stage_and_only_that(self, folder):
        frame = pd.read_csv(folder / "results_randomization.csv")
        expected = ci_table(frame[frame.stage == "cls_token"], "spearman_vs_intact")
        entry = summarise(folder).data["randomisation"]["results_randomization.csv"]
        assert entry["last_stage"] == "cls_token"
        got = pd.DataFrame(entry["table"]).set_index("method").loc[expected.index]
        np.testing.assert_allclose(got["spearman_vs_intact"], expected["spearman_vs_intact"], atol=1e-12)
        assert entry["not_ordered"] == [], "three methods 0.4 apart with sd 0.03 are separated"

    def test_overlapping_methods_are_named_as_not_ordered(self, folder):
        frame = randomisation_frame()
        same = frame.method.isin(METHODS[:2]) & (frame.stage == "cls_token")
        frame.loc[same, "spearman_vs_intact"] = np.random.default_rng(3).normal(0.1, 0.2, int(same.sum()))
        frame.to_csv(folder / "results_randomization.csv", index=False)
        entry = summarise(folder).data["randomisation"]["results_randomization.csv"]
        assert entry["not_ordered"] == ["attention_rollout / gradcam"] or \
            entry["not_ordered"] == ["gradcam / attention_rollout"]


class TestWhatItRefusesToDo:
    def test_case_ids_in_the_patient_column_get_no_interval(self, folder):
        """The file `REPORT.md` C8j is about: written before v3.3.0."""
        frame = faithfulness_frame()
        frame["patient_id"] = frame["case_id"]
        frame.to_csv(folder / "results_faithfulness.csv", index=False)
        summary = summarise(folder)
        rows = summary.data["faithfulness"]["results_faithfulness.csv"]["deletion_auc"]
        assert all("ci_lo" not in r for r in rows), "a row bootstrap must not appear under this heading"
        assert any(name == "results_faithfulness.csv" for name, _ in summary.refused)
        assert "no interval" in "\n".join(summary.lines)

    def test_one_patient_is_not_evidence_of_an_ordering(self, folder):
        randomisation_frame(n_patients=1, per_patient=2).to_csv(folder / "results_randomization.csv", index=False)
        summary = summarise(folder)
        assert summary.data["randomisation"]["results_randomization.csv"]["not_ordered"] is None
        text = "\n".join(summary.lines)
        assert "No ordering is claimed" in text
        assert "the ordering above is supported" not in text

    def test_a_file_that_is_not_there_is_named_with_the_step_that_writes_it(self, folder):
        summary = summarise(folder)
        absent = dict(summary.absent)
        assert "cv_pooled_metrics.json" in absent and "pool_cv.py" in absent["cv_pooled_metrics.json"]
        assert "results_localization.csv" in absent
        assert "results_ablations.csv" in absent
        assert "results_faithfulness.csv" not in absent
        assert "localization" not in summary.data, "nothing is filled in for a file that is absent"

    def test_a_blank_results_file_costs_its_own_table_and_no_other(self, folder):
        """What `--randomization-cases 0` wrote before v3.11.0: a file with no header."""
        pd.DataFrame([]).to_csv(folder / "results_randomization.csv", index=False)
        (folder / "results_agreement.csv").write_text("", encoding="utf-8")
        (folder / "xai_sanity.csv").write_text("method\n", encoding="utf-8")
        summary = summarise(folder)
        text = "\n".join(summary.lines)
        assert "The file holds no randomisation rows." in text
        assert "The file holds no agreement rows." in text
        assert "deletion_auc" in summary.data["faithfulness"]["results_faithfulness.csv"]

    def test_an_empty_folder_is_a_list_of_what_is_missing_and_not_a_crash(self, tmp_path):
        summary = summarise(tmp_path)
        assert len(summary.absent) >= 8
        md, js = write(summary, tmp_path)
        assert md.name == SUMMARY_MD and js.name == SUMMARY_JSON
        strict(js.read_text(encoding="utf-8"))


class TestSecondRunsAndOldFiles:
    def test_a_renamed_or_tagged_second_run_gets_its_own_table(self, folder):
        faithfulness_frame(seed=5, baseline="mean", score="deviation").to_csv(
            folder / "results_faithfulness_mean_deviation.csv", index=False)
        data = summarise(folder).data["faithfulness"]
        assert list(data) == ["results_faithfulness.csv", "results_faithfulness_mean_deviation.csv"]
        assert "neither direction is founded" in data["results_faithfulness.csv"]["reading"]
        assert "restored by score=deviation" in data["results_faithfulness_mean_deviation.csv"]["reading"]

    def test_the_direction_comes_from_the_rows(self):
        assert how_to_read(faithfulness_frame(unit="probability")).startswith("Deletion AUC: lower is better")
        assert "no direction is asserted" in how_to_read(pd.DataFrame({"deletion_auc": [0.1]}))

    def test_a_pooled_file_from_before_the_feasibility_block_says_so(self, folder):
        (folder / "cv_pooled_metrics.json").write_text(json.dumps({
            "pooled": {"per_label": {"needs_implant": {"auroc": 0.8, "auroc_ci": [0.7, 0.9], "ap": 0.4,
                                                        "n_pos": 10, "prevalence": 0.1}}},
            "per_fold": [{"fold": 0, "n": 10, "macro_auroc": 0.8}], "n_cases": 10, "n_patients": 5,
            "n_folds": 1}), encoding="utf-8")
        text = "\n".join(summarise(folder).lines)
        assert "No `feasibility` block" in text and "No `regression` block" in text

    def test_the_json_is_json_even_with_undefined_values(self, folder):
        frame = randomisation_frame()
        frame.loc[frame.method == "gradcam", "spearman_vs_intact"] = np.nan   # a constant map
        frame.to_csv(folder / "results_randomization.csv", index=False)
        summary = summarise(folder)
        _, js = write(summary, folder)
        data = strict(js.read_text(encoding="utf-8"))
        table = data["randomisation"]["results_randomization.csv"]["table"]
        row = next(r for r in table if r["method"] == "gradcam")
        assert row["spearman_vs_intact"] is None and row["n_undefined"] > 0
        assert data["randomisation"]["results_randomization.csv"]["not_ordered"] is None


class TestTheCommand:
    def test_it_writes_both_files_and_prints_the_tables(self, folder):
        done = subprocess.run([sys.executable, str(ROOT / "scripts" / "summarise_results.py"),
                               "--artifacts", str(folder)], capture_output=True, text=True, cwd=ROOT)
        assert done.returncode == 0, done.stderr
        assert "Faithfulness: deletion and insertion" in done.stdout
        assert (folder / SUMMARY_MD).is_file() and (folder / SUMMARY_JSON).is_file()

    def test_a_path_that_is_not_a_folder_is_one_line_and_exit_1(self, tmp_path):
        done = subprocess.run([sys.executable, str(ROOT / "scripts" / "summarise_results.py"),
                               "--artifacts", str(tmp_path / "nowhere")], capture_output=True, text=True, cwd=ROOT)
        assert done.returncode == 1 and "is not a folder" in done.stdout
