"""Every table the paper quotes, recomputed from the files a run left behind.

The run scripts PRINT their tables and write only rows. So a table has reached
this project twice as a transcription of somebody's terminal, and twice a
transcribed figure failed when it was recomputed from the rows: once because the
intervals had been clustered on case ids (`REPORT.md` C8j), once because a
recomputation used the mean where the published statistic was the median
(C8m). This module is the recomputation, written once.

It reads files and nothing else: no checkpoint, no scan, no GPU. Point it at the
`artifacts_sites` folder of a run, or of a hand-back that `pack_handback.py
--verify` unpacked.

TWO RULES IT KEEPS, because breaking either is how the earlier errors happened:

* Every interval comes from `src.xai.runner.clustered_ci`, the function the run
  scripts call, at its defaults unless a table says otherwise: the MEDIAN, 4000
  resamples of PATIENTS, seed 1337. A file whose `patient_id` column holds case
  ids is refused by that function, and the refusal is reported in the table's
  place.
* A file that is absent is named as absent, with the step that writes it. No
  table is left out silently and none is filled from another source.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.xai.runner import ci_table, clustered_ci, has_an_interval, unordered_pairs

SUMMARY_MD = "RESULTS_SUMMARY.md"
SUMMARY_JSON = "results_summary.json"

# How every interval below was made, stated once at the top of the output.
METHOD_NOTE = ("Intervals are 95%, from 4000 resamples of PATIENTS (not rows), seed 1337, around "
               "the MEDIAN unless a column says mean. They are computed by "
               "`src.xai.runner.clustered_ci`, the function the run scripts call.")


NO_INTERVAL = ("At least one row has no usable interval: one patient, where every resample is that "
               "patient and the interval has zero width, or nothing defined.")


ONE_PATIENT = "a row rests on one patient or on nothing defined, so its interval says nothing"


@dataclass
class Summary:
    lines: list[str] = field(default_factory=list)       # the Markdown document
    data: dict = field(default_factory=dict)             # the same figures, for a program
    absent: list[tuple[str, str]] = field(default_factory=list)   # (file, the step that writes it)
    refused: list[tuple[str, str]] = field(default_factory=list)  # (file, why no interval was made)

    def heading(self, text: str, level: int = 2) -> None:
        self.lines += ["", "#" * level + " " + text, ""]

    def say(self, text: str = "") -> None:
        self.lines.append(text)

    def table(self, frame: pd.DataFrame, digits: int = 4, index: bool = True) -> None:
        self.lines += ["```", frame.round(digits).to_string(index=index), "```"]

    def missing(self, name: str, step: str) -> None:
        self.absent.append((name, step))
        self.say(f"_Not in this folder: `{name}` -- written by {step}._")


def _records(frame: pd.DataFrame) -> list[dict]:
    """A frame as plain records, with NaN as None so the JSON stays valid JSON."""
    out = frame.reset_index() if frame.index.name else frame
    return json.loads(out.to_json(orient="records", double_precision=15))


def _interval(pair) -> str:
    if pair is None or len(pair) != 2 or any(v is None for v in pair):
        return "[no interval]"
    return f"[{pair[0]:.4f}, {pair[1]:.4f}]"


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return exc


def _read(path: Path) -> pd.DataFrame:
    """A results file as a frame; an empty frame if it holds nothing a table can be made from.

    Before v3.11.0 `run_faithfulness.py --randomization-cases 0` wrote a file
    with no header at all, which `read_csv` raises on. Every section already
    says so when a frame is empty, and one unreadable file should cost its own
    table and not the other ten.
    """
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame()


def _ci(summary: Summary, name: str, frame: pd.DataFrame, column: str, group: str = "method",
        stat=np.median) -> pd.DataFrame | None:
    """`ci_table`, or None with the reason recorded when the file cannot carry one."""
    if "patient_id" not in frame:
        summary.refused.append((name, "no patient_id column, so no patient-clustered interval"))
        return None
    try:
        return ci_table(frame, column, group_col=group, stat=stat)
    except ValueError as exc:
        # clustered_ci refusing case ids in the patient column. A row bootstrap
        # under a patient-clustered heading is the error this project already
        # published once; no interval is the honest output.
        summary.refused.append((name, str(exc).split(" -- ")[0]))
        return None


def _variants(artifacts: Path, stem: str, suffix: str = ".csv") -> list[Path]:
    """`results_faithfulness.csv` and every renamed copy beside it, oldest name first."""
    return sorted(artifacts.glob(f"{stem}*{suffix}"), key=lambda p: (len(p.name), p.name))


def _settings(frame: pd.DataFrame) -> str:
    parts = []
    for column in ("target_label", "target_unit", "baseline_kind", "score"):
        if column in frame:
            parts.append(f"{column} {sorted(map(str, frame[column].dropna().unique()))}")
    return "; ".join(parts) if parts else "no settings recorded in the rows"


def how_to_read(frame: pd.DataFrame,
                usual: str = "Deletion AUC: lower is better. Insertion AUC: higher is better.") -> str:
    """Which direction is better for a deletion / insertion AUC, from the rows themselves.

    The same rule as `run_faithfulness.direction_note`, read from the file and
    not from the flags of a run. A file without the columns cannot say.
    """
    if "target_unit" not in frame or "score" not in frame:
        return "The rows record neither the target's unit nor the scoring mode, so no direction is asserted."
    units, scores = set(frame["target_unit"].dropna()), set(frame["score"].dropna())
    if units <= {"probability"}:
        return usual
    if scores == {"deviation"}:
        return usual + " (Millimetre head, restored by score=deviation.)"
    return ("A millimetre head at score='response': neither direction is founded. "
            "Compare methods; do not rank them against an absolute.")


# --------------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------------

def cross_validation(s: Summary, artifacts: Path) -> None:
    s.heading("Cross-validation, pooled")
    path = artifacts / "cv_pooled_metrics.json"
    if not path.is_file():
        s.missing(path.name, "`scripts/pool_cv.py` (RUNBOOK 4c)")
        return
    pooled = _load_json(path)
    if isinstance(pooled, Exception):
        s.say(f"`{path.name}` cannot be read: {pooled}")
        return
    s.data["cross_validation"] = pooled
    s.say(f"{pooled.get('n_cases')} sites from {pooled.get('n_patients')} patients, "
          f"{pooled.get('n_folds')} folds; every site scored by a model that never saw its patient.")

    rows = []
    for label, m in (pooled.get("pooled", {}).get("per_label") or {}).items():
        rows.append({"label": label, "n_pos": m.get("n_pos"), "prevalence": m.get("prevalence"),
                     "auroc": m.get("auroc"), "auroc_ci": _interval(m.get("auroc_ci")),
                     "ap": m.get("ap"), "ap_ci": _interval(m.get("ap_ci")),
                     "f1": m.get("f1"), "f1_ci": _interval(m.get("f1_ci"))})
    if rows:
        s.say()
        s.say("Classification. AP is read against the prevalence, which is its floor.")
        s.table(pd.DataFrame(rows), index=False)

    rows = []
    for name, m in (pooled.get("regression") or {}).items():
        floor = m.get("mae_floor")
        rows.append({"target": name, "n": m.get("n"), "mae": m.get("mae"),
                     "mae_ci": _interval(m.get("mae_ci")), "mae_floor": floor,
                     "mae/floor": (m["mae"] / floor) if floor else float("nan"),
                     "rmse": m.get("rmse"), "rmse_floor": m.get("rmse_floor")})
    if rows:
        s.say()
        s.say("Millimetre targets. The floor is the error of predicting, for every site, the "
              "median of these sites' own measured values; a ratio at or above 1 is no better than that.")
        s.table(pd.DataFrame(rows), index=False)
    else:
        s.say()
        s.say("_No `regression` block: this file was written before v3.8.0. Re-run "
              "`scripts/pool_cv.py` with the current code; it needs the five checkpoints and the cache._")

    feasibility = pooled.get("feasibility") or {}
    rows = []
    for group, m in feasibility.items():
        if not m.get("n"):
            continue
        rows.append({"sites": group.replace("_", " "), "n": m["n"],
                     "measured_feasible": m.get("measured_feasible_rate"),
                     "predicted_feasible": m.get("predicted_feasible_rate"),
                     "agreement": m.get("agreement"), "agreement_ci": _interval(m.get("agreement_ci")),
                     "feasible_when_not": m.get("called_feasible_when_not"),
                     "feasible_when_not_ci": _interval(m.get("called_feasible_when_not_ci")),
                     "infeasible_when_feasible": m.get("called_infeasible_when_feasible")})
    s.say()
    if rows:
        s.say("Feasibility at the configured rule -- the headline. `feasible_when_not` is the share "
              "of ALL the sites in the row that are called feasible and measure infeasible -- the "
              "error that matters clinically -- and not a share of the sites called feasible. "
              "`agreement` and the two error columns sum to 1.")
        s.table(pd.DataFrame(rows), index=False)
    else:
        s.say("_No `feasibility` block: the headline figure is not in this file. It is written by "
              "`scripts/pool_cv.py` since v3.8.0._")

    sweep = pooled.get("threshold_sensitivity") or []
    if sweep:
        s.say()
        s.say("Agreement across the height rule (width rule held at its configured value):")
        s.table(pd.DataFrame(sweep), index=False)

    per_fold = pooled.get("per_fold") or []
    if per_fold:
        frame = pd.DataFrame(per_fold)
        s.say()
        s.say("Per fold, test patients only:")
        s.table(frame, index=False)
        if "macro_auroc" in frame and len(frame) > 1:
            m = frame["macro_auroc"]
            s.say(f"Macro AUROC across folds: mean {m.mean():.4f}, sd {m.std(ddof=1):.4f}, "
                  f"range {m.min():.4f}-{m.max():.4f}. The pooled figure above is the one to quote; "
                  f"a mean of five fold AUROCs is a different statistic.")


def calibration(s: Summary, runs: Path) -> None:
    s.heading("Calibration, per fold")
    rows = []
    for folder in sorted(runs.glob("cv_fold*")) if runs.is_dir() else []:
        path = folder / "calibration.json"
        if not path.is_file():
            rows.append({"fold": folder.name, "temperature": float("nan")})
            continue
        c = _load_json(path)
        if isinstance(c, Exception):
            continue
        rows.append({"fold": folder.name, "temperature": c.get("temperature"),
                     "ece_before": c.get("ece_before"), "ece_after": c.get("ece_after"),
                     "val_cases": c.get("n_val_cases"), "val_patients": c.get("n_val_patients"),
                     "gate_threshold": c.get("gate_threshold"),
                     "ensemble_fraction": c.get("gate_ensemble_fraction")})
    if not rows:
        s.missing("runs/cv_fold*/calibration.json", "`scripts/run_adaptive.py` (RUNBOOK 4d)")
        return
    frame = pd.DataFrame(rows)
    s.data["calibration"] = _records(frame)
    s.say("Temperature and gate fitted on each fold's validation patients. A fold with no "
          "temperature has no `calibration.json`, and the app says so on every result from it.")
    s.table(frame, index=False)


# --------------------------------------------------------------------------
# Explainability
# --------------------------------------------------------------------------

def faithfulness(s: Summary, artifacts: Path) -> None:
    s.heading("Faithfulness: deletion and insertion")
    files = _variants(artifacts, "results_faithfulness")
    if not files:
        s.missing("results_faithfulness.csv", "`scripts/run_faithfulness.py` (RUNBOOK 4d)")
        return
    s.data["faithfulness"] = {}
    for path in files:
        frame = _read(path)
        s.heading(f"`{path.name}`", 3)
        if frame.empty or "method" not in frame:
            s.say("The file holds no faithfulness rows.")
            continue
        cases = frame["case_id"].nunique() if "case_id" in frame else len(frame)
        patients = frame["patient_id"].nunique() if "patient_id" in frame else "?"
        s.say(f"{cases} cases from {patients} patients. Settings in the rows: {_settings(frame)}.")
        s.say(how_to_read(frame))
        entry = {"cases": int(cases), "settings": _settings(frame), "reading": how_to_read(frame)}
        for column in ("deletion_auc", "insertion_auc"):
            if column not in frame:
                continue
            means = frame.groupby("method")[column].agg(["mean", "std", "count"])
            table = _ci(s, path.name, frame, column)
            s.say()
            if table is None:
                s.say(f"`{column}` -- no interval could be made (see the end of this document):")
                s.table(means)
                entry[column] = _records(means)
                continue
            merged = table.join(means)
            s.say(f"`{column}`, median with its interval, and the mean the run printed:")
            s.table(merged)
            entry[column] = _records(merged)
        s.data["faithfulness"][path.name] = entry


def randomisation(s: Summary, artifacts: Path) -> None:
    s.heading("Model randomisation")
    files = _variants(artifacts, "results_randomization")
    if not files:
        s.missing("results_randomization.csv", "`scripts/run_faithfulness.py` (RUNBOOK 4d)")
        return
    s.data["randomisation"] = {}
    s.say("Spearman correlation between a method's map on the trained model and on the model with "
          "its weights re-initialised, at the last stage of the cascade. Lower means the map "
          "depends on what the model learned. Adebayo et al. define no pass mark, so none is applied.")
    for path in files:
        frame = _read(path)
        s.heading(f"`{path.name}`", 3)
        if frame.empty or "spearman_vs_intact" not in frame:
            s.say("The file holds no randomisation rows.")
            continue
        last = frame["stage"].iloc[-1]
        tail = frame[frame["stage"] == last]
        counts = pd.DataFrame({
            "n_undefined": frame.groupby("method")["spearman_vs_intact"].apply(lambda v: int(v.isna().sum())),
            "n_stages": frame.groupby("method")["stage"].nunique(),
        })
        s.say(f"{tail['case_id'].nunique() if 'case_id' in tail else len(tail)} cases; "
              f"last stage `{last}`.")
        table = _ci(s, path.name, tail, "spearman_vs_intact")
        entry = {"last_stage": str(last)}
        if table is None:
            plain = tail.groupby("method")["spearman_vs_intact"].median().to_frame("median").join(counts)
            s.table(plain)
            entry["table"] = _records(plain)
        else:
            merged = table.join(counts)
            s.table(merged)
            entry["table"] = _records(merged)
            pairs = unordered_pairs(table)
            entry["not_ordered"] = None if pairs is None else [f"{a} / {b}" for a, b in pairs]
            if pairs is None:
                s.say(NO_INTERVAL + " No ordering is claimed.")
                s.refused.append((path.name, ONE_PATIENT))
            elif pairs:
                s.say("Intervals overlap, so these pairs are NOT ordered by the data: "
                      + "; ".join(entry["not_ordered"]) + ".")
            else:
                s.say("No intervals overlap: the ordering above is supported.")
        undefined = counts[counts["n_undefined"] > 0]
        if not undefined.empty:
            s.say(f"Undefined (a constant map, so Spearman is NaN and not 0) for: "
                  f"{undefined.index.tolist()}. Counted, not averaged away.")
        s.data["randomisation"][path.name] = entry


def agreement(s: Summary, artifacts: Path) -> None:
    s.heading("Agreement between methods")
    files = _variants(artifacts, "results_agreement")
    if not files:
        s.missing("results_agreement.csv", "`scripts/run_faithfulness.py` (RUNBOOK 4d)")
        return
    s.data["agreement"] = {}
    for path in files:
        frame = _read(path)
        s.heading(f"`{path.name}`", 3)
        if frame.empty or not {"method_a", "method_b", "spearman"} <= set(frame.columns):
            s.say("The file holds no agreement rows.")
            continue
        # One row per case and unordered pair, whichever way round the run wrote it.
        frame = frame.assign(pair=[" | ".join(sorted(p)) for p in zip(frame["method_a"], frame["method_b"])])
        frame = frame.drop_duplicates([c for c in ("case_id", "pair") if c in frame])
        table = _ci(s, path.name, frame, "spearman", group="pair")
        if table is None:
            table = frame.groupby("pair")["spearman"].median().to_frame("spearman")
        extra = [c for c in ("jaccard_top1pct", "jaccard_top5pct") if c in frame]
        if extra:
            table = table.join(frame.groupby("pair")[extra].median().add_suffix("_median"))
        s.say("Spearman correlation between two methods' maps on the same case.")
        s.table(table)
        s.data["agreement"][path.name] = _records(table)


def localisation(s: Summary, artifacts: Path) -> None:
    s.heading("Localisation against the annotated anatomy")
    files = _variants(artifacts, "results_localization")
    if not files:
        s.missing("results_localization.csv", "`scripts/run_localization.py` (RUNBOOK 4d)")
        return
    s.data["localisation"] = {}
    for path in files:
        frame = _read(path)
        s.heading(f"`{path.name}`", 3)
        if frame.empty or "enrichment" not in frame:
            s.say("The file holds no localisation rows.")
            continue
        structure = sorted(map(str, frame["structure"].unique())) if "structure" in frame else "?"
        chance = float(frame["mask_fraction"].median()) if "mask_fraction" in frame else float("nan")
        scored = int(frame["case_id"].nunique())
        entry = {"cases": scored, "structure": structure, "mask_fraction": chance}
        if "cases_selected" in frame:
            selected = int(frame["cases_selected"].iloc[0])
            absent = int(frame["cases_without_structure"].iloc[0])
            unusable = int(frame["cases_without_mask"].iloc[0])
            entry.update(cases_selected=selected, cases_without_structure=absent, cases_without_mask=unusable)
            denominator = (f"Scored on {scored} of {selected} selected cases: {absent} had no such structure "
                           f"inside the patch, {unusable} had no usable mask. Quote that fraction beside "
                           f"any figure from this table.")
        else:
            denominator = ("Cases with no such structure in the patch are not in this file, and it was "
                           "written before the count was: the run's log has it, and it belongs beside "
                           "any figure quoted from here.")
        s.say(f"{scored} cases scored; structure {structure}, which is {chance:.5%} of the patch. "
              f"Enrichment 1.0 is chance. {denominator}")
        for column, null in (("enrichment", 1.0), ("pointing_hit", 0.0)):
            if column not in frame:
                continue
            table = _ci(s, path.name, frame, column)
            s.say()
            if table is None:
                plain = frame.groupby("method")[column].median().to_frame(column)
                s.table(plain)
                entry[column] = _records(plain)
                continue
            s.say(f"`{column}`:")
            s.table(table)
            entry[column] = _records(table)
            if not has_an_interval(table):
                s.say(NO_INTERVAL + " Nothing is claimed about chance or about a best method.")
                s.refused.append((path.name, ONE_PATIENT))
                continue
            at_chance = [m for m in table.index if table.loc[m, "ci_lo"] <= null <= table.loc[m, "ci_hi"]]
            if at_chance:
                s.say(f"Interval includes {null}, so not distinguishable from "
                      f"{'chance' if null else 'never hitting'}: {at_chance}.")
            best = table.index[-1]
            tied = [m for m in table.index[:-1] if table.loc[m, "ci_hi"] >= table.loc[best, "ci_lo"]]
            if tied:
                s.say(f"`{best}` is NOT separated from {tied}: no best method here.")
        s.data["localisation"][path.name] = entry


def adaptive(s: Summary, artifacts: Path) -> None:
    s.heading("The adaptive layer")
    files = _variants(artifacts, "results_ablations")
    if not files:
        s.missing("results_ablations.csv", "`scripts/run_adaptive.py` (RUNBOOK 4d)")
    s.data["adaptive"] = {}
    for path in files:
        frame = _read(path)
        s.heading(f"`{path.name}`", 3)
        if frame.empty or "fused_eval" not in frame:
            s.say("The file holds no ablation rows.")
            continue
        metric = sorted(map(str, frame["eval_metric"].unique())) if "eval_metric" in frame else "?"
        s.say(f"{len(frame)} cases. Evaluated on {metric}. Settings in the rows: {_settings(frame)}.")
        s.say(how_to_read(frame, usual="Lower is better."))
        columns = [c for c in frame.columns
                   if c.startswith("eval_") and pd.api.types.is_numeric_dtype(frame[c])]
        rows = [{"scored": c.replace("eval_", ""), "mean": frame[c].mean(), "n": int(frame[c].notna().sum())}
                for c in sorted(columns, key=lambda c: frame[c].mean())]
        rows += [{"scored": "FUSED (agreement-weighted)", "mean": frame["fused_eval"].mean(),
                  "n": int(frame["fused_eval"].notna().sum())},
                 {"scored": "UNIFORM ensemble", "mean": frame["uniform_eval"].mean(),
                  "n": int(frame["uniform_eval"].notna().sum())}]
        s.say()
        s.say("Mean held-out score. Unequal `n` means the rows were scored on different cases "
              "and the ordering is not like for like.")
        s.table(pd.DataFrame(rows), index=False)
        entry = {"cases": int(len(frame)), "means": rows, "claims": {}}
        for claim, column in (("1: fusion beats every individual method", "beats_best_individual"),
                              ("2: agreement-weighted beats uniform", "beats_uniform")):
            if column not in frame:
                continue
            share = float(frame[column].astype(float).mean())
            interval = None
            if "patient_id" in frame and frame["patient_id"].nunique() < 2:
                s.refused.append((path.name, "one patient, so no interval on the claims"))
            elif "patient_id" in frame:
                try:
                    _, lo, hi = clustered_ci(frame.assign(**{column: frame[column].astype(float)}),
                                             column, stat=np.mean)
                    interval = [lo, hi]
                except ValueError as exc:
                    s.refused.append((path.name, str(exc).split(" -- ")[0]))
            else:
                s.refused.append((path.name, "no patient_id column, so no patient-clustered interval"))
            verdict = "" if share >= 0.5 else "  -> NOT supported; report as a negative result."
            s.say(f"Claim {claim}: {share:.1%} of cases {_interval(interval)} "
                  f"(mean, patient-clustered).{verdict}")
            entry["claims"][column] = {"share": share, "ci": interval}
        s.data["adaptive"][path.name] = entry

    for path in _variants(artifacts, "results_pareto"):
        frame = _read(path)
        s.heading(f"`{path.name}`", 3)
        if frame.empty:
            s.say("The file holds no rows.")
            continue
        s.say("Claim 3: measured compute against faithfulness as the gate sends more cases to the ensemble.")
        s.table(frame, index=False)
        s.data["adaptive"][path.name] = _records(frame)


def method_checks(s: Summary, artifacts: Path) -> None:
    s.heading("Method checks from `run_xai.py`")
    found = False
    for name, note in (("xai_runtime.csv", "Measured wall-clock per method per volume."),
                       ("xai_ig_completeness.csv", "Integrated Gradients' completeness: sum(IG) against "
                                                   "F(x) - F(baseline)."),
                       ("xai_sanity.csv", None)):
        path = artifacts / name
        if not path.is_file():
            continue
        found = True
        frame = _read(path)
        s.heading(f"`{name}`", 3)
        if frame.empty or (name == "xai_sanity.csv" and not {"method", "enrichment"} <= set(frame.columns)):
            s.say("The file holds no rows a table can be made from.")
            continue
        if name == "xai_sanity.csv":
            s.say("Saliency mass on a planted signal; enrichment at or below 1.0 is no better than chance.")
            frame = frame.groupby("method")["enrichment"].agg(["median", "mean", "count"])
            s.table(frame)
        else:
            s.say(note)
            s.table(frame, index=False)
        s.data.setdefault("method_checks", {})[name] = _records(frame)
    if not found:
        s.missing("xai_runtime.csv", "`scripts/run_xai.py` (RUNBOOK 4d)")


# --------------------------------------------------------------------------
# The two things that are not the site model
# --------------------------------------------------------------------------

def geometric_baseline(s: Summary, artifacts: Path) -> None:
    s.heading("Geometric baseline")
    files = sorted(artifacts.glob("results_geometric_baseline_fold*.json"))
    if not files:
        s.missing("results_geometric_baseline_fold<k>.json",
                  "`scripts/run_geometric_baseline.py` (RUNBOOK 4e)")
        return
    rows = []
    for path in files:
        g = _load_json(path)
        if isinstance(g, Exception):
            continue
        for name in g.get("targets", []):
            m, floor = g["regression"].get(name, {}), g.get("floors", {}).get(name, {})
            rows.append({"file": path.name, "split": g.get("split"), "n": g.get("n"), "target": name,
                         "mae": m.get("mae"), "mae_floor": floor.get("mae"),
                         "mae/floor": (m["mae"] / floor["mae"]) if m.get("mae") is not None and floor.get("mae")
                         else float("nan"),
                         "unmeasured": g.get("unmeasured")})
    frame = pd.DataFrame(rows)
    s.data["geometric_baseline"] = _records(frame)
    s.say("An untuned estimator that reads cached intensities and no mask. Compare it with the "
          "model on the SAME split: `--split val` and `--split test` are different patients.")
    s.table(frame, index=False)


def localiser(s: Summary, localiser_runs: Path) -> None:
    s.heading("Site localiser")
    files = sorted(localiser_runs.glob("cv_fold*/eval_*.json")) if localiser_runs.is_dir() else []
    if not files:
        s.missing("localiser_runs/cv_fold<k>/eval_test.json", "`scripts/eval_localiser.py` (RUNBOOK 4f)")
        return
    rows, e2e_rows, extra_rows = [], [], []
    for path in files:
        e = _load_json(path)
        if isinstance(e, Exception):
            continue
        loc = e.get("localisation", {})
        rows.append({"fold": path.parent.name, "split": e.get("split"),
                     "patients": loc.get("n_patients"), "sites": loc.get("n_sites_3d"),
                     "median_error_mm": loc.get("median_error_mm"),
                     "median_ci_mm": _interval(loc.get("median_error_ci_mm")),
                     "p90_mm": loc.get("p90_error_mm"), "within_3mm": loc.get("within_3mm"),
                     "orientation": loc.get("orientation_accuracy"), "in_view": loc.get("in_view_accuracy")})
        end = e.get("end_to_end")
        if not end:
            continue
        for group in ("predicted_sites", "all_sites"):
            for tag, placed in (("mask", "mask-placed"), ("loc", "localiser-placed")):
                r = (end.get(group) or {}).get(tag) or {}
                e2e_rows.append({"fold": path.parent.name, "sites": group.replace("_", " "),
                                 "patches": placed, "height_mae_mm": r.get("height_mae_mm"),
                                 "width_mae_mm": r.get("width_mae_mm"),
                                 "feasibility_agreement": r.get("feasibility_agreement"),
                                 "coverage": end.get("coverage")})
            extra = (end.get("extra_height_error") or {}).get(group)
            if extra:
                extra_rows.append({"fold": path.parent.name, "sites": group.replace("_", " "),
                                   "extra_height_error_mean_mm": extra.get("mean_mm"),
                                   "ci_mm": _interval(extra.get("ci_mm"))})
    frame = pd.DataFrame(rows)
    s.data["localiser"] = {"localisation": _records(frame)}
    s.say("Error of the predicted site position, on each fold's test patients.")
    s.table(frame, index=False)
    if e2e_rows:
        e2e = pd.DataFrame(e2e_rows)
        s.data["localiser"]["end_to_end"] = _records(e2e)
        s.say()
        s.say("End to end: the same site model on mask-placed and on localiser-placed patches. "
              "`coverage` is the share of test sites the app would predict; quote it beside any "
              "image-only figure. `predicted sites` are those, `all sites` is every site at face value.")
        s.table(e2e, index=False)
        if extra_rows:
            extra = pd.DataFrame(extra_rows)
            s.data["localiser"]["extra_height_error"] = _records(extra)
            s.say()
            s.say("What localising adds to the height error, per site: |localiser-placed error| minus "
                  "|mask-placed error|, as a MEAN with its patient-clustered interval.")
            s.table(extra, index=False)
    else:
        s.say("_No end-to-end block: `eval_localiser.py` was run without `--site-checkpoint`._")


# --------------------------------------------------------------------------

def _finite(value):
    """NaN and infinity as None, all the way down: `json` would write them as bare NaN."""
    if isinstance(value, dict):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def summarise(artifacts: Path, runs: Path | None = None, localiser_runs: Path | None = None) -> Summary:
    """Every table that can be made from what `artifacts` holds, and a list of what cannot."""
    artifacts = Path(artifacts)
    runs = Path(runs) if runs else artifacts / "runs"
    localiser_runs = Path(localiser_runs) if localiser_runs else artifacts / "localiser_runs"
    s = Summary()
    s.lines += ["# Results, recomputed from the files", "", f"Folder: `{artifacts}`", "", METHOD_NOTE]

    manifest = _load_json(artifacts.parent / "handback_manifest.json")
    if isinstance(manifest, dict):
        env = manifest.get("environment", {})
        s.data["produced_by"] = env
        s.say()
        s.say(f"Produced by code {env.get('code_version')} ({env.get('git_describe')}), "
              f"python {env.get('python')}, torch {env.get('torch')}, {env.get('gpu')}.")

    cross_validation(s, artifacts)
    calibration(s, runs)
    for section in (faithfulness, randomisation, agreement, localisation, adaptive, method_checks,
                    geometric_baseline):
        section(s, artifacts)
    localiser(s, localiser_runs)

    s.heading("What this summary could not include")
    if not s.absent and not s.refused:
        s.say("Nothing: every file the run list writes was found, and every interval was made.")
    for name, step in s.absent:
        s.say(f"- `{name}` is not in the folder. Written by {step}.")
    for name, reason in dict.fromkeys(s.refused):
        s.say(f"- `{name}`: no interval -- {reason}.")
    s.data["absent"] = [{"file": n, "written_by": w} for n, w in s.absent]
    s.data["no_interval"] = [{"file": n, "reason": r} for n, r in dict.fromkeys(s.refused)]
    return s


def write(summary: Summary, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    md, js = out_dir / SUMMARY_MD, out_dir / SUMMARY_JSON
    md.write_text("\n".join(summary.lines) + "\n", encoding="utf-8", newline="\n")
    js.write_text(json.dumps(_finite(summary.data), indent=1, allow_nan=False), encoding="utf-8",
                  newline="\n")
    return md, js
