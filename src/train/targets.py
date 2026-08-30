"""Mixed binary and millimetre targets, and the loss and metrics that go with them.

WHY THE MILLIMETRES ARE THE TARGET
----------------------------------
`feasible` used to be a label. It is not an observation -- it is a rule applied
to two measurements:

    feasible = available_height_mm >= 12.0 and ridge_width_mm >= 6.0

Training on it compiles 12.0 into the weights. That matters more than it sounds,
because the threshold is the single largest lever in the project. Measured on the
mandibular teeth-tier cohort, over the 709 sites that need an implant:

    height rule 10 mm -> 266 infeasible (37.5%)
    height rule 12 mm -> 390 infeasible (55.0%)
    height rule 14 mm -> 503 infeasible (70.9%)

A 2 mm revision moves a third of the answers. As a classifier that revision costs
five folds of retraining; predicting millimetres, it costs a re-score. This is the
project's own stated principle -- thresholds are configuration, never code --
applied to the model rather than only to the label builder.

It is also the more useful output. "14.3 mm of bone here" tells a clinician which
fixture to use. "No" does not, and it silently assumes a 10 mm fixture, which is
one of the softer assumptions in the whole pipeline.

WHY IT IS A MIX AND NOT A CLEAN SWITCH
--------------------------------------
`needs_implant` is occupancy -- is this socket empty -- and is genuinely binary.
There is no millimetre quantity underneath it, so it stays a classification head.
Expect it to be easy: "is there a tooth in this patch" is a simple visual task,
and a high AUROC on it is a sanity check rather than a finding. The contribution
is the feasibility half.

STANDARDISATION
---------------
The millimetre heads are trained on standardised targets. Height has sd 8.0 mm
and width 4.4 mm, so on raw millimetres height contributes ~3x the gradient of
width for the same relative error, and the two heads stop being comparable. The
scaler is fitted on the TRAINING split only and travels in the checkpoint;
predictions are converted back to millimetres before any metric is computed, so
every reported number is in millimetres regardless.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from src.utils.log import get_logger

log = get_logger("targets")


@dataclass
class TargetSpec:
    """Which output column is what, and how to get millimetres back.

    Binary columns come first, then millimetre columns. Order is fixed by the
    config and carried in the checkpoint, because a model whose head order does
    not match the evaluator's silently reports one target's score under another's
    name.
    """

    binary: list[str] = field(default_factory=list)
    millimetres: list[str] = field(default_factory=list)
    mean: np.ndarray | None = None       # per mm-target, training split
    std: np.ndarray | None = None

    @property
    def names(self) -> list[str]:
        return list(self.binary) + list(self.millimetres)

    @property
    def n_outputs(self) -> int:
        return len(self.binary) + len(self.millimetres)

    @property
    def is_hybrid(self) -> bool:
        return bool(self.millimetres)

    def slice_binary(self, a):
        return a[..., : len(self.binary)]

    def slice_mm(self, a):
        return a[..., len(self.binary):]

    def fit(self, y: np.ndarray) -> "TargetSpec":
        """Fit the standardiser on the training split, ignoring NaN."""
        if not self.millimetres:
            return self
        mm = np.asarray(self.slice_mm(y), dtype=np.float64)

        # A column with no finite value in the TRAINING split makes `nanmean`
        # return NaN, and every standardised target with it -- which surfaces as
        # a NaN loss and looks exactly like a model that will not train. C8d
        # records this project losing time to that shape of symptom once
        # already, so it is named here rather than diagnosed again.
        usable = np.isfinite(mm).sum(axis=0)
        if (usable == 0).any():
            dead = [n for n, c in zip(self.millimetres, usable) if c == 0]
            raise ValueError(
                f"no finite training values for {dead} -- the standardiser cannot "
                f"be fitted, and every target for {'that head' if len(dead) == 1 else 'those heads'} "
                f"would become NaN. Check `drop_unmeasurable` and the label build."
            )

        self.mean = np.nanmean(mm, axis=0)
        std = np.nanstd(mm, axis=0)
        # A constant column would divide by zero and make the head untrainable;
        # 1.0 leaves it in raw millimetres, which is the honest fallback.
        self.std = np.where(std > 1e-6, std, 1.0)
        return self

    def standardise(self, y: np.ndarray) -> np.ndarray:
        if not self.millimetres or self.mean is None:
            return y
        out = np.array(y, dtype=np.float32, copy=True)
        out[..., len(self.binary):] = (self.slice_mm(out) - self.mean) / self.std
        return out

    def to_millimetres(self, pred: np.ndarray) -> np.ndarray:
        """Undo standardisation. Every metric is computed after this."""
        if not self.millimetres or self.mean is None:
            return pred
        out = np.array(pred, dtype=np.float32, copy=True)
        out[..., len(self.binary):] = self.slice_mm(out) * self.std + self.mean
        return out

    def state_dict(self) -> dict:
        return {
            "binary": list(self.binary),
            "millimetres": list(self.millimetres),
            "mean": None if self.mean is None else np.asarray(self.mean).tolist(),
            "std": None if self.std is None else np.asarray(self.std).tolist(),
        }

    @classmethod
    def from_state(cls, state: dict | None) -> "TargetSpec":
        if not state:
            return cls()
        return cls(
            binary=list(state.get("binary", [])),
            millimetres=list(state.get("millimetres", [])),
            mean=None if state.get("mean") is None else np.asarray(state["mean"], dtype=np.float64),
            std=None if state.get("std") is None else np.asarray(state["std"], dtype=np.float64),
        )


def spec_from_config(cfg) -> TargetSpec:
    task = getattr(cfg, "task", None)
    return TargetSpec(
        binary=list(getattr(task, "labels", []) or []),
        millimetres=list(getattr(task, "targets_mm", []) or []),
    )


def to_report_units(out: np.ndarray, spec: TargetSpec, temperature: float = 1.0) -> np.ndarray:
    """Model outputs converted to the unit each head is actually reported in.

    Binary columns become probabilities -- calibrated, when a temperature is
    given. Millimetre columns become millimetres, by undoing the standardiser
    that travelled in the checkpoint. Nothing else is touched.

    This is one function because the same two-line mistake was made
    independently in three scripts: `sigmoid` applied to the whole output row.
    On a millimetre head that is not a rounding error. `scripts/evaluate.py`
    was corrected when the hybrid head landed and carries the comment
    explaining why; `pool_cv.py`, `run_adaptive.py` and `make_figures.py` were
    not, so the POOLED millimetre metrics -- the project's headline regression
    numbers -- were computed on sigmoid(standardised mm), a quantity in (0, 1),
    scored against a truth in millimetres. The calibration fit was handed
    millimetres as if they were binary labels, which is what drove the
    temperature to NaN.

    Call this instead of sigmoiding a row of model outputs. On the hybrid task
    there is no case where sigmoiding the whole row is right.

    A spec with no declared targets is a checkpoint from before the hybrid head.
    Those are all-binary by construction, so they are treated that way rather
    than silently returning raw logits.
    """
    if temperature is None or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(
            f"temperature must be a finite positive number, got {temperature!r}. "
            f"A NaN here means fit_temperature was handed something that is not "
            f"a binary label -- calibrate on the binary block only."
        )
    a = np.asarray(out, dtype=np.float64)
    n_bin = len(spec.binary) if spec.names else a.shape[-1]

    rep = np.array(a, copy=True)
    rep[..., :n_bin] = 1.0 / (1.0 + np.exp(-a[..., :n_bin] / temperature))
    if spec.is_hybrid:
        rep = spec.to_millimetres(rep.astype(np.float32)).astype(np.float64)
    return rep


class HybridLoss(nn.Module):
    """BCE on the binary block, Huber on the standardised millimetre block.

    Huber rather than MSE because the height distribution has a long right tail
    -- sites with 30 mm of bone are common and clinically uninteresting, and
    under MSE they would dominate the gradient over the sites near the 12 mm
    decision boundary, which are the ones that matter.

    NaN targets are masked rather than dropped. A site can have a valid
    occupancy label and an unmeasurable width, and throwing the whole row away
    would discard the half that is fine.
    """

    def __init__(self, spec: TargetSpec, pos_weight=None, mm_weight: float = 1.0,
                 huber_delta: float = 1.0):
        super().__init__()
        self.spec = spec
        self.mm_weight = float(mm_weight)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if spec.binary else None
        self.huber = nn.HuberLoss(reduction="none", delta=float(huber_delta))
        # THE LOSS STANDARDISES, because nothing upstream does. The dataset
        # yields raw millimetres -- which is right, since that is what the CSV
        # holds and what every metric reports -- so if the comparison here were
        # also raw, the model would learn to emit millimetres and `predict`
        # would then un-standardise them a second time. That is not a subtle
        # failure: training loss fell from 14.0 to 2.9 while validation MAE rose
        # to 54 mm on a target whose whole range is 17-26 mm.
        mean = torch.tensor(spec.mean if spec.mean is not None else [0.0], dtype=torch.float32)
        std = torch.tensor(spec.std if spec.std is not None else [1.0], dtype=torch.float32)
        self.register_buffer("mm_mean", mean)
        self.register_buffer("mm_std", std)

    def forward(self, out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n_bin = len(self.spec.binary)
        total = out.new_zeros(())

        if n_bin:
            total = total + self.bce(out[:, :n_bin], y[:, :n_bin])

        if self.spec.millimetres:
            pred, true = out[:, n_bin:], y[:, n_bin:]
            true = (true - self.mm_mean) / self.mm_std
            seen = torch.isfinite(true)
            if seen.any():
                per = self.huber(pred, torch.nan_to_num(true))
                total = total + self.mm_weight * (per * seen).sum() / seen.sum()
        return total


def regression_metrics(y_true_mm: np.ndarray, y_pred_mm: np.ndarray,
                       names: list[str]) -> dict:
    """MAE and RMSE in millimetres, against the floor of predicting the median.

    The floor is the same discipline the BCE floor enforces: a model that has
    learned nothing still scores something, and on these targets that something
    is MAE 6.91 mm for height and 3.58 mm for width. A result quoted without it
    is unreadable.
    """
    out = {}
    for i, name in enumerate(names):
        t, p = np.asarray(y_true_mm)[:, i], np.asarray(y_pred_mm)[:, i]
        seen = np.isfinite(t) & np.isfinite(p)
        if not seen.any():
            out[name] = {"mae": float("nan"), "rmse": float("nan"),
                         "mae_floor": float("nan"), "n": 0}
            continue
        t, p = t[seen], p[seen]
        out[name] = {
            "mae": float(np.abs(t - p).mean()),
            "rmse": float(np.sqrt(((t - p) ** 2).mean())),
            "mae_floor": float(np.abs(t - np.median(t)).mean()),
            "rmse_floor": float(t.std()),
            "n": int(seen.sum()),
        }
    return out


def no_information_regression(y_mm: np.ndarray, names: list[str]) -> dict:
    """What a model that has learned nothing scores, per millimetre target."""
    y = np.asarray(y_mm, dtype=np.float64)
    return regression_metrics(y, np.tile(np.nanmedian(y, axis=0), (len(y), 1)), names)


def derived_feasible(pred_mm: np.ndarray, names: list[str], rules: dict,
                     jaw: str = "lower") -> np.ndarray:
    """Apply the clinical rules to predicted millimetres.

    This is the whole point of regression: `feasible` is recovered here, at
    inference, from configuration -- so revising a threshold is a re-score rather
    than a retrain. Any threshold can be passed, including ones the model never
    saw, which is what makes the sensitivity analysis possible at all.

    AN UNMEASURABLE SITE RETURNS NaN, NOT ZERO. `nan >= 12.0` is `False`, so a
    measurement that could not be made used to come out of here as a confident
    "not feasible". Two consequences, and the second is worse than the first: an
    infeasible rate computed over failures, and -- when truth and prediction were
    both unmeasurable -- two failures counted as an agreement. The geometric
    baseline returns NaN for about 20% of sites on purpose, and
    `run_geometric_baseline.py` prints that count immediately before feeding the
    same array to `threshold_sensitivity`, which silently disagreed with it.
    """
    idx = {n: i for i, n in enumerate(names)}
    ok = np.ones(len(pred_mm), dtype=bool)
    usable = np.ones(len(pred_mm), dtype=bool)
    height_rule = ("min_height_mandible_mm" if jaw == "lower" else "min_height_maxilla_mm")
    if "available_height_mm" in idx and height_rule in rules:
        col = pred_mm[:, idx["available_height_mm"]]
        ok &= col >= float(rules[height_rule])
        usable &= np.isfinite(col)
    if "ridge_width_mm" in idx and "min_width_mm" in rules:
        col = pred_mm[:, idx["ridge_width_mm"]]
        ok &= col >= float(rules["min_width_mm"])
        usable &= np.isfinite(col)
    out = ok.astype(np.float32)
    out[~usable] = np.nan
    return out


def threshold_sensitivity(true_mm: np.ndarray, pred_mm: np.ndarray, names: list[str],
                          rules: dict, sweep=(10.0, 11.0, 12.0, 13.0, 14.0),
                          jaw: str = "lower") -> list[dict]:
    """Agreement between predicted and measured feasibility, across thresholds.

    Report this instead of a single number. It is strictly more informative than
    any one threshold, and it is only possible because the model predicts
    millimetres -- which is the argument for doing so, made visible.

    `n` is the number of sites on which BOTH sides could be measured, and it can
    be well below `len(true_mm)` for an estimator that declines to answer. Quote
    it beside the agreement or the comparison is not like for like.
    """
    rows = []
    key = "min_height_mandible_mm" if jaw == "lower" else "min_height_maxilla_mm"
    for t in sweep:
        r = dict(rules)
        r[key] = t
        want = derived_feasible(true_mm, names, r, jaw)
        got = derived_feasible(pred_mm, names, r, jaw)
        seen = np.isfinite(want) & np.isfinite(got)
        rows.append({
            "height_threshold_mm": float(t),
            "measured_feasible_rate": float(want[seen].mean()) if seen.any() else float("nan"),
            "predicted_feasible_rate": float(got[seen].mean()) if seen.any() else float("nan"),
            "agreement": float((want[seen] == got[seen]).mean()) if seen.any() else float("nan"),
            "n": int(seen.sum()),
        })
    return rows


def format_regression(metrics: dict, title: str = "") -> str:
    """One table, millimetres, floor beside every value."""
    lines = []
    if title:
        lines += ["", title.upper(), "=" * max(len(title), 62)]
    lines.append(f"{'target':24}{'n':>6}{'MAE':>9}{'floor':>9}{'RMSE':>9}{'floor':>9}")
    lines.append("-" * 66)
    for name, m in metrics.items():
        lines.append(
            f"{name:24}{m['n']:>6}{m['mae']:>9.3f}{m.get('mae_floor', float('nan')):>9.3f}"
            f"{m['rmse']:>9.3f}{m.get('rmse_floor', float('nan')):>9.3f}")
    lines.append("-" * 66)
    lines.append("MAE below the floor is the only evidence the head learned anything.")
    return "\n".join(lines)


def validation_skill(y_true: np.ndarray, out: np.ndarray, spec: TargetSpec,
                     binary_metrics: dict | None = None) -> tuple[float, dict]:
    """One number to select a checkpoint on, when the heads are not commensurable.

    Macro AUROC cannot rank a hybrid model -- it says nothing about the
    millimetre heads -- and MAE cannot either, for the mirror reason. Selecting
    on the binary head alone would pick the checkpoint that is best at the EASY
    half of the task ("is there a tooth here"), which is not the half the project
    is about.

    So each head is converted to SKILL: how far it has moved from its own
    no-information floor toward perfect.

        binary       (AUROC - 0.5) / 0.5      0 = chance,  1 = perfect
        millimetre   1 - MAE / MAE_floor      0 = floor,   1 = exact

    Both are 0 for a useless model and 1 for a perfect one, so the mean is
    meaningful. Skill can go negative, which is informative rather than a bug: a
    head predicting worse than its own floor should drag the score down.

    A HEAD THAT PRODUCES NO NUMBER IS DROPPED, AND THAT IS REPORTED. If a
    millimetre head's MAE or floor comes back non-finite it cannot contribute,
    and the mean silently becomes the binary-only skill -- which is exactly the
    selection this function exists to prevent, since the binary head is the easy
    one. `skill_parts` names what actually went in, and `skill_heads_missing`
    names what did not, so a checkpoint chosen on half the objective is visible
    in `metrics.json` rather than inferable from a count.
    """
    parts: dict[str, float] = {}

    n_bin = len(spec.binary)
    if n_bin and binary_metrics:
        for name in spec.binary:
            auroc = binary_metrics.get("per_label", {}).get(name, {}).get("auroc", float("nan"))
            if np.isfinite(auroc):
                parts[name] = (auroc - 0.5) / 0.5

    if spec.millimetres:
        true_mm, pred_mm = y_true[:, n_bin:], out[:, n_bin:]
        got = regression_metrics(true_mm, pred_mm, spec.millimetres)
        for name, m in got.items():
            floor = m.get("mae_floor", float("nan"))
            if np.isfinite(m["mae"]) and np.isfinite(floor) and floor > 1e-9:
                parts[name] = 1.0 - m["mae"] / floor

    skill = float(np.mean(list(parts.values()))) if parts else float("nan")

    # Not raised: a head can legitimately be undefined on a degenerate split and
    # refusing to score would stop a run mid-training. But the mean above is then
    # over a DIFFERENT objective than the config asked for, so the caller is told
    # which heads did not contribute rather than left to infer it from a count.
    missing = [n for n in list(spec.binary) + list(spec.millimetres) if n not in parts]
    if missing:
        log.warning("validation skill computed WITHOUT %s -- selecting on a "
                    "partial objective", ", ".join(missing))
    return skill, parts, missing
