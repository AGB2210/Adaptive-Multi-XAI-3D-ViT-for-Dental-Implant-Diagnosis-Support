"""What the model is asked to predict, and on which cohort.

The label set belongs to the dataset, not to the code. Previously it was a
module-level constant, which meant swapping cohorts required editing source; here
it comes from the config so a new dataset is a YAML change:

    data:
      primary: toothfairy3        # trained and model-selected on this cohort
      external: mmdental          # scored once, at the end, never trained on
    task:
      labels: [implant, crown, bridge]

Producing `artifacts/labels_<dataset>.csv` is a separate step and deliberately so.
Whether the labels come from voxel masks, from radiology reports, or from clinical
notes is a property of the cohort; everything downstream only sees a CSV of
patient_id plus one binary column per label.
"""

from __future__ import annotations

from types import SimpleNamespace


def label_names_for(cfg: SimpleNamespace) -> list[str]:
    names = list(getattr(getattr(cfg, "task", SimpleNamespace()), "labels", []) or [])
    if not names:
        raise ValueError(f"{cfg.config_path} declares no task.labels; "
                         "the label set has to be stated explicitly")
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"task.labels contains duplicates: {dupes}")
    return names


def primary_dataset(cfg: SimpleNamespace) -> str:
    name = getattr(cfg.data, "primary", None)
    if not name:
        raise ValueError(f"{cfg.config_path} declares no data.primary "
                         "(the cohort to train and select on)")
    return name


def external_dataset(cfg: SimpleNamespace) -> str | None:
    """The held-out cohort, or None when the study is single-cohort.

    Scored once, at the very end. A model can look good on one hospital's scanner
    by learning its noise and field of view; this is the check that it did not.
    """
    return getattr(cfg.data, "external", None) or None
