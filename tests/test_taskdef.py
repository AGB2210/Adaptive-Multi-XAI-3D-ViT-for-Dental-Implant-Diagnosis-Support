"""The task definition comes from config, never from a module constant."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.data.taskdef import external_dataset, label_names_for, primary_dataset


def _cfg(**kw):
    data = SimpleNamespace(**kw.pop("data", {}))
    task = SimpleNamespace(**kw.pop("task", {}))
    return SimpleNamespace(data=data, task=task, config_path="test.yaml", **kw)


def test_labels_come_from_config():
    assert label_names_for(_cfg(task={"labels": ["implant", "crown"]})) == ["implant", "crown"]


def test_missing_labels_is_an_error_not_a_default():
    # A silent default here would train a model against the wrong columns.
    with pytest.raises(ValueError, match="no task.labels"):
        label_names_for(_cfg(task={}))


def test_duplicate_labels_rejected():
    with pytest.raises(ValueError, match="duplicates"):
        label_names_for(_cfg(task={"labels": ["implant", "implant"]}))


def test_primary_dataset_required():
    assert primary_dataset(_cfg(data={"primary": "toothfairy3"})) == "toothfairy3"
    with pytest.raises(ValueError, match="no data.primary"):
        primary_dataset(_cfg(data={}))


def test_external_dataset_is_optional():
    assert external_dataset(_cfg(data={"primary": "a", "external": "b"})) == "b"
    assert external_dataset(_cfg(data={"primary": "a"})) is None
    assert external_dataset(_cfg(data={"primary": "a", "external": ""})) is None
