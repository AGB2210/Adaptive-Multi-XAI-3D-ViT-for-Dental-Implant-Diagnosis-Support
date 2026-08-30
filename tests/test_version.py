"""The version gate must check what its name claims.

The CI step is called "Version is the single source of truth". It ran
`version.py --check`, which compared VERSION against a dict of registered
copies -- and that dict was empty, while `pyproject.toml` carried
`version = "0.1.0"` against a VERSION of 3.4.0. So the step verified only that
VERSION parsed as semver, and the second copy it exists to catch went unchecked
through every release.

A guard that names what it does not do is worse than no guard, because the name
is what gets trusted. These tests assert the check has teeth.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VERSION_PY = ROOT / "scripts" / "version.py"
SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


def run(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(VERSION_PY), *args],
                          capture_output=True, text=True)


def test_version_file_is_semver():
    assert SEMVER.match((ROOT / "VERSION").read_text(encoding="utf-8").strip())


def test_check_passes_on_the_repo_as_committed():
    assert run("--check").returncode == 0


def test_pyproject_agrees_with_the_version_file():
    """The copy that was never checked."""
    from scripts.version import _pyproject_version, read_version
    assert _pyproject_version() == read_version()


def test_check_actually_compares_something():
    """Guard the guard: an empty `copies` dict passes every input.

    This is the test that would have failed while the dict was empty, and it is
    the whole point of the file -- `test_check_passes_on_the_repo` passed
    throughout the period the gate was inert.
    """
    from scripts import version
    assert version._pyproject_version() is not None, (
        "pyproject.toml declares no version, so --check has nothing to compare "
        "and the gate is inert again")


def test_check_fails_when_a_copy_disagrees(tmp_path, monkeypatch):
    """Prove the comparison bites, without touching the real files."""
    from scripts import version

    monkeypatch.setattr(version, "_pyproject_version", lambda: "0.0.1")
    assert version.check() == 1


def test_set_updates_every_copy(tmp_path, monkeypatch):
    """A bump that left a copy behind would fail CI on the next push."""
    from scripts import version

    vfile = tmp_path / "VERSION"
    vfile.write_text("1.0.0\n", encoding="utf-8")
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "x"\nversion = "1.0.0"\n', encoding="utf-8")

    monkeypatch.setattr(version, "VERSION_FILE", vfile)
    monkeypatch.setattr(version, "REPO_ROOT", tmp_path)

    assert version.set_version("2.3.4") == 0
    assert vfile.read_text(encoding="utf-8").strip() == "2.3.4"
    assert 'version = "2.3.4"' in pyproject.read_text(encoding="utf-8")


def test_set_rejects_a_non_semver():
    from scripts import version

    with pytest.raises(SystemExit):
        version.set_version("3.4")
