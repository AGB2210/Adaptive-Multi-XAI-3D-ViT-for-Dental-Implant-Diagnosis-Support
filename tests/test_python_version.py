"""One supported Python, stated in five places that must not drift apart.

pyproject.toml declares it, `src/__init__.py` enforces it, start.bat looks for
it, and the two workflows install it. A version raised in one and forgotten in
another is how CI ends up green on an interpreter the code no longer accepts,
or the launcher picks one the app then refuses.
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pytest

import src

REPO = Path(__file__).resolve().parents[1]


def declared() -> tuple[int, int]:
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    major, minor = re.search(r'requires-python\s*=\s*">=(\d+)\.(\d+)"', text).groups()
    return int(major), int(minor)


def test_the_guard_enforces_what_pyproject_declares():
    assert src.MIN_PYTHON == declared()


def test_start_bat_looks_for_the_same_version():
    text = (REPO / "start.bat").read_text(encoding="ascii")
    major, minor = declared()
    assert f"sys.version_info < ({major}, {minor})" in text
    assert f"Python {major}.{minor} or newer was not found" in text


@pytest.mark.parametrize("workflow", ["ci.yml", "release.yml"])
def test_the_workflows_install_the_same_version(workflow):
    text = (REPO / ".github" / "workflows" / workflow).read_text(encoding="utf-8")
    versions = re.findall(r'python-version:\s*"?([\d.]+)"?', text)
    major, minor = declared()
    assert versions == [f"{major}.{minor}"], f"{workflow} installs {versions}"


def test_an_older_interpreter_is_refused_on_import(monkeypatch):
    major, minor = src.MIN_PYTHON
    monkeypatch.setattr(sys, "version_info", (major, minor - 1, 9, "final", 0))
    with pytest.raises(RuntimeError, match=f"Python {major}.{minor} or newer is required"):
        importlib.reload(src)
    monkeypatch.undo()
    importlib.reload(src)                      # leave the module as it was found
    assert src.MIN_PYTHON == (major, minor)


def test_the_app_names_the_version_and_not_a_missing_package():
    """`python -m app` on an old Python used to get as far as the first import
    that failed -- usually a package -- and advised installing it."""
    import subprocess

    major, minor = src.MIN_PYTHON
    code = (f"import sys, runpy; sys.version_info = ({major}, {minor - 1}, 9, 'final', 0); "
            f"sys.argv = ['app']; runpy.run_module('app', run_name='__main__')")
    done = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)
    assert done.returncode == 1
    assert f"Python {major}.{minor} or newer is required" in done.stderr
    assert "Traceback" not in done.stderr and "Missing Python package" not in done.stdout


def test_start_bat_does_not_trust_an_environment_it_has_not_checked():
    """A .venv built under the old minimum is still on disk after an update. It
    must go through the same probe as any other Python, and be rebuilt."""
    text = (REPO / "start.bat").read_text(encoding="ascii")
    assert 'call :probe ".venv\\Scripts\\python.exe"' in text
    assert 'set PY=".venv\\Scripts\\python.exe"\r\n' not in text.split(":packages")[0]
    assert "--clear" in text


def test_no_document_still_offers_the_dropped_version():
    """3.11 was supported once. A page that still says so sends someone to
    build an environment the code will refuse."""
    for name in ("README.md", "RUNBOOK.md", "requirements.txt", "requirements-app.txt",
                 "pyproject.toml", "start.bat", ".github/workflows/ci.yml",
                 ".github/workflows/release.yml"):
        text = (REPO / name).read_text(encoding="utf-8")
        assert not re.search(r"(?<![\d.])3\.11(?![\d.])|\(3, ?11\)", text), name
