"""Version is authored in exactly one place: the VERSION file at the repo root.

Everything else reads it. `--check` fails if any copy disagrees, and CI runs
that on every push, because a version that lives in several files drifts apart
silently and the running code is then unable to say what it is.

    python scripts/version.py            # print the current version
    python scripts/version.py --check    # exit 1 on any mismatch
    python scripts/version.py --set 1.2.0
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = REPO_ROOT / "VERSION"
SEMVER = re.compile(r"^\d+\.\d+\.\d+$")
# Shared by the reader and the writer so they cannot drift apart.
PYPROJECT_VERSION = re.compile(r'^version\s*=\s*"[^"]+"', re.MULTILINE)


def _pyproject_version() -> str | None:
    """The version declared in pyproject.toml, or None if it declares none.

    Read with a regex rather than a TOML parser: this runs as the first CI gate,
    before anything is guaranteed installed, and it must not be the step that
    fails for want of a dependency.
    """
    path = REPO_ROOT / "pyproject.toml"
    if not path.exists():
        return None
    match = PYPROJECT_VERSION.search(path.read_text(encoding="utf-8"))
    return match.group(0).split('"')[1] if match else None


def read_version() -> str:
    if not VERSION_FILE.exists():
        raise SystemExit(f"VERSION file missing at {VERSION_FILE}")
    version = VERSION_FILE.read_text(encoding="utf-8").strip()
    if not version:
        raise SystemExit("VERSION file is empty")
    return version


def check() -> int:
    """Verify the single source of truth is well-formed and internally consistent."""
    version = read_version()
    problems: list[str] = []

    if not SEMVER.match(version):
        problems.append(f"VERSION {version!r} is not MAJOR.MINOR.PATCH")

    # Every OTHER place the version appears registers here. This dict was empty
    # while `pyproject.toml` carried `version = "0.1.0"` and VERSION carried
    # 3.4.0 -- so the CI step named "Version is the single source of truth"
    # verified only that VERSION parsed as semver, and the second copy it exists
    # to catch went unchecked through every release. A guard that names what it
    # does not do is worse than no guard, because the name is what gets trusted.
    copies: dict[str, str | None] = {"pyproject.toml": _pyproject_version()}
    for name, found in copies.items():
        if found != version:
            problems.append(f"{name} says {found!r}, VERSION says {version!r}")

    if problems:
        for p in problems:
            print(f"version mismatch: {p}", file=sys.stderr)
        return 1

    print(f"version OK: {version}")
    return 0


def set_version(new: str) -> int:
    """Write the new version everywhere it appears, so --check cannot fail after.

    A --set that updated only VERSION would leave the copies it just registered
    disagreeing, and the next CI run would fail on a bump that looked complete.
    """
    if not SEMVER.match(new):
        raise SystemExit(f"{new!r} is not MAJOR.MINOR.PATCH")
    VERSION_FILE.write_text(new + "\n", encoding="utf-8")
    print(f"VERSION set to {new}")

    path = REPO_ROOT / "pyproject.toml"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        updated, n = re.subn(PYPROJECT_VERSION, f'version = "{new}"',
                             text, count=1)
        if n:
            path.write_text(updated, encoding="utf-8")
            print(f"pyproject.toml set to {new}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="exit 1 on any mismatch")
    ap.add_argument("--set", dest="new", help="write a new version")
    args = ap.parse_args()

    if args.new:
        return set_version(args.new)
    if args.check:
        return check()
    print(read_version())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
