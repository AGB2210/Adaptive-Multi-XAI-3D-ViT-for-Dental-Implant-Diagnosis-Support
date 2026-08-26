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

    # Any future copy of the version registers here so it is checked, not trusted.
    copies: dict[str, str | None] = {}
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
    if not SEMVER.match(new):
        raise SystemExit(f"{new!r} is not MAJOR.MINOR.PATCH")
    VERSION_FILE.write_text(new + "\n", encoding="utf-8")
    print(f"VERSION set to {new}")
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
