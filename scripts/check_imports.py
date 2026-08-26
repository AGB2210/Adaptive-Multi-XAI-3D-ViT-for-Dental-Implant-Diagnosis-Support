"""Import every module under src/ and scripts/, so a broken import fails in CI
rather than in front of whoever runs it on a GPU box an hour into a session.

The test suite does not reach every module — visualisation and the LIME ablation
are only touched on demand — and a module that merely fails to import is the
cheapest possible bug to catch. scripts/ is included because those are the entry
points people actually run; importing one executes only its imports and function
definitions, since every script guards its body behind __main__.

    python scripts/check_imports.py
"""

from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def module_names() -> list[str]:
    names = []
    for package in ("src", "scripts"):
        for path in sorted((REPO_ROOT / package).rglob("*.py")):
            rel = path.relative_to(REPO_ROOT).with_suffix("")
            names.append(".".join(rel.parts))
    return names


def main() -> int:
    names = module_names()
    if not names:
        print("no modules found under src/ or scripts/ — wrong working directory?",
              file=sys.stderr)
        return 1

    failed = 0
    for name in names:
        try:
            importlib.import_module(name)
        except Exception:
            failed += 1
            print(f"FAILED to import {name}", file=sys.stderr)
            traceback.print_exc()

    if failed:
        print(f"\n{failed} of {len(names)} modules failed to import", file=sys.stderr)
        return 1

    print(f"{len(names)} modules imported")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
