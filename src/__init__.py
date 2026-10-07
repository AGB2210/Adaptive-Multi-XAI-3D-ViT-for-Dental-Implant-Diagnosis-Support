"""The project's code. Importing it checks the interpreter first.

Python 3.12 is the one version this is developed, tested and gated on. The
scripts are run from the repo root rather than installed, so `requires-python`
in pyproject.toml is never consulted by anything -- an older interpreter would
simply run, and whatever it did differently would arrive as a number. Every
script and the app import `src`, so the refusal lives here.
"""

import sys

MIN_PYTHON = (3, 12)

if sys.version_info < MIN_PYTHON:
    raise RuntimeError(
        f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer is required; this is "
        f"{sys.version_info[0]}.{sys.version_info[1]}. Create the environment "
        f"with Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} -- see RUNBOOK.md, section 2."
    )
