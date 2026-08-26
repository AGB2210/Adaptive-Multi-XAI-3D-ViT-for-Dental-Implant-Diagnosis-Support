"""Makes the repo root importable so tests can `import src...` unmodified.

pyproject sets pythonpath, but conftest.py is what pytest guarantees to load
first regardless of how it is invoked -- `pytest`, `pytest tests/`, or from an
IDE that sets its own rootdir. Without it, whether the suite runs depends on the
caller's working directory.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
