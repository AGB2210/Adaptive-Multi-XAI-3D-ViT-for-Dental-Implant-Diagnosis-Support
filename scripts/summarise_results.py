"""Recompute every table the paper quotes from the files a run left behind.

    python scripts/summarise_results.py --artifacts path/to/artifacts_sites

For a hand-back, `--artifacts` is the `artifacts_sites` folder inside what
`pack_handback.py --verify` unpacked; that command prints this one with the path
filled in. With no `--artifacts` it reads the folder `--config` names.

It writes, into that folder unless `--out` says otherwise,

    RESULTS_SUMMARY.md       the tables, to read and to quote from
    results_summary.json     the same figures, for a program

No checkpoint, no scan and no GPU are needed, and it takes seconds: it reads
rows and recomputes. Every interval is the patient-clustered one the run scripts
print, from the same function. A file that is not there is listed with the step
that writes it, and nothing is filled in from anywhere else.

Quote from this file, not from a terminal log or a message. That is the whole
reason it exists: see the top of `src/inference/results_summary.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.results_summary import summarise, write  # noqa: E402
from src.utils.config import load_config  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--artifacts", default=None,
                    help="the artifacts_sites folder to read; default: the one --config names")
    ap.add_argument("--config", default="configs/localiser.yaml",
                    help="used only to find the folders when --artifacts is not given")
    ap.add_argument("--out", default=None, help="where to write the two files; default: --artifacts")
    args = ap.parse_args()

    def resolve(value) -> Path:
        return Path(value) if Path(value).is_absolute() else REPO / value

    if args.artifacts:
        artifacts = Path(args.artifacts)
        runs = localiser_runs = None
    else:
        cfg = load_config(resolve(args.config))
        artifacts = resolve(cfg.data.artifacts_dir)
        runs, localiser_runs = resolve(cfg.train.out_dir), resolve(cfg.localiser.out_dir)
    if not artifacts.is_dir():
        print(f"{artifacts} is not a folder.")
        return 1

    summary = summarise(artifacts, runs, localiser_runs)
    md, js = write(summary, Path(args.out) if args.out else artifacts)
    print("\n".join(summary.lines))
    print(f"\nwrote {md}\nwrote {js}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
