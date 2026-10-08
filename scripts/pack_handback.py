"""Check a finished run against what the app needs, then pack it for transfer.

ON THE MACHINE THAT RAN IT, when the run list is done:

    python scripts/pack_handback.py

It loads every checkpoint the way the app's Models dialog will -- so a file the
app would refuse is found here, while fixing it costs a command -- lists what
the run list should have produced and did not, with what each absence costs,
and writes

    handback/capstone_handback_v<version>.tar.gz      everything, cache excluded
    handback/capstone_handback_v<version>.tar.gz.sha256

Send both. Nothing is packed while a file is unusable; `--allow-problems` packs
anyway, records the problems in the archive, and includes the refused
checkpoints so they can be examined.

ON THE MACHINE THAT RECEIVES IT:

    python scripts/pack_handback.py --verify path/to/capstone_handback_v<version>.tar.gz

It unpacks beside the archive, recomputes every checksum -- three uploads on
this project arrived truncated at exact powers of two, and one cache file
matched in size and differed in content -- then loads every checkpoint again
with THIS machine's torch, which is the test that matters, and prints which
files to pick in the app and the command that recomputes every table from the
rows that came back (`scripts/summarise_results.py`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.handback import (  # noqa: E402
    MISSING,
    OK,
    PROBLEM,
    environment,
    extract_and_verify,
    inspect,
    sha256,
    write_archive,
)
from src.utils.config import load_config  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def show(report, base: Path) -> None:
    def rel(path: str) -> str:
        try:
            return Path(path).resolve().relative_to(base.resolve()).as_posix()
        except ValueError:
            return path

    for status in (PROBLEM, MISSING, OK):
        rows = [f for f in report.findings if f.status == status]
        if not rows:
            continue
        print(f"\n{status} ({len(rows)})")
        print("-" * 78)
        for f in rows:
            print(f"  {rel(f.path)}")
            print(f"      {f.detail}")


def app_instructions(manifest: dict, root: Path, folds_json: Path) -> None:
    print("\n" + "=" * 78)
    print("IN THE APP  (start.bat, or python -m app --open)  ->  Models  ->  Add model")
    print("=" * 78)
    for model in manifest.get("site_models", []):
        folder = (root / model["checkpoint"]).parent
        picks = ["best.pt", *[c for c in model["companions"]
                              if c in ("best_val_metrics.json", "calibration.json")]]
        number = "".join(ch for ch in model["fold"] if ch.isdigit())
        print(f"\nSite model, {model['fold']}  (kind: Site model, fold {number})")
        print(f"  folder: {folder}")
        print(f"  pick together: {', '.join(picks)}, and {folds_json}")
        if not model["calibrated"]:
            print("  no calibration.json for this fold: the app will say every result is uncalibrated")
    for loc in manifest.get("localisers", []):
        print(f"\nLocaliser, {loc['fold']}  (kind: Localiser)")
        print(f"  pick: {root / loc['checkpoint']}")
    print("\nUse the site model and the localiser of the SAME fold: only then has neither seen "
          "that fold's test patients.")
    print("\nA model is used only once the Models list marks it Active. The first one added of each "
          "kind is; after that, 'Use' beside a model switches to it. If a model was already listed "
          "before these -- a demonstration one, say -- it is still the active one until you switch, "
          "and a scan uploaded meanwhile is judged by it. Every result names the model that made it.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/localiser.yaml",
                    help="names the artifacts, run and localiser directories")
    ap.add_argument("--app-config", dest="app_config", default="configs/app.yaml")
    ap.add_argument("--out", default="handback", help="where the archive is written")
    ap.add_argument("--allow-problems", dest="allow_problems", action="store_true",
                    help="pack even though a file is unusable")
    ap.add_argument("--check-only", dest="check_only", action="store_true",
                    help="report and stop; write nothing")
    ap.add_argument("--verify", default=None, metavar="ARCHIVE",
                    help="unpack a received archive, check it, and say what to load in the app")
    args = ap.parse_args()

    cfg = load_config(args.config if Path(args.config).is_absolute() else REPO / args.config)
    app_cfg = load_config(REPO / args.app_config)
    min_orientation = float(getattr(app_cfg.app, "min_orientation_accuracy", 0.95))

    def resolve(value) -> Path:
        return Path(value) if Path(value).is_absolute() else REPO / value

    # The three directories a run writes to, and the folder they all sit under:
    # paths inside the archive are relative to it, so it unpacks to the same
    # layout on the other machine.
    artifacts, runs, localiser_runs = (resolve(cfg.data.artifacts_dir), resolve(cfg.train.out_dir),
                                       resolve(cfg.localiser.out_dir))
    base = artifacts.parent

    if args.verify:
        archive = Path(args.verify)
        if not archive.is_file():
            print(f"{archive} is not a file.")
            return 1
        # The archive's own checksum first: it is the cheapest check, and a
        # truncated archive fails here with one clear line.
        sidecar = Path(str(archive) + ".sha256")
        if sidecar.is_file():
            expected = sidecar.read_text(encoding="utf-8").split()[0].lower()
            if sha256(archive) != expected:
                print(f"{archive.name} does not match {sidecar.name}: it changed in transfer.\n"
                      "Ask for it again, over scp or rsync and not a browser upload.")
                return 1
            print(f"{archive.name} matches {sidecar.name}.")
        else:
            print(f"no {sidecar.name} beside the archive, so the archive as a whole is not checked; "
                  "each file inside it still is.")
        stem = archive.name[:-len(".tar.gz")] if archive.name.endswith(".tar.gz") else archive.name + "_unpacked"
        root = archive.with_name(stem)
        manifest, faults = extract_and_verify(archive, root)
        if faults:
            print(f"{len(faults)} file(s) did not survive the transfer:")
            for fault in faults:
                print("  ", fault)
            print("\nAsk for the archive again, over SSH/rsync and not a browser upload, "
                  "and compare the .sha256 before unpacking.")
            return 1
        env = manifest.get("environment", {})
        print(f"{len(manifest['files'])} files, every checksum matches.")
        for known in manifest.get("problems", []):
            print(f"sent with a known problem: {known['path']}\n      {known['reason']}")
        print(f"produced by code {env.get('code_version')} ({env.get('git_describe')}), "
              f"python {env.get('python')}, torch {env.get('torch')}, {env.get('gpu')}")
        there = [root / d.relative_to(base) for d in (artifacts, runs, localiser_runs)]
        report = inspect(*there, REPO / args.app_config, min_orientation)
        show(report, root)
        if report.problems:
            print(f"\n{len(report.problems)} file(s) arrived intact and this machine's app would still "
                  f"refuse them. That is a version difference between the two machines: send the "
                  f"lines above back with `pip list`.")
            return 1
        app_instructions(manifest, root, there[0] / "cv_folds.json")
        print()
        print("=" * 78)
        print("THE TABLES, recomputed from the rows that came back -- quote from its output, not from a log")
        print("=" * 78)
        print(f'python scripts/summarise_results.py --artifacts "{there[0]}"')
        return 0

    report = inspect(artifacts, runs, localiser_runs, REPO / args.app_config, min_orientation)
    show(report, base)
    print(f"\n{len(report.problems)} problem(s), {len(report.missing)} missing, "
          f"{len(report.files)} file(s) to send.")
    if args.check_only:
        return 1 if report.problems else 0
    if report.problems and not args.allow_problems:
        print("\nNothing was packed. Fix the PROBLEM lines above, or pass --allow-problems to send "
              "what there is with the problems recorded.")
        return 1
    if not report.files:
        print("\nThere is nothing to pack.")
        return 1

    env = environment(REPO)
    archive = resolve(args.out) / f"capstone_handback_v{env['code_version']}.tar.gz"
    # With --allow-problems the refused checkpoints go too: they are the only
    # way to find out, after the rental has ended, why the app would not load them.
    manifest = write_archive(report, base, archive, env, include_refused=args.allow_problems)
    digest = sha256(archive)
    # LF whatever the platform: `sha256sum -c` on Linux reads a CRLF line as a
    # file name ending in a carriage return, and reports the archive missing.
    Path(str(archive) + ".sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8",
                                              newline="\n")
    total = sum(e["bytes"] for e in manifest["files"])
    print(f"\nwrote {archive}")
    print(f"      {archive.stat().st_size / 1e6:.1f} MB, {len(manifest['files'])} files "
          f"({total / 1e6:.1f} MB before compression)")
    print(f"      sha256 {digest}")
    print(f"\nSend {archive.name} and {archive.name}.sha256. Use scp or rsync, not a browser upload.")
    print(json.dumps(env, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
