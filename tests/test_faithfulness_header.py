"""The direction line printed above the deletion/insertion table.

This exists because that one line raised `UnboundLocalError` on every
`--only-randomization` run: it read the loop variable `target`, and that flag
sets the loop's iterable to empty, so the name was never assigned. The command
the RUNBOOK tells an operator to use was the one command that could not run.

The fault is not reachable from a unit test as long as the decision lives
inside `main` beside a checkpoint load, a CUDA device and a dataset. So it was
moved into a pure function, and this file pins its behaviour -- including the
case that crashed, which is simply calling it at all without a case loop having
run first.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    """Import the script by path; `scripts/` is not a package."""
    path = ROOT / "scripts" / "run_faithfulness.py"
    spec = importlib.util.spec_from_file_location("run_faithfulness_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


direction_note = _load().direction_note

USUAL = "deletion AUC: LOWER is better    insertion AUC: HIGHER is better"


class TestProbabilityTarget:
    """One binary head, so the metric's own assumption holds."""

    @pytest.mark.parametrize("score", ["response", "deviation"])
    def test_the_usual_reading_is_stated_whatever_the_score(self, score):
        assert direction_note(0, 1, score, "needs_implant") == USUAL

    def test_every_binary_column_reads_the_same_way(self):
        for target in range(3):
            assert direction_note(target, 3, "response", f"label{target}") == USUAL


class TestMillimetreTarget:
    """The site task: one binary head, then two millimetre heads."""

    def test_response_mode_refuses_to_name_a_direction(self):
        note = direction_note(1, 1, "response", "available_height_mm")
        assert "LOWER is better" not in note
        assert "neither direction is founded" in note
        assert "available_height_mm" in note, "name the head the reader is looking at"

    def test_deviation_mode_restores_it_and_says_why(self):
        note = direction_note(1, 1, "deviation", "available_height_mm")
        assert note.startswith(USUAL)
        assert "deviation" in note, "state what restored the reading, not just that it holds"

    def test_the_two_modes_disagree(self):
        """If these ever coincide the guard has stopped guarding anything."""
        assert (direction_note(1, 1, "response", "h")
                != direction_note(1, 1, "deviation", "h"))


class TestTheCrashItself:
    def test_it_needs_no_case_loop_to_have_run(self):
        """The regression. Under `--only-randomization` no case is loaded, so
        nothing derived from a prediction may be required to print this line.

        `direction_note` takes only ints and strings the caller already holds
        before the loop, so there is no way to reach it with an unbound name.
        """
        assert direction_note(1, 1, "response", "available_height_mm")

    def test_it_is_a_pure_function_of_its_arguments(self):
        first = direction_note(1, 1, "response", "available_height_mm")
        second = direction_note(1, 1, "response", "available_height_mm")
        assert first == second

    def test_header_target_does_not_come_from_the_skipped_loop(self):
        """`main` must bind the header's target from the config and spec, not
        from a per-case prediction. Pinning the call site, since that is the
        half of the fix a refactor could silently undo.
        """
        source = (ROOT / "scripts" / "run_faithfulness.py").read_text(encoding="utf-8")
        assert "header_target = explanation_target(cfg, spec)" in source
        assert "direction_note(header_target" in source
        before = source.index("header_target = explanation_target")
        loop = source.index("for case_index, pid in enumerate(todo):")
        assert before < loop, "the header's target must be bound before the loop"


class TestTheAdaptiveSummaryReadsItsRows:
    """`run_adaptive.py` had the same fault one script over: its summary read
    the per-case loop's `target`, and `--from-csv` -- the flag that exists so
    an hour of compute is not lost to a reporting bug -- skips that loop."""

    @staticmethod
    def note():
        path = ROOT / "scripts" / "run_adaptive.py"
        spec = importlib.util.spec_from_file_location("run_adaptive_under_test", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.direction_note

    def test_a_probability_target_reads_the_usual_way(self):
        assert self.note()(["probability"] * 3, ["response"] * 3) == "lower is better"

    def test_a_millimetre_head_has_no_founded_direction_at_the_default_score(self):
        assert "not founded" in self.note()(["mm", "mm"], ["response", "response"])

    def test_deviation_restores_the_direction_for_a_millimetre_head(self):
        note = self.note()(["mm"], ["deviation"])
        assert note.startswith("lower is better") and "deviation" in note

    def test_the_summary_no_longer_reads_the_loop_variable(self):
        source = (ROOT / "scripts" / "run_adaptive.py").read_text(encoding="utf-8")
        summary = source[source.index("# ---- report the three claims"):]
        assert "direction_note(" in summary
        assert "target <" not in summary, "the per-case `target` is unbound under --from-csv"
