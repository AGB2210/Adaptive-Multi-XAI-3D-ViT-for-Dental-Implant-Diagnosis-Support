"""Mixed binary and millimetre heads: the spec, the loss, and the metrics.

The reason `feasible` stopped being a label is in `src/train/targets.py`. What
this file guards is the machinery, and in particular the one failure that looked
exactly like a model that would not train.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.taskdef import all_target_names, regression_names_for
from src.models.vit3d import ViT3D
from src.train.targets import (
    HybridLoss,
    TargetSpec,
    derived_feasible,
    no_information_regression,
    regression_metrics,
    spec_from_config,
    threshold_sensitivity,
    to_report_units,
    validation_skill,
)
from src.utils.config import load_config

MM = ["available_height_mm", "ridge_width_mm"]


def sample(n=200, seed=0):
    rng = np.random.default_rng(seed)
    binary = (rng.random(n) < 0.3).astype(np.float32)
    height = 12.0 + 8.0 * binary + rng.normal(0, 1.0, n)
    width = 8.0 + 3.0 * binary + rng.normal(0, 0.5, n)
    return np.stack([binary, height, width], axis=1).astype(np.float32)


class TestSpec:
    def test_standardise_round_trips(self):
        y = sample()
        spec = TargetSpec(binary=["b"], millimetres=MM).fit(y)
        assert np.allclose(spec.to_millimetres(spec.standardise(y)), y, atol=1e-3)

    def test_the_binary_column_is_never_touched(self):
        y = sample()
        spec = TargetSpec(binary=["b"], millimetres=MM).fit(y)
        assert np.array_equal(spec.standardise(y)[:, 0], y[:, 0])

    def test_it_survives_a_checkpoint_round_trip(self):
        """The standardiser has to travel with the weights. Without it a reloaded
        model returns standardised units that look like millimetres -- every
        prediction near zero, every MAE near the target's mean, nothing raised."""
        y = sample()
        spec = TargetSpec(binary=["b"], millimetres=MM).fit(y)
        back = TargetSpec.from_state(spec.state_dict())
        assert back.binary == spec.binary and back.millimetres == spec.millimetres
        assert np.allclose(back.to_millimetres(back.standardise(y)), y, atol=1e-3)

    def test_a_constant_column_does_not_divide_by_zero(self):
        y = np.column_stack([np.ones(20), np.full(20, 7.0), np.arange(20.0)]).astype(np.float32)
        spec = TargetSpec(binary=["b"], millimetres=MM).fit(y)
        assert np.isfinite(spec.standardise(y)).all()

    def test_a_pure_classification_spec_is_a_no_op(self):
        """The superseded detection path must be untouched by any of this."""
        spec = TargetSpec(binary=["implant", "crown", "bridge"])
        y = np.ones((5, 3), np.float32)
        assert not spec.is_hybrid
        assert np.array_equal(spec.standardise(y), y)
        assert np.array_equal(spec.to_millimetres(y), y)

    def test_nan_targets_do_not_poison_the_scaler(self):
        y = sample()
        y[::7, 1] = np.nan
        spec = TargetSpec(binary=["b"], millimetres=MM).fit(y)
        assert np.isfinite(spec.mean).all() and np.isfinite(spec.std).all()


class TestHybridLoss:
    def spec_and_loss(self, y):
        spec = TargetSpec(binary=["b"], millimetres=MM).fit(y)
        return spec, HybridLoss(spec)

    def test_the_loss_standardises_its_own_targets(self):
        """THE BUG THIS FILE EXISTS FOR. The dataset yields raw millimetres and
        `predict` un-standardises, so if the loss compared raw to raw the
        transform would be applied once in the wrong direction. Training loss
        fell 14.0 -> 2.9 while validation MAE rose to 54 mm on a target whose
        range is 17-26 mm: a model that looked like it was learning and was
        diverging.

        A prediction of ZERO in standardised space is the target's mean. So the
        loss at zero must be small, and the loss at the raw millimetre value
        must be large -- the exact opposite of the broken version.
        """
        y = torch.tensor(sample(64), dtype=torch.float32)
        spec, loss = self.spec_and_loss(y.numpy())
        at_mean = loss(torch.zeros(64, 3), y)
        at_raw = loss(y.clone(), y)
        assert at_mean < at_raw, "the loss is comparing raw millimetres to standardised output"

    def test_perfect_standardised_predictions_score_near_zero(self):
        y = sample(64)
        spec, loss = self.spec_and_loss(y)
        out = torch.tensor(spec.standardise(y), dtype=torch.float32)
        out[:, 0] = torch.where(out[:, 0] > 0.5, 20.0, -20.0)      # confident logits
        assert float(loss(out, torch.tensor(y))) < 0.05

    def test_nan_millimetres_are_masked_not_dropped(self):
        """A site can have a valid occupancy label and an unmeasurable width.
        Discarding the row would throw away the half that is fine."""
        y = sample(32)
        y[:8, 2] = np.nan
        spec, loss = self.spec_and_loss(sample(32))
        value = loss(torch.zeros(32, 3), torch.tensor(y))
        assert torch.isfinite(value)

    def test_all_nan_millimetres_still_returns_a_finite_loss(self):
        y = sample(16)
        y[:, 1:] = np.nan
        spec, loss = self.spec_and_loss(sample(16))
        assert torch.isfinite(loss(torch.zeros(16, 3), torch.tensor(y)))

    def test_a_binary_only_spec_reduces_to_bce(self):
        spec = TargetSpec(binary=["a", "b"])
        loss = HybridLoss(spec)
        y = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        bce = torch.nn.BCEWithLogitsLoss()(torch.zeros(2, 2), y)
        assert float(loss(torch.zeros(2, 2), y)) == pytest.approx(float(bce), abs=1e-6)


class TestItActuallyLearns:
    """The gate. A head that cannot fit a trivially learnable millimetre target
    on synthetic data will not fit a real one, and every number downstream would
    be reported against a floor it never beat."""

    def test_a_regression_head_beats_its_own_floor(self):
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        n, size = 128, 16

        # One bright cube whose intensity IS the millimetre target.
        depth = rng.uniform(6.0, 30.0, n).astype(np.float32)
        x = rng.normal(0, 0.1, (n, 1, size, size, size)).astype(np.float32)
        for i, d in enumerate(depth):
            x[i, 0, 6:10, 6:10, 6:10] += d / 10.0
        y = np.stack([(depth > 18).astype(np.float32), depth], axis=1)

        spec = TargetSpec(binary=["deep"], millimetres=["depth_mm"]).fit(y)
        model = ViT3D(img_size=size, stem_channels=8, embed_dim=32, patch_size=2,
                      depth=2, num_heads=2, drop_path=0.0, num_classes=2)
        loss_fn = HybridLoss(spec)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

        xt, yt = torch.from_numpy(x), torch.from_numpy(y)
        for _ in range(60):
            opt.zero_grad()
            loss_fn(model(xt), yt).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            out = model(xt).numpy()
        pred_mm = spec.to_millimetres(out)[:, 1:]
        got = regression_metrics(y[:, 1:], pred_mm, ["depth_mm"])["depth_mm"]

        assert got["mae"] < got["mae_floor"] * 0.6, (
            f"MAE {got['mae']:.2f} mm did not beat 60% of the "
            f"{got['mae_floor']:.2f} mm floor -- the head is not learning")


class TestMetrics:
    def test_the_floor_is_the_median_predictor(self):
        y = sample()[:, 1:]
        floor = no_information_regression(y, MM)
        assert floor["available_height_mm"]["mae"] > 0
        exact = regression_metrics(y, y, MM)
        assert exact["available_height_mm"]["mae"] == pytest.approx(0.0, abs=1e-6)

    def test_nan_rows_are_excluded_from_the_metric(self):
        y = sample(50)[:, 1:]
        p = y.copy()
        y[:10, 0] = np.nan
        got = regression_metrics(y, p, MM)
        assert got["available_height_mm"]["n"] == 40

    def test_skill_is_zero_at_the_floor_and_one_when_exact(self):
        y = sample()
        spec = TargetSpec(binary=["b"], millimetres=MM).fit(y)
        exact, _ = validation_skill(y, y.copy(), spec, {"per_label": {"b": {"auroc": 1.0}}})
        assert exact == pytest.approx(1.0, abs=1e-3)

        floored = y.copy()
        floored[:, 1:] = np.median(y[:, 1:], axis=0)
        at_floor, _ = validation_skill(y, floored, spec, {"per_label": {"b": {"auroc": 0.5}}})
        assert at_floor == pytest.approx(0.0, abs=1e-3)

    def test_skill_goes_negative_below_the_floor(self):
        """Informative, not a bug: a head worse than its own floor should drag
        the selection score down rather than being clipped away."""
        y = sample()
        spec = TargetSpec(binary=["b"], millimetres=MM).fit(y)
        bad = y.copy()
        bad[:, 1:] += 50.0
        skill, _ = validation_skill(y, bad, spec, {"per_label": {"b": {"auroc": 0.5}}})
        assert skill < -1.0


class TestFeasibilityIsRecoveredAtInference:
    RULES = {"min_height_mandible_mm": 12.0, "min_height_maxilla_mm": 10.0, "min_width_mm": 6.0}

    def test_the_rule_reproduces_the_label(self):
        mm = np.array([[20.0, 8.0], [10.0, 8.0], [20.0, 4.0], [11.9, 5.9]], dtype=np.float32)
        got = derived_feasible(mm, MM, self.RULES)
        assert got.tolist() == [1.0, 0.0, 0.0, 0.0]

    def test_a_threshold_the_model_never_saw_can_be_applied(self):
        """The entire argument for regressing millimetres. Revising 12 mm to
        10 mm is a re-score here; as a classifier it is five folds of retraining
        and, measured on the real cohort, it moves a third of the answers."""
        mm = np.array([[11.0, 8.0]], dtype=np.float32)
        assert derived_feasible(mm, MM, self.RULES)[0] == 0.0
        relaxed = {**self.RULES, "min_height_mandible_mm": 10.0}
        assert derived_feasible(mm, MM, relaxed)[0] == 1.0

    def test_the_maxilla_uses_its_own_rule(self):
        mm = np.array([[11.0, 8.0]], dtype=np.float32)
        assert derived_feasible(mm, MM, self.RULES, jaw="upper")[0] == 1.0
        assert derived_feasible(mm, MM, self.RULES, jaw="lower")[0] == 0.0

    def test_the_sweep_reports_agreement_at_every_threshold(self):
        y = sample(120)[:, 1:]
        rows = threshold_sensitivity(y, y.copy(), MM, self.RULES)
        assert len(rows) == 5
        for r in rows:
            assert r["agreement"] == pytest.approx(1.0)      # exact predictions
            assert r["n"] == 120

    def test_the_sweep_shows_the_threshold_moving_the_answer(self):
        y = sample(200)[:, 1:]
        rows = threshold_sensitivity(y, y.copy(), MM, self.RULES)
        rates = [r["measured_feasible_rate"] for r in rows]
        assert rates == sorted(rates, reverse=True), "a stricter rule must not pass more sites"
        assert rates[0] > rates[-1], "the threshold does not move the answer at all"


class TestConfigWiring:
    def test_the_site_config_declares_the_hybrid_head(self):
        cfg = load_config("configs/sites.yaml")
        spec = spec_from_config(cfg)
        assert spec.binary == ["needs_implant"]
        assert spec.millimetres == MM
        assert spec.is_hybrid

    def test_num_classes_counts_every_head(self):
        """Sizing the head from the binary block alone silently drops the
        regression outputs, and nothing raises."""
        cfg = load_config("configs/sites.yaml")
        assert cfg.model.num_classes == len(all_target_names(cfg)) == 3

    def test_the_superseded_task_declares_no_millimetres(self):
        cfg = load_config("configs/default.yaml")
        assert regression_names_for(cfg) == []
        assert not spec_from_config(cfg).is_hybrid

    def test_the_smoke_config_matches_the_real_one(self):
        real, smoke = load_config("configs/sites.yaml"), load_config("configs/sites_smoke.yaml")
        assert all_target_names(smoke) == all_target_names(real)


class TestReportUnits:
    """`to_report_units` -- the one conversion three scripts each got wrong.

    Each of `pool_cv.py`, `run_adaptive.py` and `make_figures.py` independently
    wrote `sigmoid(logits)` across the whole output row. On the binary block
    that is right. On a millimetre head it maps an 18 mm prediction to 0.9999
    and then scores it against a truth of 18, so the pooled MAE reported
    approximately the mean of the target and called it the model's error.
    """

    def spec(self):
        s = TargetSpec(binary=["needs_implant"], millimetres=MM)
        return s.fit(sample())

    def test_millimetres_come_back_as_millimetres(self):
        spec = self.spec()
        y = sample(n=50, seed=3)
        standardised = spec.standardise(y)
        # The model's job is to emit the standardised value; a perfect model
        # emits exactly this, and reporting must return the original mm.
        report = to_report_units(standardised, spec)
        assert np.allclose(report[:, 1:], y[:, 1:], atol=1e-4)

    def test_the_binary_column_becomes_a_probability(self):
        spec = self.spec()
        out = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        report = to_report_units(out, spec)
        assert report[0, 0] == pytest.approx(0.5)

    def test_no_millimetre_output_is_squashed_into_the_unit_interval(self):
        """The actual regression: every mm column landing inside (0, 1)."""
        spec = self.spec()
        report = to_report_units(spec.standardise(sample(n=50, seed=4)), spec)
        assert report[:, 1:].max() > 1.0, (
            "millimetre outputs are all inside (0, 1) -- they have been "
            "sigmoided, which is the bug this function exists to prevent"
        )

    def test_temperature_calibrates_only_the_binary_block(self):
        spec = self.spec()
        y = sample(n=30, seed=5)
        cold = to_report_units(spec.standardise(y), spec, temperature=1.0)
        warm = to_report_units(spec.standardise(y), spec, temperature=2.0)
        assert not np.allclose(cold[:, 0], warm[:, 0])
        assert np.allclose(cold[:, 1:], warm[:, 1:])

    def test_a_non_finite_temperature_is_refused(self):
        """NaN in, exception out. Silently propagating it is how the gate died."""
        spec = self.spec()
        for bad in (float("nan"), 0.0, -1.0):
            with pytest.raises(ValueError, match="temperature"):
                to_report_units(spec.standardise(sample(n=4)), spec, temperature=bad)

    def test_a_legacy_checkpoint_is_treated_as_all_binary(self):
        report = to_report_units(np.zeros((3, 4), dtype=np.float32), TargetSpec())
        assert np.allclose(report, 0.5)


# --- An unmeasurable site is not an infeasible one ----------------------------
#
# `derived_feasible` returned float32 0/1 unconditionally, so `nan >= 12.0`
# being False turned "could not measure" into a confident "not feasible". The
# `seen` mask in `threshold_sensitivity` was written to catch exactly that and
# was inert, because the values reaching it were never NaN. The geometric
# baseline declines to answer on ~20% of sites by design and prints that count
# one line before feeding the same array in.

NAMES = ["available_height_mm", "ridge_width_mm"]
RULES = {"min_height_mandible_mm": 12.0, "min_width_mm": 6.0}


def test_unmeasurable_site_is_nan_not_infeasible():
    mm = np.array([[15.0, 7.0], [np.nan, 7.0], [15.0, np.nan]])
    got = derived_feasible(mm, NAMES, RULES)
    assert got[0] == 1.0
    assert np.isnan(got[1]), "a missing height must not read as 'not feasible'"
    assert np.isnan(got[2]), "a missing width must not read as 'not feasible'"


def test_two_failed_measurements_do_not_count_as_agreement():
    """The worst form of the bug: agreement manufactured out of two failures."""
    true = np.array([[np.nan, np.nan], [15.0, 7.0]])
    pred = np.array([[np.nan, np.nan], [15.0, 7.0]])
    row = threshold_sensitivity(true, pred, NAMES, RULES, sweep=(12.0,))[0]
    assert row["n"] == 1, "the unmeasurable row must be excluded, not agreed with"
    assert row["agreement"] == 1.0


def test_n_reports_the_sites_both_sides_could_measure():
    true = np.tile(np.array([[15.0, 7.0]]), (10, 1))
    pred = true.copy()
    pred[:4] = np.nan
    row = threshold_sensitivity(true, pred, NAMES, RULES, sweep=(12.0,))[0]
    assert row["n"] == 6
