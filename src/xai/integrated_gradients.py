"""Integrated Gradients (Sundararajan et al., 2017).

    IG(x) = (x - x') * integral_0^1 dF(x' + a(x - x'))/dx da

approximated with the trapezoidal rule.

Two things this implementation takes seriously:

  * The baseline is NOT a zero volume. On z-scored CBCT, zero is a real tissue
    intensity near the foreground mean -- it means "average tissue", not "nothing
    there". Default is a Gaussian-blurred copy of the input; the dataset mean
    volume is the configurable alternative.

  * The completeness axiom, sum(IG) == F(x) - F(x'), is IG's core guarantee. It
    is checked on every call and recorded on `.last_completeness_error`. If it
    does not hold, the implementation is wrong -- the unit test fails above ~5%
    rather than letting a broken attribution through.
"""

from __future__ import annotations

import torch

from src.utils.log import get_logger
from src.xai.base import SaliencyMethod, make_baseline, target_logit

log = get_logger("integrated_gradients")


class IntegratedGradients(SaliencyMethod):
    name = "integrated_gradients"

    def __init__(
        self,
        model,
        device=None,
        steps: int = 256,
        baseline: str = "blur",
        blur_sigma: float = 4.0,
        mean_volume: torch.Tensor | None = None,
        batch_size: int = 8,
        absolute: bool = True,
        completeness_tolerance: float = 0.05,
    ):
        super().__init__(model, device)
        self.steps = steps
        self.baseline_kind = baseline
        self.blur_sigma = blur_sigma
        self.mean_volume = mean_volume
        self.batch_size = batch_size
        self.absolute = absolute
        self.completeness_tolerance = completeness_tolerance
        self.last_completeness_error: float = float("nan")
        self.last_completeness: dict = {}
        self.completeness_ok: bool | None = None

    def _attribute_raw(self, volume: torch.Tensor, target_label: int) -> torch.Tensor:
        self.model.eval()
        # Every step backpropagates, which accumulates into the model's own
        # parameter .grad buffers. Left behind they waste memory and contaminate
        # anything that later reads .grad, so they are cleared either side.
        self.model.zero_grad(set_to_none=True)
        baseline = make_baseline(volume, self.baseline_kind, self.mean_volume, self.blur_sigma)
        delta = volume - baseline

        # Trapezoidal rule over [0, 1]: endpoints carry half weight.
        alphas = torch.linspace(0.0, 1.0, self.steps + 1, device=volume.device)
        weights = torch.full_like(alphas, 1.0 / self.steps)
        weights[0] *= 0.5
        weights[-1] *= 0.5

        total = torch.zeros_like(volume)
        for start in range(0, alphas.numel(), self.batch_size):
            chunk = alphas[start : start + self.batch_size]
            w = weights[start : start + self.batch_size]

            points = baseline + chunk.view(-1, 1, 1, 1, 1) * delta
            points = points.detach().requires_grad_(True)

            logits = self.model(points)
            logits[:, target_label].sum().backward()

            total += (points.grad * w.view(-1, 1, 1, 1, 1)).sum(dim=0, keepdim=True)

        self.model.zero_grad(set_to_none=True)

        ig = delta * total
        self._check_completeness(ig, volume, baseline, target_label)

        attribution = ig[0].sum(dim=0)  # collapse channels
        return attribution.abs() if self.absolute else attribution

    @torch.no_grad()
    def _check_completeness(self, ig, volume, baseline, target_label) -> None:
        """sum(IG) should equal F(x) - F(x'). Records the relative error."""
        f_x = float(target_logit(self.model, volume, target_label))
        f_base = float(target_logit(self.model, baseline, target_label))
        expected = f_x - f_base
        actual = float(ig.sum())

        scale = max(abs(expected), 1e-6)
        self.last_completeness_error = abs(actual - expected) / scale
        self.completeness_ok = bool(self.last_completeness_error <= self.completeness_tolerance)
        self.last_completeness = {
            "sum_ig": actual,
            "f_x_minus_f_baseline": expected,
            "relative_error": self.last_completeness_error,
            "steps": self.steps,
            "within_tolerance": self.completeness_ok,
        }

        if not self.completeness_ok:
            # Loud, every call. A violated axiom means the attribution is not IG,
            # and a wrong map that looks plausible is the worst outcome here.
            log.warning(
                "IG completeness violated: relative error %.4f > %.4f at %d steps "
                "(sum_IG=%.5f vs F(x)-F(x')=%.5f). Increase steps before trusting this map.",
                self.last_completeness_error, self.completeness_tolerance, self.steps, actual, expected,
            )
