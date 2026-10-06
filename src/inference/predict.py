"""Site patches -> model outputs in report units -> a verdict per site.

The patch is cut by `site_dataset.patch_centre` and `cut_patch`, the pair every
training sample and every explanation is cut with. `CaseSet.load` does exactly
this: cut, cast to float32, add batch and channel axes. A third way of cutting
it is how an explanation ends up describing an input the model never saw.

FEASIBILITY IS NOT A MODEL OUTPUT. It is `derived_feasible` over the two
predicted lengths, at whatever thresholds are passed -- so a revised clinical
threshold is a re-score of numbers already on the page, never a re-run.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from src.data.site_dataset import cut_patch, patch_centre
from src.inference.checkpoint import ModelBundle
from src.inference.sites import has_position
from src.train.targets import derived_feasible, to_report_units
from src.xai.calibration import uncertainty

HEIGHT, WIDTH = "available_height_mm", "ridge_width_mm"


def site_patch(volume: np.ndarray, site: dict, patch_size: int) -> np.ndarray:
    """The model's input for one site, (D, H, W) float32 -- `CaseSet.load` minus the axes."""
    row = {"site_x": site["site_x"], "site_y": site["site_y"],
           "site_z": site["site_z"], "jaw": site.get("jaw", "lower")}
    centre = patch_centre(row, volume.shape, int(patch_size))
    return np.asarray(cut_patch(volume, centre, int(patch_size)), dtype=np.float32)


def as_input(patch: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(patch))[None, None].to(device)


@torch.no_grad()
def predict_sites(bundle: ModelBundle, volume: np.ndarray, sites: list[dict]) -> list[dict]:
    """Outputs for every site that has a position, in each head's own unit.

    One forward pass over all of a scan's sites. The model normalises with
    GroupNorm and LayerNorm in eval mode, so a site's output does not depend on
    which other sites share its batch.
    """
    todo = [s for s in sites if has_position(s)]
    if todo:
        batch = torch.cat([as_input(site_patch(volume, s, bundle.img_size), bundle.device)
                           for s in todo])
        raw = bundle.model(batch).float().cpu().numpy()
    else:
        raw = np.zeros((0, len(bundle.names)))
    report = to_report_units(raw, bundle.spec, bundle.temperature)

    n_bin = bundle.n_binary
    unc = (uncertainty(report[:, :n_bin], "margin") if n_bin and len(report)
           else np.full(len(report), np.nan))

    by_tooth = {}
    for i, site in enumerate(todo):
        by_tooth[site["tooth"]] = {
            "outputs": {name: float(report[i, j]) for j, name in enumerate(bundle.names)},
            "logits": [float(v) for v in raw[i]],
            "uncertainty": float(unc[i]),
        }
    return [{**s, "prediction": by_tooth.get(s["tooth"])} for s in sites]


def _finite(v) -> bool:
    return v is not None and math.isfinite(v)


def verdict(site: dict, bundle_info: dict, rules: dict) -> dict:
    """The colour on the jaw chart, and the reasons for it, for one site.

    `bundle_info` carries what the verdict needs from the model without the
    model itself -- decision threshold, validation MAE -- so a re-score never
    touches a GPU.
    """
    pred = site.get("prediction")
    if pred is None:
        return {"status": "no_position", "label": "No position",
                "reasons": [site.get("note") or "this site could not be located"]}

    out = pred["outputs"]
    jaw = site.get("jaw", "lower")
    height_rule = rules["min_height_mandible_mm" if jaw == "lower" else "min_height_maxilla_mm"]
    width_rule = rules["min_width_mm"]
    h, w = out.get(HEIGHT), out.get(WIDTH)

    mm = np.array([[h if _finite(h) else np.nan, w if _finite(w) else np.nan]], dtype=np.float64)
    feasible = derived_feasible(mm, [HEIGHT, WIDTH], rules, jaw=jaw)[0]

    reasons = []
    if _finite(h) and h < height_rule:
        reasons.append(f"height {h:.1f} mm < {height_rule:g} mm rule")
    if _finite(w) and w < width_rule:
        reasons.append(f"width {w:.1f} mm < {width_rule:g} mm rule")

    # Borderline: the prediction sits closer to a threshold than this model's
    # own validation error. Calling it either way would claim a precision the
    # model has not shown. Without a measured MAE there is no band to apply, and
    # the result says that instead of inventing one.
    mae = bundle_info.get("mae", {})
    near = []
    if _finite(h) and HEIGHT in mae and abs(h - height_rule) < mae[HEIGHT]:
        near.append(f"height within {mae[HEIGHT]:.1f} mm (validation MAE) of the rule")
    if _finite(w) and WIDTH in mae and abs(w - width_rule) < mae[WIDTH]:
        near.append(f"width within {mae[WIDTH]:.1f} mm (validation MAE) of the rule")

    need_p = out.get("needs_implant")
    threshold = bundle_info.get("thresholds", {}).get("needs_implant", 0.5)
    needs = None if need_p is None else bool(need_p >= threshold)

    if not np.isfinite(feasible):
        status, label = "unmeasurable", "Unmeasurable"
        reasons = ["a predicted length is not finite"]
    elif needs is False:
        status, label = "not_needed", "No implant needed"
        reasons = [f"P(needs implant) {need_p:.2f} < {threshold:.2f}"]
    elif near:
        status, label = "borderline", "Borderline"
        reasons = near + reasons
    elif feasible == 1.0:
        status, label = "feasible", "Needs implant, feasible"
        reasons = [f"height {h:.1f} mm, width {w:.1f} mm clear both rules"]
    else:
        status, label = "not_feasible", "Needs implant, not feasible"

    return {"status": status, "label": label, "reasons": reasons,
            "needs_implant": needs, "feasible": None if not np.isfinite(feasible) else bool(feasible),
            "rules": {"height_mm": height_rule, "width_mm": width_rule},
            "decision_threshold": threshold}


def score_sites(sites: list[dict], bundle_info: dict, rules: dict) -> list[dict]:
    """Attach a verdict to every site. Pure: safe to call on every threshold change."""
    return [{**s, "verdict": verdict(s, bundle_info, rules)} for s in sites]
