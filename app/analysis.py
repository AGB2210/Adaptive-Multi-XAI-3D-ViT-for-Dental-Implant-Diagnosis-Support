"""The work the queue runs: analyse a scan, re-predict it, explain one site.

These glue `src/inference` to the stores and nothing more. Every number they
write comes from a function the training and XAI scripts also call.
"""

from __future__ import annotations

import shutil

import numpy as np

from app.registry import ModelRegistry
from app.settings import Settings
from app.store import ScanStore
from src.inference.explain import explain_patch
from src.inference.predict import as_input, predict_sites, score_sites, site_patch
from src.inference.scan import prepare_scan
from src.inference.sites import has_position, sites_from_mask

# Overview slab around the lower crest, in millimetres: deep enough to show the
# canal's course below the ridge, high enough to show the crowns above it.
OVERVIEW_BELOW_MM, OVERVIEW_ABOVE_MM = 18.0, 12.0


def to_uint8(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    scaled = (np.asarray(values, dtype=np.float32) - lo) / max(hi - lo, 1e-6)
    return np.clip(np.round(scaled * 255.0), 0, 255).astype(np.uint8)


def bundle_info(model_description: dict) -> dict:
    """What a verdict needs from the model, taken from its stored description."""
    return {"thresholds": model_description.get("decision_thresholds") or {},
            "mae": model_description.get("validation_mae_mm") or {}}


def overview(volume: np.ndarray, sites: list[dict], spacing_mm: float, window) -> tuple:
    """Axial maximum-intensity projection over a slab around the lower crest.

    A full-height MIP stacks the upper teeth on the lower ones and hides the
    sites; a slab around the median crest height shows the lower arch alone.
    Downsampled 2x for the browser. Returns (uint8 image, info).
    """
    zs = [s["site_z"] for s in sites if has_position(s)]
    zc = float(np.median(zs)) if zs else volume.shape[2] / 2.0
    lo = int(max(0, round(zc - OVERVIEW_BELOW_MM / spacing_mm)))
    hi = int(min(volume.shape[2], round(zc + OVERVIEW_ABOVE_MM / spacing_mm)))
    if hi <= lo:
        lo, hi = 0, volume.shape[2]
    step = 2
    mip = np.asarray(volume[::step, ::step, lo:hi], dtype=np.float32).max(axis=2)
    info = {"shape": list(mip.shape), "step": step, "z_range": [lo, hi],
            "window": list(window)}
    return to_uint8(mip, *window), info


def trusted(located: dict, minimum: float) -> bool:
    """Whether this localiser's orientation head has earned being acted on."""
    accuracy = located.get("orientation_accuracy")
    return accuracy is not None and float(accuracy) >= float(minimum)


def orientation_note(located: dict, minimum: float) -> str | None:
    """What the result says when the orientation head wanted the scan turned over.

    Turning a scan over changes every patch the site model is shown, and an
    upside-down jaw still produces confident millimetres. So the head is only
    obeyed when its measured accuracy clears `minimum`, and either way the
    result states what it said and how good it has been.
    """
    if not located.get("flip"):
        return None
    accuracy = located.get("orientation_accuracy")
    measured = ("has no measured accuracy" if accuracy is None
                else f"was {float(accuracy):.0%} correct on its validation patients")
    said = f"The localiser's orientation head says this scan is stored upside down (P = {located['flip_prob']:.2f})"
    if trusted(located, minimum):
        return f"{said}, and it {measured}: the scan was turned over."
    return (f"{said}, but it {measured}, below the {float(minimum):.0%} required to act on it: "
            f"the configured default orientation was kept.")


def analyse(settings: Settings, registry: ModelRegistry, store: ScanStore,
            scan_id: str, progress) -> dict:
    meta = store.meta(scan_id)
    site_model = registry.active("site")
    if site_model is None:
        raise RuntimeError("no site model is loaded -- add a .pt under Models first")

    folder = store.dir(scan_id)
    mask_path = folder / meta["mask_file"] if meta.get("mask_file") else None
    localiser = None
    sign, sign_source = None, None
    if mask_path is None:
        loc_meta = registry.active("localiser")
        if loc_meta is None:
            raise RuntimeError(
                "this scan came without a mask and no localiser model is loaded, so the "
                "tooth sites cannot be found. Upload the scan's mask with it, or add a "
                "localiser model under Models.")
        progress("Loading the localiser", 0.03)
        localiser = registry.load(loc_meta["id"])
        sign, sign_source = settings.default_orientation_sign, "configured default"

    progress("Reading and normalising the scan", 0.06)
    prepared = prepare_scan(
        folder / meta["image_file"], mask_path,
        clip_window=settings.clip_window, air_threshold=settings.air_threshold,
        target_spacing=settings.spacing_mm, spacing_tolerance=settings.spacing_tolerance,
        sign=sign, sign_source=sign_source)
    warnings = list(prepared.warnings)

    if localiser is not None:
        progress("Locating tooth sites (localiser)", 0.45)
        located = localiser.locate(prepared.volume, settings.spacing_mm,
                                   jaws=settings.site_jaws)
        note = orientation_note(located, settings.min_orientation_accuracy)
        if located.get("flip") and trusted(located, settings.min_orientation_accuracy):
            # The localiser's orientation head disagrees with the default: the
            # scan is the other way up. Re-prepare rather than flip a z-scored
            # array, so the volume is produced by the one shared transform.
            prepared = prepare_scan(
                folder / meta["image_file"], None,
                clip_window=settings.clip_window, air_threshold=settings.air_threshold,
                target_spacing=settings.spacing_mm, spacing_tolerance=settings.spacing_tolerance,
                sign=-prepared.sign, sign_source="localiser orientation head")
            located = localiser.locate(prepared.volume, settings.spacing_mm,
                                       jaws=settings.site_jaws)
        sites = located["sites"]
        warnings += located.get("warnings", [])
        if note:
            warnings.append(note)
    else:
        progress("Locating and measuring tooth sites (mask)", 0.45)
        sites = sites_from_mask(prepared.mask, prepared.spacing, settings.rules,
                                jaws=settings.site_jaws)

    progress("Saving the prepared volume", 0.7)
    store.save_array(scan_id, "volume.npy", prepared.volume)
    image, info = overview(prepared.volume, sites, settings.spacing_mm, settings.window)
    store.save_array(scan_id, "overview.npy", image)

    result = {
        "scan": {**prepared.summary(), "patient_id": meta["patient_id"]},
        "overview": info,
        "sites": sites,
        "warnings": warnings,
        "site_source": "localiser" if localiser is not None else "mask",
    }
    store.save_json(scan_id, "result.json", result)
    return predict(settings, registry, store, scan_id, progress, start=0.8)


def predict(settings: Settings, registry: ModelRegistry, store: ScanStore, scan_id: str,
            progress, start: float = 0.1) -> dict:
    """(Re-)run the active site model over a scan whose sites are already known."""
    result = store.result(scan_id)
    if result is None:
        raise RuntimeError("this scan has not been analysed yet")
    site_model = registry.active("site")
    if site_model is None:
        raise RuntimeError("no site model is loaded")
    bundle = registry.load(site_model["id"])

    progress("Predicting every site", start)
    volume = store.load_array(scan_id, "volume.npy")
    sites = [{k: v for k, v in s.items() if k not in ("prediction", "verdict")}
             for s in result["sites"]]
    with registry.lock(site_model["id"]):      # see ModelRegistry.lock
        sites = predict_sites(bundle, volume, sites)

    result["sites"] = sites
    result["model"] = {"id": site_model["id"], "name": site_model["name"],
                       "fold": site_model.get("fold"),
                       "description": bundle.describe(settings.spacing_mm)}
    result["patient_role"] = registry.patient_role(site_model["id"],
                                                   result["scan"]["patient_id"])
    store.save_json(scan_id, "result.json", result)
    # Explanations belong to the model that produced them.
    explain_root = store.dir(scan_id) / "explain"
    if explain_root.is_dir():
        shutil.rmtree(explain_root, ignore_errors=True)
    progress("Done", 1.0)
    return {"scan_id": scan_id}


def scored(result: dict, rules: dict) -> dict:
    """The result with a verdict on every site, at the given thresholds."""
    info = bundle_info(result.get("model", {}).get("description", {}))
    return {**result, "rules": rules, "sites": score_sites(result["sites"], info, rules)}


def explain_key(tooth: int, target: str, force: bool) -> str:
    return f"{int(tooth)}-{target}-{'full' if force else 'auto'}"


def explain(settings: Settings, registry: ModelRegistry, store: ScanStore, scan_id: str,
            tooth: int, target: str, force: bool, progress) -> dict:
    result = store.result(scan_id)
    if result is None or "model" not in result:
        raise RuntimeError("this scan has no predictions to explain")
    site = next((s for s in result["sites"] if int(s["tooth"]) == int(tooth)), None)
    if site is None or site.get("prediction") is None:
        raise RuntimeError(f"site {tooth} has no prediction")

    model_id = result["model"]["id"]
    if registry.get(model_id) is None:
        raise RuntimeError("the model that made these predictions has been removed; "
                           "re-run the scan with the active model")
    progress("Loading the model", 0.02)
    bundle = registry.load(model_id)
    if target not in bundle.names:
        raise ValueError(f"{target!r} is not one of this model's outputs {bundle.names}")

    volume = store.load_array(scan_id, "volume.npy")
    patch = as_input(site_patch(volume, site, bundle.img_size), bundle.device)
    with registry.lock(model_id):              # see ModelRegistry.lock
        out = explain_patch(bundle, patch, bundle.names.index(target),
                            site["prediction"]["logits"], settings.xai,
                            force_ensemble=force, progress=progress)

    key = explain_key(tooth, target, force)
    folder = store.explain_dir(scan_id, key)
    folder.mkdir(parents=True, exist_ok=True)
    for name, saliency in out["maps"].items():
        np.save(folder / f"{name}.npy", to_uint8(saliency.numpy(), 0.0, 1.0))
    meta = {k: v for k, v in out.items() if k != "maps"}
    meta.update({"key": key, "tooth": int(tooth), "methods": list(out["maps"]),
                 "shape": list(next(iter(out["maps"].values())).shape)})
    store.save_json(scan_id, f"explain/{key}/meta.json", meta)
    return meta

