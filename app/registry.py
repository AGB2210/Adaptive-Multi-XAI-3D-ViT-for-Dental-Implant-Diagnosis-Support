"""Models added in the browser, kept on disk, one active per kind.

No path is configured anywhere. A model is whatever `.pt` the user picks, plus
whichever companions they pick with it -- its calibration, its validation
metrics, its config, the fold partition it was trained under -- each stored
under a fixed name beside the weights, which is where `load_bundle` looks for
them. A model is validated by LOADING it before it is kept: a file that cannot
be read as a site model never appears in the list.

Two kinds:
  site       the 3D ViT that predicts need, height and width per site
  localiser  the network that finds the sites on a scan with no mask
"""

from __future__ import annotations

import json
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

from app.settings import Settings
from app.store import safe_name
from src.data.splits import fold_assignment, load_folds
from src.inference.checkpoint import load_bundle
from src.xai.runner import fold_from_checkpoint

KINDS = ("site", "localiser")

# Companion files, by what they are -> the name they are stored under.
ROLE_NAMES = {
    "checkpoint": "model.pt",
    "calibration": "calibration.json",
    "metrics": "best_val_metrics.json",
    "config": "config.yaml",
    "folds": "cv_folds.json",
}


def role_of(filename: str) -> str | None:
    """Which companion an uploaded file is, from its name alone."""
    name = filename.lower()
    if name.endswith((".pt", ".pth")):
        return "checkpoint"
    if name.endswith((".yaml", ".yml")):
        return "config"
    if name.endswith(".json"):
        if "calib" in name:
            return "calibration"
        if "fold" in name:
            return "folds"
        if "metric" in name:
            return "metrics"
    return None


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "model"


class ModelRegistry:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.root = settings.models_dir
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._loaded: dict[str, object] = {}
        self._model_locks: dict[str, threading.RLock] = {}

    # ---- listing --------------------------------------------------------
    def _meta_path(self, model_id: str) -> Path:
        return self.root / model_id / "meta.json"

    def get(self, model_id: str) -> dict | None:
        try:
            path = self._meta_path(safe_name(model_id))
        except KeyError:
            return None                  # not a name a model could have
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def list(self) -> list[dict]:
        out = []
        for d in sorted(self.root.iterdir()) if self.root.is_dir() else []:
            meta = self.get(d.name) if d.is_dir() else None
            if meta:
                out.append(meta)
        active = self._active()
        for m in out:
            m["active"] = active.get(m["kind"]) == m["id"]
        return sorted(out, key=lambda m: m.get("created", 0), reverse=True)

    def _active(self) -> dict:
        path = self.root / "active.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    def active(self, kind: str) -> dict | None:
        model_id = self._active().get(kind)
        return self.get(model_id) if model_id else None

    def activate(self, model_id: str) -> dict:
        meta = self.get(model_id)
        if meta is None:
            raise KeyError(model_id)
        with self._lock:
            active = self._active()
            active[meta["kind"]] = model_id
            (self.root / "active.json").write_text(json.dumps(active, indent=2), encoding="utf-8")
        return meta

    # ---- adding ---------------------------------------------------------
    def add(self, files: list[tuple[str, Path]], kind: str = "site", name: str | None = None,
            fold: int | None = None) -> dict:
        """Validate and keep one model. `files` is [(original filename, temp path)]."""
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        roles: dict[str, tuple[str, Path]] = {}
        for filename, path in files:
            role = role_of(filename)
            if role is None:
                raise ValueError(
                    f"{filename}: not recognised. Pick the .pt, and optionally its "
                    f"calibration.json, best_val_metrics.json, config .yaml and cv_folds.json")
            if role in roles:
                raise ValueError(f"two files look like the {role}: {roles[role][0]} and {filename}")
            roles[role] = (filename, path)
        if "checkpoint" not in roles:
            raise ValueError("no .pt checkpoint among the selected files")

        original = roles["checkpoint"][0]
        model_id = f"{_slug(name or Path(original).stem)}-{uuid.uuid4().hex[:6]}"
        target = self.root / model_id
        target.mkdir(parents=True)
        try:
            for role, (_, path) in roles.items():
                shutil.copyfile(path, target / ROLE_NAMES[role])
            # Validated by loading it, on the device it will run on -- and the
            # loaded model is kept, so the first scan does not load it again.
            loaded = self._load_from(kind, target)
            description = (loaded.describe(self.settings.spacing_mm) if kind == "site"
                           else loaded.describe())
            if fold is None:
                fold = fold_from_checkpoint(original)
            meta = {
                "id": model_id, "kind": kind, "name": name or Path(original).stem,
                "original_filename": original, "fold": fold,
                "created": time.time(), "files": sorted(ROLE_NAMES[r] for r in roles),
                "has_folds": "folds" in roles,
                "description": description,
            }
            if meta["has_folds"]:
                folds = load_folds(target / ROLE_NAMES["folds"])
                if fold is not None and not 0 <= int(fold) < len(folds):
                    raise ValueError(f"fold {fold} is out of range for {len(folds)} folds")
            self._meta_path(model_id).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except Exception:
            shutil.rmtree(target, ignore_errors=True)
            raise
        with self._lock:
            self._loaded[model_id] = loaded
        if self.active(kind) is None:
            self.activate(model_id)
        meta["active"] = self.active(kind)["id"] == model_id
        return meta

    def _load_from(self, kind: str, directory: Path):
        """Load one model onto the configured device; whatever cannot load is refused."""
        if kind == "site":
            return load_bundle(directory / ROLE_NAMES["checkpoint"], self.settings.device,
                               fallback_config=self.settings.config_path,
                               allow_unsafe=self.settings.allow_unsafe_checkpoints)
        from src.models.localiser import load_localiser
        return load_localiser(directory / ROLE_NAMES["checkpoint"], self.settings.device,
                              allow_unsafe=self.settings.allow_unsafe_checkpoints)

    def remove(self, model_id: str) -> None:
        if self.get(model_id) is None:
            raise KeyError(model_id)
        with self._lock:
            self._loaded.pop(model_id, None)
            self._model_locks.pop(model_id, None)
            active = self._active()
            for kind, mid in list(active.items()):
                if mid == model_id:
                    del active[kind]
            (self.root / "active.json").write_text(json.dumps(active, indent=2), encoding="utf-8")
        shutil.rmtree(self.root / model_id, ignore_errors=True)

    # ---- loading for inference -----------------------------------------
    def load(self, model_id: str):
        """The model on the configured device. Loaded once and kept until removed:
        switching between models, or explaining a scan an earlier model analysed,
        never reloads one."""
        with self.lock(model_id):
            with self._lock:
                if model_id in self._loaded:
                    return self._loaded[model_id]
            meta = self.get(model_id)
            if meta is None:
                raise KeyError(model_id)
            obj = self._load_from(meta["kind"], self.root / model_id)
            with self._lock:
                self._loaded[model_id] = obj
            return obj

    def lock(self, model_id: str) -> threading.RLock:
        """Held while a model runs. Jobs are concurrent, but ONE MODEL OBJECT is
        not re-entrant: attention rollout switches its attention capture on and
        off, Grad-CAM registers hooks on its blocks, and every attribution method
        clears its gradients. Two jobs inside the same model would read each
        other's activations, and a prediction made mid-rollout would overwrite
        the attention it is composing. Different models do not share a lock."""
        with self._lock:
            return self._model_locks.setdefault(model_id, threading.RLock())

    def patient_role(self, model_id: str, patient_id: str) -> dict:
        """Was this patient in the model's training data? Only answerable with folds."""
        meta = self.get(model_id) or {}
        if not meta.get("has_folds") or meta.get("fold") is None:
            return {"role": "unknown",
                    "detail": "add the model's cv_folds.json and fold to check whether this "
                              "patient was in its training data"}
        folds = load_folds(self.root / model_id / ROLE_NAMES["folds"])
        split = fold_assignment(folds, int(meta["fold"]))
        for role in ("test", "val", "train"):
            if patient_id in split[role]:
                detail = {
                    "test": "held out: this checkpoint never saw this patient",
                    "val": "validation patient: never trained on, but used to select the "
                           "checkpoint and fit its calibration",
                    "train": "TRAINING patient: the model has seen this scan, so its "
                             "predictions here are optimistic",
                }[role]
                return {"role": role, "detail": detail}
        return {"role": "not_in_cohort",
                "detail": "not in this model's fold partition: an unseen scan"}
