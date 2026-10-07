"""Uploaded scans and everything computed from them, one directory each.

    scans/<id>/meta.json           what was uploaded, and when
    scans/<id>/image.nii[.gz]      the scan as uploaded
    scans/<id>/mask.nii[.gz]       its segmentation, if one came with it
    scans/<id>/volume.npy          the prepared volume (float16), for patches
    scans/<id>/overview.npy        axial MIP over the lower jaw, uint8
    scans/<id>/result.json         sites, predictions, provenance
    scans/<id>/explain/<key>/      one explanation: meta.json + a uint8 map per method
"""

from __future__ import annotations

import json
import math
import re
import shutil
import time
import uuid
from pathlib import Path

import numpy as np

from src.inference.scan import patient_id_from_filename


def clean(obj):
    """JSON-safe: non-finite floats become null, numpy scalars become Python."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    return obj


# An id or a key from a URL becomes a directory name. Anything that is not a
# plain name -- a separator, a dot, a drive letter -- would resolve OUTSIDE the
# store: on Windows a backslash is a path separator inside one URL segment, and
# `DELETE /api/scans/..%5Cmodels%5C<id>` removed a model's folder through the
# scan route. Refusing the name is cheaper than reasoning about each join.
SAFE_NAME = re.compile(r"[A-Za-z0-9_-]+")


def safe_name(value: str) -> str:
    """`value` if it is a plain file-system name, else KeyError (a 404)."""
    if not isinstance(value, str) or not SAFE_NAME.fullmatch(value):
        raise KeyError(value)
    return value


# Windows will not replace a file while another thread has it open, nor open one
# in the instant it is being replaced. Both pass in well under a millisecond, so
# the request that reads `result.json` as a job rewrites it waits instead of
# failing the job with "Access is denied". Other systems never raise here.
ATTEMPTS, PAUSE_S = 25, 0.02


def _patiently(action):
    for attempt in range(ATTEMPTS):
        try:
            return action()
        except PermissionError:
            if attempt == ATTEMPTS - 1:
                raise
            time.sleep(PAUSE_S)


def replace_file(tmp: Path, path: Path) -> None:
    """Move a finished temporary file over `path`, so a reader sees old or new, never half."""
    _patiently(lambda: tmp.replace(path))


def read_text(path: Path) -> str:
    return _patiently(lambda: path.read_text(encoding="utf-8"))


def write_json(path: Path, data, indent: int = 1) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(data), indent=indent), encoding="utf-8")
    replace_file(tmp, path)


def nifti_suffix(filename: str) -> str:
    name = filename.lower()
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    if name.endswith(".nii"):
        return ".nii"
    raise ValueError(f"{filename}: expected a NIfTI file (.nii or .nii.gz), the format "
                     f"ToothFairy3 ships its scans in")


class ScanStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def dir(self, scan_id: str) -> Path:
        path = self.root / safe_name(scan_id)
        if not (path / "meta.json").is_file():
            raise KeyError(scan_id)
        return path

    def create(self, image: tuple[str, Path], mask: tuple[str, Path] | None) -> dict:
        image_name, image_tmp = image
        image_file = "image" + nifti_suffix(image_name)
        mask_file = ("mask" + nifti_suffix(mask[0])) if mask else None

        scan_id = uuid.uuid4().hex[:10]
        path = self.root / scan_id
        path.mkdir(parents=True)
        shutil.move(str(image_tmp), path / image_file)
        if mask:
            shutil.move(str(mask[1]), path / mask_file)
        meta = {
            "id": scan_id,
            "patient_id": patient_id_from_filename(image_name),
            "image_name": image_name,
            "mask_name": mask[0] if mask else None,
            "image_file": image_file,
            "mask_file": mask_file,
            "created": time.time(),
        }
        self.save_json(scan_id, "meta.json", meta)
        return meta

    def meta(self, scan_id: str) -> dict:
        return self.load_json(scan_id, "meta.json")

    def list(self) -> list[dict]:
        out = []
        for d in self.root.iterdir() if self.root.is_dir() else []:
            if (d / "meta.json").is_file():
                meta = json.loads(read_text(d / "meta.json"))
                meta["analysed"] = (d / "result.json").is_file()
                out.append(meta)
        return sorted(out, key=lambda m: m.get("created", 0), reverse=True)

    def delete(self, scan_id: str) -> None:
        shutil.rmtree(self.dir(scan_id))

    # ---- files ----------------------------------------------------------
    def save_json(self, scan_id: str, name: str, data: dict) -> None:
        path = self.root / safe_name(scan_id) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, data)

    def load_json(self, scan_id: str, name: str) -> dict:
        path = self.dir(scan_id) / name
        if not path.is_file():
            raise FileNotFoundError(name)
        return json.loads(read_text(path))

    def result(self, scan_id: str) -> dict | None:
        try:
            return self.load_json(scan_id, "result.json")
        except FileNotFoundError:
            return None

    def save_array(self, scan_id: str, name: str, array: np.ndarray) -> None:
        path = self.root / safe_name(scan_id) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npy")
        np.save(tmp, np.ascontiguousarray(array))
        replace_file(tmp, path)

    def load_array(self, scan_id: str, name: str, mmap: bool = True) -> np.ndarray:
        path = self.dir(scan_id) / name
        if not path.is_file():
            raise FileNotFoundError(name)
        return np.load(path, mmap_mode="r" if mmap else None)

    def explain_dir(self, scan_id: str, key: str) -> Path:
        return self.dir(scan_id) / "explain" / safe_name(key)
