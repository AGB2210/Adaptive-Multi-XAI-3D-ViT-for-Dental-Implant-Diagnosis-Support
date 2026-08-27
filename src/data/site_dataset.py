"""Training samples that are tooth SITES, not whole scans.

One 532-scan dataset becomes ~14,900 site samples, and each sample is a small
box cut from the scan at its native 0.3 mm. That is the whole reason for this
module, and it is worth being explicit about why it beats the obvious
alternative of feeding the model a whole downsampled head:

    128^3 at 1.0 mm      whole head, 3.3x blurred     532 samples
    256^3 at 0.5 mm      whole head, 1.7x blurred     532 samples
    96^3  at 0.3 mm      one site, NOT blurred      ~14,900 samples

The patch is both sharper and 19x smaller than the 256^3 volume. It matters
because the structure the model has to respect is the inferior alveolar canal,
which is 2-3 mm across: at 1.0 mm that is two or three voxels, and no
explanation method can point at something it cannot resolve.

SITE POSITIONS COME FROM THE MASKS. scripts/build_implant_labels.py fits the
dental arch to the teeth a scan still has and writes each site's voxel
coordinates into the CSV. That is legitimate for training and for measuring
this model, and it is NOT a deployable pipeline: a new patient arrives with an
image and no segmentation, so a fielded system needs a site-detection step that
does not exist here. Say so in the report rather than implying otherwise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SITE_TARGETS = ("needs_implant", "feasible")


def load_sites(
    sites_csv: str | Path,
    targets=SITE_TARGETS,
    methods=("teeth",),
    jaws=("lower",),
    drop_unmeasurable: bool = True,
) -> pd.DataFrame:
    """Read sites_*.csv and keep the rows that can honestly be trained on.

    `methods` filters on how each site position was found. The default keeps
    only positions fitted to three or more real teeth; passing
    ("teeth", "sparse", "opposite_jaw") includes the weaker tiers, which is how
    the edentulous patients get represented. That is a real trade-off between
    coverage and position accuracy, so it is a parameter rather than a decision
    buried in here.

    `jaws` defaults to the mandible alone, and that is a finding rather than a
    preference. Measured over all 522 scans, of the sites that need an implant:

        mandible   637 needed, 585 measurable   (91.8%)
        maxilla   1682 needed,  72 measurable   ( 4.3%)

    After a maxillary tooth is lost the alveolar ridge resorbs and the sinus
    pneumatises, and ToothFairy3's UpperJaw mask does not cover what remains --
    98% of those sites have literally no bone voxels to measure. Training on
    them would teach the model to reproduce an annotation gap as a clinical
    verdict. The mandible loses nothing important: the inferior alveolar canal
    is annotated in 100% of scans, and nerve clearance is the limiting factor in
    289 of the 361 infeasible sites, which is the question worth explaining.
    """
    df = pd.read_csv(sites_csv, dtype={"patient_id": str})
    for column in ("patient_id", "tooth", "site_x", "site_y", *targets):
        if column not in df.columns:
            raise ValueError(f"{sites_csv} has no {column!r} column -- rebuild it "
                             "with scripts/build_implant_labels.py")

    if jaws is not None:
        df = df[df.jaw.isin(jaws)]
    if methods is not None:
        df = df[df.site_method.isin(methods)]
    df = df[df.site_x.notna() & df.site_y.notna()]
    if drop_unmeasurable:
        # A site we could not measure is missing data, not a negative finding.
        # Training on it as "not feasible" would teach the model to reproduce
        # our own measurement failures as clinical verdicts.
        #
        # TWO CHECKS, NOT ONE. `feasibility()` returns feasible=False when a
        # site is unmeasurable -- it cannot return True -- so the value written
        # to the CSV is a plain False and a notna() test sails straight past it.
        # Only the `reason` column distinguishes "measured, and the answer is
        # no" from "could not measure". Missed, this silently mislabels every
        # site where the arch fit landed off the bone.
        df = df[df[list(targets)].notna().all(axis=1)]
        if "reason" in df.columns:
            df = df[df.reason != "unmeasurable"]
    return df.reset_index(drop=True)


def target_matrix(df: pd.DataFrame, targets=SITE_TARGETS) -> np.ndarray:
    return df[list(targets)].to_numpy(dtype=np.float32)



def patch_centre(row, volume_shape, patch_size: int):
    """Where the model's input box is centred, for one site.

    ONE function because training and explanation must agree voxel for voxel. If
    they drift, an attribution map describes an input the model never saw and
    every number downstream still looks reasonable.

    The box is pushed off-centre toward the structure that decides the answer.
    Centred, 96^3 at 0.3 mm reaches 14.4 mm below the crest against a 12.0 mm
    threshold -- 2.4 mm of margin, with 26.2% of sites needing an implant having
    their limiting structure outside the input entirely -- while half the box sat
    above the crest on air, soft tissue and opposing crowns. Quarter-shifted it
    reaches 21.6 mm below and 7.2 mm above.

    The nerve lies BELOW the crest in the mandible (`measure_site` computes
    crest - canal_top) and the sinus ABOVE it in the maxilla, so the sign follows
    the jaw.
    """
    def field(name, default=np.nan):
        return row[name] if not hasattr(row, "get") else row.get(name, default)

    z = float(field("site_z"))
    if not np.isfinite(z):
        return (float(field("site_x")), float(field("site_y")), volume_shape[2] / 2.0)
    shift = patch_size // 4
    jaw = field("jaw", "lower")
    z = z - shift if jaw == "lower" else z + shift
    return (float(field("site_x")), float(field("site_y")), z)


def cut_patch(volume: np.ndarray, centre, size: int, pad_value: float = 0.0) -> np.ndarray:
    """A `size`^3 box centred on `centre`, zero-padded where it leaves the scan.

    Padding rather than shifting the box: a site near the edge of the field of
    view genuinely has less context, and sliding the window would silently
    centre it on different anatomy than the label describes.
    """
    out = np.full((size, size, size), pad_value, dtype=volume.dtype)
    half = size // 2
    src, dst = [], []
    for axis in range(3):
        c = int(round(float(centre[axis])))
        lo, hi = c - half, c - half + size
        s0, s1 = max(0, lo), min(volume.shape[axis], hi)
        if s0 >= s1:
            return out                      # box lies entirely outside the scan
        src.append(slice(s0, s1))
        dst.append(slice(s0 - lo, s1 - lo))
    out[dst[0], dst[1], dst[2]] = volume[src[0], src[1], src[2]]
    return out


class SitePatchDataset(Dataset):
    """One sample per tooth site: a native-resolution box and its labels.

    Volumes are memory-mapped, so a 50 GB native-resolution cache costs no RAM
    and a patch read touches only the pages it needs. Do not switch this to
    np.load without mmap_mode unless the cache genuinely fits in memory.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        sites: pd.DataFrame,
        targets=SITE_TARGETS,
        patch_size: int = 96,
        augment=None,
        seed: int = 0,
    ):
        self.cache_dir = Path(cache_dir)
        self.sites = sites.reset_index(drop=True)
        self.targets = list(targets)
        self.patch_size = int(patch_size)
        self.augment = augment
        self.seed = int(seed)
        self.labels = target_matrix(self.sites, self.targets)

        wanted = sorted(set(self.sites.patient_id))
        missing = [p for p in wanted if not (self.cache_dir / f"{p}.npy").exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} of {len(wanted)} cached volumes missing under "
                f"{self.cache_dir} (first few: {missing[:5]}). Run scripts/build_cache.py."
            )
        self._open: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.sites)

    def volume(self, patient_id: str) -> np.ndarray:
        cached = self._open.get(patient_id)
        if cached is None:
            cached = np.load(self.cache_dir / f"{patient_id}.npy", mmap_mode="r")
            self._open[patient_id] = cached
        return cached

    def __getitem__(self, idx: int):
        row = self.sites.iloc[idx]
        volume = self.volume(row.patient_id)

        # z comes from the crest so the box straddles the ridge rather than
        # sitting wherever the arch fit happened to put it -- then it is pushed
        # OFF-CENTRE, toward the structure that decides the answer.
        #
        # Centred, a 96^3 box at 0.3 mm reaches 14.4 mm below the crest against a
        # 12.0 mm threshold: 2.4 mm of margin, with 26.2% of sites needing an
        # implant having available height beyond 14.4 mm, i.e. their limiting
        # structure outside the input entirely. The other half of the box was
        # spent above the crest on air, soft tissue and opposing crowns.
        #
        # Quarter-shifted it reaches 21.6 mm below and 7.2 mm above. The nerve
        # lies BELOW the crest in the mandible (measure_site computes
        # crest - canal_top) and the sinus ABOVE it in the maxilla, so the sign
        # follows the jaw.
        centre = patch_centre(row, volume.shape, self.patch_size)
        patch = cut_patch(volume, centre, self.patch_size)
        patch = np.asarray(patch, dtype=np.float32)

        if self.augment is not None:
            rng = np.random.default_rng(
                (self.seed, idx, int(torch.randint(0, 2**31, (1,)).item()))
            )
            patch = self.augment(patch, rng)

        x = torch.from_numpy(np.ascontiguousarray(patch)).unsqueeze(0)
        y = torch.from_numpy(self.labels[idx])
        return x, y


def group_ids(sites: pd.DataFrame) -> np.ndarray:
    """Patient id per row, for splitting.

    THE SPLIT MUST BE BY PATIENT, NOT BY SITE. Twenty-eight sites from one scan
    share the same anatomy, the same field of view and the same annotator. Split
    them at random and the model sees a patient's left molars in training and
    their right molars in test, which reads as generalisation and is not.
    """
    return sites.patient_id.to_numpy()


def patient_label_matrix(sites: pd.DataFrame, targets=SITE_TARGETS):
    """Per-PATIENT summary used only to stratify the split.

    The splitter balances rare label combinations across folds, and it has to
    operate on the unit being split. That unit is the patient, so each one is
    summarised by whether they have ANY site of each kind: a patient with an
    infeasible site is what must be spread evenly across folds, not each of
    their 28 rows independently.

    Returns (patient_ids, matrix) aligned row for row.
    """
    grouped = sites.groupby("patient_id")[list(targets)].max()
    return grouped.index.tolist(), grouped.to_numpy(dtype=np.float32)


def sites_for_patients(sites: pd.DataFrame, patient_ids) -> pd.DataFrame:
    """Every site row belonging to the given patients, order preserved."""
    wanted = set(patient_ids)
    return sites[sites.patient_id.isin(wanted)].reset_index(drop=True)


def restrict_sites_to_cache(sites: pd.DataFrame, cache_dir) -> pd.DataFrame:
    """Drop sites whose scan never made it through preprocessing."""
    cache_dir = Path(cache_dir)
    have = {p for p in set(sites.patient_id) if (cache_dir / f"{p}.npy").exists()}
    return sites[sites.patient_id.isin(have)].reset_index(drop=True)
