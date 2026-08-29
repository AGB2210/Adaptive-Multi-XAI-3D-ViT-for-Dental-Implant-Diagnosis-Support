# RUNBOOK — running this on a rented GPU

Written for someone who has **not** worked on this project and is running it on
a machine that is not the one it was developed on. Follow it top to bottom.

Everything before Step 4 is free and takes about fifteen minutes. **Do all of it
before you start paying for a GPU.**

---

## 0. What you are running

For each tooth position in a CBCT scan, the model predicts two things:

| Head | Kind | Question |
|---|---|---|
| `needs_implant` | binary | Is this site missing a tooth that should be replaced? |
| `available_height_mm` | **millimetres** | How much bone is there, crest to nerve? |
| `ridge_width_mm` | **millimetres** | How wide is the ridge? |

**Feasibility is not predicted — it is computed afterwards**, from the two
millimetre outputs and the thresholds in the config. That is deliberate: the
12 mm rule moves a third of the answers, and applying it at inference means
revising it is a re-score rather than five folds of retraining. `train.py`
prints the whole threshold sweep.

It is a 3D Vision Transformer written from scratch, plus eight explainability
methods also written from scratch. Explanations are scored against the inferior
alveolar canal: if the model says *"not feasible, nerve too close"*, we check
whether the explanation actually points at the nerve.

**Scope is the lower jaw only.** The upper jaw could be measured on 4.3% of
scans against 91.8% for the lower, so it is excluded on purpose. This is a
finding, not an oversight — see `README.md`.

---

## 1. Machine you need

| | Minimum | Why |
|---|---|---|
| GPU | 16 GB VRAM | 96³ patches at batch 64. 24 GB lets you raise the batch size |
| Disk | **80 GB free** | 28 GB dataset + 27 GB cache + checkpoints and headroom |
| RAM | 32 GB | cache building holds whole volumes in memory |
| Python | 3.11 or 3.12 | both are gated in CI |

**The cache is 26.4 GB over 522 volumes and builds in about 19 minutes** —
measured on the rented box during the fold-0 run, over the whole cohort. It is
float16 at native 0.3 mm; this pipeline deliberately does *no* downsampling, and
scans average ~48 MB.

This line used to read "~100 MB per scan × 532 ≈ 52 GB", and that was wrong in a
way worth keeping on the page. It was extrapolated from the 14 scans cached on
the development machine — which are the first 14 in alphabetical order, all of
them from the `ToothFairy3F` cohort, and F scans are among the largest at
512x512x262. The cohort is 61 F, 381 P and 44 S, so a head-of-list sample of one
cohort overestimated the mean by a factor of two.

`src/xai/runner.py` already carries the same lesson for a different reason:
*"Head-truncation (`ids[:n]`) is not a sample."* It applies to disk estimates as
much as to case selection. Whatever figure you are working from, run `du -sh`
against a few hundred real files before you rent to it.

---

## 2. Setup

```bash
git clone https://github.com/AGB2210/Adaptive-Multi-XAI-3D-ViT-for-Dental-Implant-Diagnosis-Support.git capstone-code
cd capstone-code
git checkout "$(git tag -l 'v*' --sort=-v:refname | head -1)"
python -m venv .venv && source .venv/bin/activate
```

**Take the latest tag, not `main`.** That second line picks it for you --
`git tag -l --sort=-v:refname` orders tags by version rather than by date, so
`head -1` is the highest, not the most recently pushed. Confirm what you got:

```bash
cat VERSION
```

Every expected number below is checked against the code you just cloned, so a
newer tag will still agree with it -- and where a number has moved, the command
that produces it is given beside it so you can see the current value rather than
trust this page. **Do not go backwards.** The older tags answer a different
question: v1.0.0's labels are wrong, and v2.0.0 classifies feasibility instead
of measuring it, so nothing here would match.

`main` is fine too if you want work that is not yet released; expect the counts
below to have moved and read them as approximate.

On Windows the activate line is `.venv\Scripts\activate` instead.

Install torch **first**, matched to the box's CUDA version — check it with
`nvidia-smi`, then take the matching command from https://pytorch.org.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

```bash
pip install -r requirements.txt
```

Point the code at the dataset. **No path is committed anywhere in this repo**;
this env var is the only way it finds data.

```bash
export TOOTHFAIRY3_ROOT=/path/to/ToothFairy3
```

On Windows PowerShell that is `$env:TOOTHFAIRY3_ROOT = "C:\path\to\ToothFairy3"`.

The directory must contain `imagesTr/` and `labelsTr/`. Confirm it resolves:

```bash
python -c "import os,glob; r=os.environ['TOOTHFAIRY3_ROOT']; print(len(glob.glob(r+'/imagesTr/*.nii.gz')), 'volumes')"
```

Expect **532**.

---

## 3. Prove it works before you pay

```bash
python -m pytest -q
```

**Everything must pass and nothing may fail.** Don't check the count against a
number written here -- it only goes up as tests are added, and a runbook that
tells you to halt on `385 passed` because it was written at 380 wastes your
time. `0 failed` is the gate. (It was 380 at v3.0.1, for reference only.)

```bash
python scripts/check_imports.py
```

Must exit 0 -- it imports every module in `src/` and `scripts/` and fails on the
first one that cannot be loaded, which is how a missing dependency shows up
before it costs you GPU hours. The module count it prints is informational.

```bash
python scripts/train.py --config configs/sites_smoke.yaml --synthetic
```

That trains on planted-signal data in about 30 seconds and needs no dataset at
all. It must finish and report a val macro AUROC **above 0.5**. If it does not,
the install is broken — stop and fix it, because nothing downstream will work
either.

Then the real smoke gate, which does use the dataset:

```bash
python scripts/build_implant_labels.py --config configs/sites_smoke.yaml --limit 14
```

```bash
python scripts/build_site_cache.py --config configs/sites_smoke.yaml --limit 14
```

```bash
python scripts/train.py --config configs/sites_smoke.yaml --num-workers 0
```

```bash
python scripts/run_xai.py --config configs/sites_smoke.yaml --checkpoint artifacts_sites/runs/vit3d/best.pt
```

> **Read the exit codes, not the tables.** This is 14 scans for 1 epoch. Its
> AUROC, its deletion curves and its enrichment numbers are all noise, and
> quoting any of them would be a mistake. It exists only to prove the pipeline
> runs end to end. Thirteen real bugs were found this way — none of which the
> unit tests caught, because they all lived in the seams between scripts.

Delete `artifacts_sites/runs/` before the real run, so smoke checkpoints cannot
later be mistaken for results.

---

## 4. The real run

### 4a. Measure every site — about 30 min, CPU only

```bash
python scripts/build_implant_labels.py --config configs/sites.yaml
```

This writes `artifacts_sites/sites_toothfairy3.csv`. Sanity-check it:

```bash
python -c "from src.data.site_dataset import load_sites; d=load_sites('artifacts_sites/sites_toothfairy3.csv',targets=['needs_implant','feasible'],jaws=['lower'],methods=['teeth']); n=d[d.needs_implant==1]; print(len(d),'sites',d.patient_id.nunique(),'patients'); print(len(n),'need an implant |',int((n.feasible==0).sum()),'not feasible')"
```

**At v3.0.1 that prints:**

```
6787 sites 486 patients
709 need an implant | 413 not feasible
```

Unlike the test count, this one is worth stopping over. It depends on the
dataset and on the label rules, not on how much code has been written since,
so a difference means one of those two moved. Check the release notes for the
tag you cloned before you continue — if the label rules changed deliberately
the notes will say so and the new figures are correct; if they did not, the
dataset is not the one this was built against. Either way, do not start paying
for a GPU until you know which.

These numbers moved once already, and the reason is worth knowing before you
trust them. An audit found occupancy was decided by centroid distance computed
as `hypot(dx, dy)` -- with z absent -- over teeth from BOTH jaws, so an upper
molar directly above a lower site claimed it. 414 mandibular sites were marked
occupied by a maxillary tooth while their own tooth was missing from the mask.
Occupancy is now a label lookup and `needs_implant` went 530 -> 709.

### 4b. Build the cache — ~19 min on 30 vCPU, ~27 GB

```bash
python scripts/build_site_cache.py --config configs/sites.yaml
```

Resumable: re-running skips whatever already exists. Use `--force` only when you
actually intend to rebuild from scratch.

**Ten of the 532 scans are refused for ambiguous anatomical orientation**,
leaving 522. That is the builder working: a wrong orientation inverts every
measurement while still producing clinically plausible numbers, so it declines
rather than guessing.

### What the rented box actually did, measured on the fold-0 run

- **`/dev/shm` was 64 MB** and could not be remounted inside the container. A
  batch of 64 patches at 96³ float32 is ~226 MB, and DataLoader workers pass
  batches through shared memory, so `--num-workers > 0` dies with
  `unable to allocate shared memory`. PyTorch's `file_system` sharing strategy
  does **not** help; it still routes through `/dev/shm` on Linux. Use
  `--num-workers 0`.
- That costs almost nothing here, because augmentation is disabled for the site
  task and the loader only slices a memory-mapped array. Measured: GPU bursting
  to 97% then idling, CPU 96.5% idle — the bottleneck is disk. Epoch time fell
  118 s to 83 s as the page cache warmed.
- **A stopped pod restarts into a new container.** `/workspace` persists;
  `/opt/venv` and `/root` do not. A run died on `ModuleNotFoundError: nibabel`
  and `TOOTHFAIRY3_ROOT` was gone from `.bashrc`. After any restart:
  `pip install -r requirements.txt` and re-export the dataset root.
- Confirmed adequate: 24 GB VRAM (peak 18 GB at batch 64), 30 vCPU. RAM
  advertised at 64 GB was delivered as 27 GB and did not bind.

### Verify large transfers by checksum, not by size or file count

Three uploads through a JupyterLab browser arrived **truncated at exact
powers of two** — a 294 MB archive short by exactly 2,097,152 bytes, cache files
landing at exactly 1 MB and exactly 5 MB. The same files over SSH were
byte-identical by MD5. Two further faults were caught only by checksum: one
cache file differed in content while matching in size exactly, which would have
fed corrupted voxels into training with no error; and a hung `tar` kept writing
for about an hour after the transfer script reported success.

```bash
md5sum artifacts_sites/cache/toothfairy3/*.npy > local.md5   # then compare on the box
```

### 4c. Train the five folds — the expensive part

```bash
for k in 0 1 2 3 4; do python scripts/train.py --config configs/sites.yaml --fold $k --folds 5; done
```

Each fold writes `artifacts_sites/runs/cv_fold$k/`. Splits are **by patient**,
never by site — two sites from one jaw must never straddle a split.

Then pool them, so every case is predicted once by a model that never saw it:

```bash
python scripts/pool_cv.py --config configs/sites.yaml --folds 5
```

### 4d. Explainability

Sample sizes come from the `xai:` block in `configs/sites.yaml` — 200 cases, 200
GradientSHAP samples, 30 randomisation cases. **Do not pass `--n-cases` or
`--shap-samples` to shrink them**; those are the sizes that make a claim.
GradientSHAP at 24 samples measured a relative standard error of 0.5689, which
is not a result, and `run_xai` will warn you if it drops below 128.

Budget from a measured rate: faithfulness ran **47 s per case on CPU** at smoke
scale, so 200 cases is ~2.6 h on CPU and considerably less on the GPU. Every
script now logs `N/M cases done` with a per-case time, so you can extrapolate
after the first case rather than guessing.

**One extra thing to run while you are here.** The model-randomisation table in
`REPORT.md` §C9 has been measured on the real model with the rebuild fix in
place, so nothing in it is provisional any more. What it still lacks is
intervals: `--only-randomization` re-runs the cascade alone and the report now
prints a patient-clustered interval per method, which is what turns the ordering
from nominal into supported.

```bash
python scripts/run_faithfulness.py --config configs/sites.yaml --checkpoint artifacts_sites/runs/cv_fold0/best.pt --only-randomization
```

About an hour. Send the result back with everything else — it decides whether a
headline claim in the paper stands as written.

All five use fold 0's checkpoint.

```bash
python scripts/run_xai.py --config configs/sites.yaml --checkpoint artifacts_sites/runs/cv_fold0/best.pt
```

```bash
python scripts/run_faithfulness.py --config configs/sites.yaml --checkpoint artifacts_sites/runs/cv_fold0/best.pt
```

```bash
python scripts/run_localization.py --config configs/sites.yaml --checkpoint artifacts_sites/runs/cv_fold0/best.pt
```

```bash
python scripts/run_adaptive.py --config configs/sites.yaml --checkpoint artifacts_sites/runs/cv_fold0/best.pt
```

```bash
python scripts/make_figures.py --config configs/sites.yaml --checkpoint artifacts_sites/runs/cv_fold0/best.pt
```

Add `--deterministic` to any of them if you need bit-reproducible attributions;
it is slower.

### 4e. Two CPU-only runs that need no GPU and no checkpoint queue

Both can run on a laptop, or on the box while folds train. Neither costs GPU
time, and between them they decide how much of the XAI section survives review.

**The geometric baseline.** A threshold-and-measure estimator scored through the
same `regression_metrics` and `threshold_sensitivity` as the model, so the two
tables compare row for row:

```bash
python scripts/run_geometric_baseline.py --config configs/sites.yaml --fold 0 --split val
```

It reads only cached CT intensities, never a segmentation mask -- which matters,
because the ground-truth labels come from the mask, and an estimator that read
the mask too would reproduce them and prove nothing. If it lands near the ViT,
the architecture argument is over; better to know before writing results around
it.

**The deletion/insertion diagnosis.** The fold-0 run put all four methods within
0.013 of each other, with deletion roughly equal to insertion, which is what a
flat response looks like. Two suspects, and both are now flags rather than
constants:

```bash
python scripts/run_faithfulness.py --config configs/sites.yaml --checkpoint artifacts_sites/runs/cv_fold0/best.pt --baseline mean --score deviation
```

`--baseline mean` destroys geometry, where the default blur (sigma 4 voxels =
1.2 mm) removes texture and leaves a crest-to-canal distance perfectly readable.
`--score deviation` integrates distance from the full-input prediction instead
of the raw output, because a millimetre head is not a confidence and has no
reason to fall when evidence is removed. Both settings are written into every
row of `results_faithfulness.csv`, so two runs can never be confused.

Run the default settings too, and report both. If the spread stays at 0.013 with
a mean baseline, the null result is real and belongs in the paper.

### Anything produced before v3.1.0 has to be re-run

Three scripts converted model outputs with a bare `sigmoid` across the whole
output row. That is right for the binary head and wrong for the two millimetre
heads, and it was wrong silently:

| Output | What was wrong before v3.1.0 |
|---|---|
| `cv_pooled_metrics.json`, `cv_predictions.csv` | millimetre predictions squashed into (0, 1) and never un-standardised, so the pooled MAE was roughly the mean of the target rather than the model's error |
| `calibration/calibration.json` | temperature fitted against millimetre targets; `T` came back NaN and `ece_before` above 1 |
| `results_ablations.csv`, `results_pareto.csv` | every calibrated probability and uncertainty NaN, so the confidence gate never escalated a case and the Pareto sweep collapsed to a single point; rows were also paired with the wrong case's prediction |
| `figures/case_manifest.csv` | captions reported a bone height as a confidence |

**Check before you trust an existing file:** `ece_before` must be `<= 1`, and
`temperature` must be finite. If either fails, that run predates the fix.

```bash
python -c "import json;d=json.load(open('artifacts_sites/calibration/calibration.json'));print(d['temperature'], d['ece_before'])"
```

`results_faithfulness.csv` is affected differently: the deletion and insertion
curves for a millimetre target were read through a sigmoid, which cannot
reverse one curve but can reorder two methods, since an AUC is an integral. The
localisation results do not depend on any of this.

---

## 5. How to tell whether it worked

`train.py` prints these before epoch 0. **A model that has learned nothing
scores exactly this**, so compare against it and never against zero:

```
needs_implant       BCE floor 0.3348   AUROC floor 0.500   AP floor 0.1045
available_height_mm MAE floor 6.91 mm  RMSE floor 7.97 mm
ridge_width_mm      MAE floor 3.58 mm  RMSE floor 4.40 mm
```

| What you see | What it means |
|---|---|
| MAE at or above its floor | that head learned nothing. Not a bug — a result. Report it |
| Macro AUROC CI **includes 0.500** | indistinguishable from chance. Say so plainly |
| AUROC above 0.5 with a CI that excludes it | a real signal |

The floor **moves with the label set**, and it moved when the audit fixes
changed the labels: it was 1.2026 before, and the superseded three-label task's
was 1.0652. Quoting any of them against another is a category error. `train.py`
solves for the current one and prints it; use what it prints.

`needs_implant` has 10.4% prevalence, so **accuracy is meaningless on it** —
quote AUROC and AP with confidence intervals. For the millimetre heads quote MAE
in millimetres beside its floor; an MAE with no floor next to it is unreadable.

Every interval must be **bootstrapped over patients, not rows**: one jaw
contributes ~14 sites, so resampling rows measures within-patient repeatability
and reports it as between-patient uncertainty -- and narrows the interval by
roughly the square root of the sites-per-patient ratio while saying nothing
about having done so.

`run_faithfulness.py` and `run_localization.py` now do this for you
(`runner.clustered_ci`) and print the interval beside every method. **Read what
they refuse to say as carefully as what they say**: the randomisation table
names the pairs whose intervals overlap, and localisation declines to name a
best localiser when the top method is not separated from the rest. There are no
pass/fail labels anywhere, because Adebayo et al. define no cutoff -- the
intervals are the claim.

---

## 6. What to send back

```
artifacts_sites/runs/cv_fold*/metrics.json
artifacts_sites/runs/cv_fold*/history.csv
artifacts_sites/cv_pooled_metrics.json
artifacts_sites/cv_predictions.csv
artifacts_sites/results_*.csv
artifacts_sites/results_geometric_baseline_fold*.json
artifacts_sites/calibration/calibration.json
artifacts_sites/xai_*.csv
artifacts_sites/figures/
```

Send `results_faithfulness.csv` from **both** faithfulness runs, renamed so the
settings are visible -- the default and the `--baseline mean --score deviation`
diagnosis. The settings are in every row, but a filename that says so saves the
person reading them from having to check.

A few hundred MB in total. **Leave the `.pt` checkpoints and `cache/` behind**
unless asked — they are many gigabytes, and both are reproducible from the
files above.

---

## 7. Traps that have already cost this project time

- **Do not trust the NIfTI affine.** ToothFairy3's header states one orientation
  and its voxel data disagrees. The code determines up from down using anatomy
  instead. If you "fix" this by trusting the header, every measurement inverts —
  and the numbers will still look clinically plausible.
- **Do not raise `patch_size`.** The conv stem has stride 2, so one token covers
  `2 × patch_size` input voxels. At `patch_size: 8` a token spans 4.8 mm — wider
  than the nerve canal the explanations exist to resolve. `4` gives 2.4 mm. This
  was got wrong once already.
- **Do not set `target_spacing`.** `null` means native 0.3 mm, which is the whole
  reason for running this on rented hardware.
- **Do not quote anything produced under `configs/sites_smoke.yaml`.**
- **`artifacts/` is a different, superseded task.** Read its `SUPERSEDED.md`. Its
  pooled AUROC of 0.835 answers *"is an implant already present?"*, which is not
  this project's question. Never mix the two directories.

---

## 8. If you get stuck

`HANDOFF.md` is the full technical briefing and `REPORT.md` Part C is the
detailed log of the current task, including why each decision was made. Both
live alongside the code repository rather than inside it.
