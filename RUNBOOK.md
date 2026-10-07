# RUNBOOK — running this on a rented GPU

Written for someone who has **not** worked on this project and is running it on
a machine that is not the one it was developed on. Follow it top to bottom.

Everything before Step 4 is free and takes about fifteen minutes. **Do all of it
before you start paying for a GPU.**

---

## 0. What you are running

For each tooth position in a CBCT scan, the model predicts three things:

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

**Scope is the lower jaw only.** Of the sites that need an implant, 92.6% are
measurable in the mandible (819 of 884) and 1.3% in the maxilla (35 of 2,682) --
after an upper tooth is lost the ridge resorbs and ToothFairy3's `UpperJaw` mask
does not cover the remnant. Excluded on purpose; a finding, not an oversight.
See `README.md`.

---

## 1. Machine you need

| | Minimum | Why |
|---|---|---|
| GPU | 16 GB VRAM | 96³ patches at batch 64. 24 GB lets you raise the batch size |
| Disk | **80 GB free** | 28 GB dataset + 26.4 GB cache + checkpoints and headroom |
| RAM | 32 GB | cache building holds whole volumes in memory |
| Python | **3.12** | the one version gated in CI. Older is refused; newer is not tested |

**The cache is 26.4 GB over 522 volumes and builds in about 19 minutes** —
measured on the rented box during the fold-0 run, over the whole cohort. It is
float16 at native 0.3 mm; this pipeline deliberately does *no* downsampling, and
scans average ~50.6 MB.

This line used to read "~100 MB per scan × 532 ≈ 52 GB", and that was wrong in a
way worth keeping on the page. It was extrapolated from the 14 scans cached on
the development machine — which are the first 14 in alphabetical order, all of
them from the `ToothFairy3F` cohort, and F scans are among the largest at
512x512x262. The cohort is 63 F, 411 P and 48 S across the 522 usable scans, so
a head-of-list sample drawn entirely from the smallest and largest-file
sub-cohort overestimated the mean by a factor of two.

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
python --version                      # 3.12
python -m venv .venv && source .venv/bin/activate
```

**Python 3.12, not whatever the image ships.** It is the only version the code
is tested on. Every pipeline script refuses an older one on its first import; a
newer one runs but nothing gates it. If `python --version` prints something
else, build the environment from 3.12
instead of the last line above -- `conda create -n capstone python=3.12` and
`conda activate capstone`, or `uv venv --python 3.12`.

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

**What has to match the machine the files go back to, and what does not.**
The checkpoints are opened there by the inference app, on a different build of
everything. Three things carry across, and they are the only three:

| | On the machine that wrote this | What you need |
|---|---|---|
| Python | 3.12.10 | **3.12**, any patch release |
| PyTorch | 2.12.1, CUDA 12.6 | **2.2 or newer, any CUDA build.** A checkpoint is a dictionary of tensors and plain numbers, which every such version reads and writes alike |
| NumPy | 2.3.5 | **2.0 or newer** -- `requirements.txt` already insists |

You do not need to match the rest, and you do not need to report any of it:
section 6's pack command writes your Python, PyTorch, CUDA, GPU and package
versions into the archive, and the receiving machine loads every checkpoint
with its own build before anyone relies on one.

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
time. `0 failed` is the gate.

**One skip is expected, and following this runbook will not clear it.** The
summary line ends `1 skipped`, and that is correct. The skipped test checks that
the **1 mm whole-volume cache** is correctly detected as stale against the 256
config, and it skips when that cache is absent. Nothing on this run list builds
it -- step 4b builds the *site* cache, `artifacts_sites/cache/`, which is a different
directory for a different task. The 1 mm cache belongs to the superseded
whole-volume pipeline, so on a machine set up for the site task this test skips
permanently and correctly. CI reports the same skip for the same reason. Any
*other* skip is worth asking about.

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

```bash
python scripts/pack_handback.py --check-only
```

That last one loads the smoke checkpoint the way the app will, and it must
report `0 problem(s)` with `runs/vit3d/best.pt` under `ok`. If the app would
refuse a checkpoint this machine writes, this is where you find out, for free.
The long `MISSING` list is expected: the smoke gate trains no fold.

> **Read the exit codes, not the tables.** This is 14 scans for 1 epoch. Its
> AUROC, its deletion curves and its enrichment numbers are all noise, and
> quoting any of them would be a mistake. It exists only to prove the pipeline
> runs end to end. Thirteen real bugs were found this way — none of which the
> unit tests caught, because they all lived in the seams between scripts.

Delete `artifacts_sites/runs/` before the real run, so smoke checkpoints cannot
later be mistaken for results.

---

## 4. The real run

### What this run is for

The August run trained all five folds and ran every analysis, and what came
back was a report. The checkpoints and the result files stayed on the rented
machine. **This run exists to bring files back**, in three groups:

| | What | Why it is wanted |
|---|---|---|
| 1 | The five site checkpoints, each with its companion files | The inference app loads them (`README.md`, "The app"). It has never been run on a real checkpoint |
| 2 | A trained site localiser, with its evaluation | The app's path for a scan with no mask. The localiser has never been trained on the cohort |
| 3 | Every result file, as rows | The paper's tables are transcribed from a report. Twice a transcribed figure has failed when recomputed |

Section 6 packs all three with one command, after checking that the app will
accept every checkpoint. **Read section 6 before you start**, not when you
finish: it is what decides whether the run was worth its cost.

**Every command in this section has been run end to end, in this order, at this
version** -- on the fourteen scans a laptop holds, one epoch per fold, twenty
minutes in all. The archive it produced was verified on the receiving side and
loaded into the app, which analysed a real scan with its mask and without. So
the sequence is known to run and the files are known to fit; what a laptop
cannot tell you is how long each step takes at full size, and where this page
gives a duration it says where the figure came from.

### Which of two starting points you are at

**A. You still have the August machine, or a copy of its `artifacts_sites/`.**
Do not retrain and do not rebuild the cache. Update the code and keep the
directory:

```bash
git fetch --tags && git checkout "$(git tag -l 'v*' --sort=-v:refname | head -1)" && cat VERSION
```

```bash
pip install -r requirements.txt
```

```bash
python scripts/pack_handback.py --check-only
```

The last line lists what is there, whether the app accepts each checkpoint, and
what is missing. Then run only what is missing: the `pool_cv.py` line of 4c, all
of 4d, 4e, 4f, and section 6.

Three facts make that sound, each checked rather than assumed:

- **The labels are the same.** The table was rebuilt from the masks with the
  current code: 6,781 sites, 486 patients, 705 positives -- the figures the
  August run pooled over.
- **The cache is the same.** `build_site_cache.py` writes its settings beside
  the volumes and refuses a directory built with different ones; those settings
  have not changed since v3.4.0, and the transform is the one function the
  cache build and the app now share.
- **The checkpoints load.** A checkpoint from before v3.5.0 carries no
  `model_config`, so the app takes `num_heads` from `configs/app.yaml` and
  checks every other field against the weight shapes. `--check-only` tells you
  which case each file is in.

`cv_pooled_metrics.json` and any `calibration.json` on that machine are from
older code and must be regenerated -- `--check-only` flags both. Neither needs
a retrain: `pool_cv.py` needs the five checkpoints, `run_adaptive.py` one.

**B. That machine is gone.** Start at 4a and run everything in order. The new
checkpoints will not reproduce August's to the last digit -- GPU training is
not bit-reproducible -- so every number in the paper is then replaced by this
run's, which is the reason section 6 asks for all of it and not a subset.

### If the budget runs short, in this order

1. **Fold 0, complete.** `cv_fold0/best.pt`, then `run_adaptive.py` on it (its
   `calibration.json`), then the fold-0 localiser and its evaluation (4f, with
   `0` in place of the loop). That is the smallest set the app is whole with:
   one site model and one localiser that have both never seen fold 0's test
   patients.
2. **The rows.** The other four folds, `pool_cv.py`, and 4d.
3. **Calibration for folds 1-4**, minutes each (4d), and **localisers for folds
   1-4**. The first localiser fold tells you what the rest will cost; it has
   never been timed on real hardware.

A partial run is worth sending. Section 6's command states what is missing and
what each absence costs, so nobody has to work it out from a file list.

### What was already settled, and one question that was not

All five folds were trained and pooled in August, the XAI suite was run, and a
CNN baseline was measured on fold 0. That run cost about ₹455.

**A faithfulness re-run was done with a mean baseline, and WHICH FLAGS produced
it is unresolved.** It was reported as `--baseline mean --score deviation`, but
`score="deviation"` integrates `-|prediction - reference|`, which is non-positive
by construction -- measured at most -0.0037 over six seeds -- while every value
in the reported column is positive. So that column is not deviation-mode output
from this code. The likely explanation is `--baseline mean` with the default
`--score response`, which would mean the baseline alone produced the separation
and the monotonicity half of the diagnosis is still untested. If you ran it, say
which flags you used; if you are re-running, run both and label them.

So read this section as two different things depending on who you are. **If you
are reproducing the run**, everything below still applies exactly as written.
**If you are adding to it, nothing here needs a GPU**, and the three CPU-only
items this section used to list as outstanding were all run on 30-31 August
under v3.4.0 (`REPORT.md` §C8m):

| | |
|---|---|
| **Localisation, re-run** | Done. The anatomy masks had been cut **7.2 mm** from the box the model was actually shown -- `patch_masks` built its own patch centre and missed the quarter-shift `patch_centre` applies -- so every localisation figure from before v3.4.0 stays withdrawn. Corrected enrichment: Grad-CAM 1.92, attention rollout 1.60, Integrated Gradients 1.45, GradientSHAP 0.69; pointing rate 0.000 for every method; no best localiser named |
| **Geometric baseline** (§4e) | Done, on fold 0's test sites. Height MAE 6.60 mm against its own floor of 7.05; width 6.68 mm against a floor of 3.63, worse than predicting the median; it declines on 81.4% of sites |
| **Ceiling control** | Done, from a script that is not in this repository. Integrated Gradients measures 1.45 against a ceiling of 1.46; the ceilings for the two token-resolution methods came back below chance and bound nothing |

**What is missing is the data, not a run.** The result files of that run were
never transferred off the rented machine, so those figures are held as a report
and not as rows -- which is group 3 above.

Randomisation and faithfulness never open a mask, so those results were
unaffected by the mask fault.


### 4a. Measure every site — about 30 min, CPU only

```bash
python scripts/build_implant_labels.py --config configs/sites.yaml
```

This writes `artifacts_sites/sites_toothfairy3.csv`. Sanity-check it:

```bash
python -c "from src.data.site_dataset import load_sites; d=load_sites('artifacts_sites/sites_toothfairy3.csv',targets=['needs_implant','feasible'],jaws=['lower'],methods=['teeth']); n=d[d.needs_implant==1]; print(len(d),'sites',d.patient_id.nunique(),'patients'); print(len(n),'need an implant |',int((n.feasible==0).sum()),'not feasible')"
```

**At v3.0.1 that printed:**

```
6787 sites 486 patients
709 need an implant | 413 not feasible
```

**At v3.4.0 the label rules changed deliberately, and it prints:**

```
6781 sites 486 patients
705 need an implant | 409 not feasible
```

`measure_site` no longer clamps an impossible negative height to 0.0 mm; it
returns NaN and the row is dropped as unmeasurable, which removes 6 trainable
rows. Recorded at v3.7.3 by rebuilding the table from the masks: it differs from
the v3.0.1 table on ten rows, all of them a height of 0.0 that is now NaN, and
on nothing else.

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

### 4b. Build the cache — ~19 min on 30 vCPU, 26.4 GB

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

It writes `artifacts_sites/cv_pooled_metrics.json` and `cv_predictions.csv`.
Besides the pooled AUROC, the json holds the pooled millimetre errors with a
patient-clustered interval on each MAE, and **feasibility agreement at the
configured rule** -- over every site, and over the sites that need an implant --
with its interval and the share of sites called feasible that measure
infeasible. That last table is the project's headline result; before v3.8.0 it
was printed per fold and saved nowhere, so bring this file back.

`--model cnn3d` trains the CNN baseline through the same loop. **It is not on
the run list.** Fold 0 was measured and the CNN is ahead there; architecture
selection is outside this project's scope, since the contribution is the
explainability framework and the backbone is the vehicle it runs on. The flag is
documented because it exists, not because anything is waiting on it.

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

**Re-run this one even if nothing else needs re-running.** `REPORT.md` §C9 is
measured on the real model with the rebuild fix in place, but the CSV it came
from wrote **case** ids into a column named `patient_id`, so the
"patient-clustered" bootstrap was a row bootstrap — 30 randomisation cases from
14 patients, all counted as 30 independent draws. §C8j has the whole story.

The fix is in `run_faithfulness.py`, which now writes `case_id` and a real
`patient_id`, and `clustered_ci` refuses a grouping column that carries case
ids. **The intervals in §C9 were recomputed by splitting the case id**, which is
correct but is not the same as the pipeline producing them. This re-run makes
the file match the claim.

`--only-randomization` re-runs the cascade alone, about an hour:

```bash
python scripts/run_faithfulness.py --config configs/sites.yaml --checkpoint artifacts_sites/runs/cv_fold0/best.pt --only-randomization
```

Every XAI figure in this project is measured on the ViT, and none is claimed to
generalise to another backbone.

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

**Run them in that order.** `run_adaptive.py` reads `xai_runtime.csv`, which
`run_xai.py` writes; without it the Pareto table is skipped.

**Every one of these overwrites its own output.** `results_faithfulness.csv` is
one file whichever flags produced it, so a second faithfulness run replaces the
first. Rename a file as soon as its run finishes if another run of the same
script is coming -- section 6 packs anything named `results_<kind>*.csv`:

```bash
mv artifacts_sites/results_faithfulness.csv artifacts_sites/results_faithfulness_default.csv
```

**Then calibrate the other four folds, which takes minutes.** The app shows a
calibrated probability and routes on a fitted gate only for a checkpoint with a
`calibration.json` beside it, and the full `run_adaptive.py` above writes one
for fold 0 alone. `--calibration-only` fits the temperature and the gate on
that fold's validation split, writes the file beside the checkpoint, and stops
before the hour-long sweep:

```bash
for k in 1 2 3 4; do python scripts/run_adaptive.py --config configs/sites.yaml --checkpoint artifacts_sites/runs/cv_fold$k/best.pt --calibration-only; done
```

**Score both Grad-CAMs if you are re-running any of these.** `gradcam`, the one
every recorded figure was measured on, takes its channel weights from the CLS
token's gradient alone and sits on chance on the planted-signal task;
`gradcam_input` is the usual construction for a CLS-pooled ViT (`README.md`,
"What `gradcam` measures here"). It is not in the default ensemble, so name it:

```bash
python scripts/run_localization.py --config configs/sites.yaml --checkpoint artifacts_sites/runs/cv_fold0/best.pt --methods attention_rollout gradcam gradcam_input integrated_gradients gradient_shap
```

`run_faithfulness.py` takes the same `--methods`. The two maps are written under
their own names, so neither replaces the other's rows.

### 4e. A CPU-only run that needs no GPU and no checkpoint queue

It can run on a laptop, or on the box while folds train, and costs no GPU time.

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

### 4f. The site localiser — for the app's image-only path

The site model is measured on mask-derived positions. A scan with no mask needs
the localiser to find its sites, and the localiser is a model in its own right,
trained on the **same folds**:

```bash
python scripts/build_localiser_cache.py --config configs/localiser.yaml
```

Reads the site cache from 4b, so it takes minutes; under a gigabyte for the
whole cohort at 1.2 mm.

```bash
python scripts/train_localiser.py --config configs/localiser.yaml --fold 0
```

**Time that one before starting the rest.** The localiser has only ever been
trained for three epochs on fourteen scans, so what a fold costs on real
hardware is not known: up to 80 epochs over about 290 volumes of 128 x 128 x 80,
stopping early after 15 without improvement. It logs one line per epoch with
its seconds; multiply the first few by 80 for the ceiling. `--num-workers` is
not a flag here -- if the box's `/dev/shm` is small (see 4b), set
`localiser.num_workers: 0` in `configs/localiser.yaml` first.

```bash
for k in 1 2 3 4; do python scripts/train_localiser.py --config configs/localiser.yaml --fold $k; done
```

```bash
for k in 0 1 2 3 4; do python scripts/eval_localiser.py --config configs/localiser.yaml --checkpoint artifacts_sites/localiser_runs/cv_fold$k/best.pt --site-checkpoint artifacts_sites/runs/cv_fold$k/best.pt; done
```

Trained fewer than five? Put the folds you have in place of `0 1 2 3 4`. The
evaluation is minutes per fold and needs the site cache from 4b.

The evaluation pairs each localiser with the site model **of the same fold**, so
both have never seen the test patients. Read three things: the median 3D
position error with its patient-clustered interval; the **coverage** line -- how
many test sites the app would predict at all, the rest being ones the localiser
itself places out of view and the app shows as "no position"; and the end-to-end
table -- the site model's height and width MAE from mask-placed patches against
localiser-placed ones, on the sites the app predicts. That last row is the cost
of running without a segmentation, and it is the number to quote beside any
image-only result, **together with the coverage it was measured at**. The
"every site, at face value" rows include sites the app would decline, and
describe the localiser rather than the app.

Check the orientation line too. The app turns a scan over on the localiser's
word only if this checkpoint was at least 95% correct on orientation on its
validation patients (`app.min_orientation_accuracy`); below that it keeps the
default and says so on every result.

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

**A `calibration.json` from before v3.8.1 holds a temperature short of its
optimum.** The fit took 200 steps of 1% and stopped 10-20% short in log T,
always on the side of T = 1 -- measured against the exact minimum on logits
miscalibrated by a known factor. Fold 0's recorded 1.2566 would be about 1.30
by that shortfall; it has not been recomputed, because the validation logits
are not held. Which cases the gate escalates does not change -- with one binary
head the ordering of the margins is the same at any temperature. What changes
is the temperature itself, `ece_after`, and every calibrated probability shown.
Re-running `run_adaptive.py` refits it.

`results_faithfulness.csv` is affected differently: the deletion and insertion
curves for a millimetre target were read through a sigmoid, which cannot
reverse one curve but can reorder two methods, since an AUC is an integral. The
localisation results do not depend on any of this.

---

## 5. How to tell whether it worked

`train.py` prints these before epoch 0. **A model that has learned nothing
scores exactly this**, so compare against it and never against zero:

```
needs_implant       BCE floor 0.3337   AUROC floor 0.500   AP floor 0.1040
available_height_mm MAE floor 6.90 mm  RMSE floor 7.95 mm
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

**The reported statistic is the MEDIAN**, which is `clustered_ci`'s default, and
it is not incidental. Recomputing the randomisation table with the sample mean
instead moves every point estimate and, on the fold-0 data, reverses whether the
two least faithful methods are separated at all -- they overlap by 0.0002 under
the median in every bootstrap seed, and separate by about 0.066 under the mean
in every seed.

Two things follow. Quote the statistic beside the number, because a reader
cannot recover it from the table. And **do not reimplement this bootstrap.** Call
`clustered_ci`. Writing a fresh loop is how one pass over these documents came to
"correct" a table that had been right all along: the loop used a mean, the
published values were medians, and the disagreement looked like evidence the
numbers had never come from the CSV.

---

## 6. What to send back

One command, on the machine that ran it, when the run list is done:

```bash
python scripts/pack_handback.py
```

It does three things, in this order.

**It loads every checkpoint the way the app will.** Site models go through the
app's own loader and localisers through theirs, both refusing to unpickle
anything but plain data -- which is how the app opens a file picked in a
browser. A checkpoint the app would refuse is reported here as a `PROBLEM`,
while fixing it costs a command and not another rental.

**It lists what the run list should have produced and did not**, as `MISSING`,
each with what its absence costs. A partial run is a legitimate thing to send;
an unexplained gap is not.

**It writes the archive**, cache excluded, with a checksum for every file
inside it:

```
handback/capstone_handback_v<version>.tar.gz
handback/capstone_handback_v<version>.tar.gz.sha256
```

**Send both, by `scp` or `rsync`, never through a browser upload** -- see 4b
for what a browser did to three transfers. A few hundred megabytes: five
checkpoints at about 70 MB each are most of it.

Nothing is packed while a `PROBLEM` stands. Read the line: it names the file
and the reason. `--allow-problems` packs anyway and records them, for when the
choice is a flawed archive or none. `--check-only` reports and writes nothing,
and is worth running after 4c, before the long steps, as well as at the end.

### What the app reads from each file

The app is given files by a person picking them in a dialog, and it recognises
them **by name**. Send them under the names the scripts wrote; a renamed
companion is not found.

| File, beside `cv_fold<k>/best.pt` | Written by | What the app does with it | Without it |
|---|---|---|---|
| `best.pt` | `train.py` | The weights, the millimetre standardiser (`target_spec`), the output names and the architecture (`model_config`) | Nothing to load |
| `best_val_metrics.json` | `train.py`, with the checkpoint | The decision threshold for `needs_implant`, and the validation MAE that decides when a site is "borderline" | No threshold, and no borderline band |
| `calibration.json` | `run_adaptive.py` | The temperature for the probability shown, and the confidence gate that routes an explanation | Every result says it is uncalibrated and gated by a default |
| `artifacts_sites/cv_folds.json` | `train.py`, first fold | Whether an uploaded scan was in that model's training data | Every result says the patient's role is unknown |
| `localiser_runs/cv_fold<k>/best.pt` | `train_localiser.py` | Finds the sites on a scan with no mask. Carries its own validation error and orientation accuracy | A scan can be analysed only with its mask |

Three things about those files that are easy to get wrong:

- **Do not edit, re-save or convert a checkpoint.** `torch.save` of anything
  but the dictionary the training script wrote -- a whole model object, or a
  dictionary with a NumPy number in it -- loads on the machine that made it and
  is refused by the app. The pack command catches this; it cannot repair it.
- **Send `best.pt`, not `last.pt`.** `best.pt` is the epoch selected on
  validation and the one every companion describes.
- **Pair by fold.** The localiser for fold k and the site model for fold k have
  both never seen fold k's test patients. Any other pairing has.

The app turns a scan over on a localiser's word only if that checkpoint's
orientation accuracy on its validation patients is at least 95%; the pack
command prints the figure for each localiser and says when it is under.

### What the paper needs, beyond the checkpoints

All of it is in the archive when it exists. Listed so a gap can be traced to a
step:

```
artifacts_sites/sites_toothfairy3.csv                    4a   the labels the run trained on
artifacts_sites/cv_folds.json                            4c   the partition
artifacts_sites/runs/cv_fold*/metrics.json, history.csv  4c
artifacts_sites/cv_pooled_metrics.json                   4c   pool_cv.py -- must hold a "feasibility" block
artifacts_sites/cv_predictions.csv                       4c
artifacts_sites/xai_*.csv                                4d   run_xai.py
artifacts_sites/results_faithfulness*.csv                4d   one per run; see the renaming note in 4d
artifacts_sites/results_randomization*.csv               4d
artifacts_sites/results_agreement*.csv                   4d
artifacts_sites/results_localization*.csv                4d
artifacts_sites/results_ablations*.csv, results_pareto*  4d   run_adaptive.py
artifacts_sites/calibration/, figures/                   4d
artifacts_sites/results_geometric_baseline_fold*.json    4e
artifacts_sites/localiser_runs/cv_fold*/eval_test.json   4f   and end_to_end_test.csv beside it
```

**`patient_id` must hold a patient.** It should read `ToothFairy3F_011`, never
`ToothFairy3F_011#45` -- the site id belongs in `case_id`. Files written before
v3.3.0 have the case id in both, which turned a patient-clustered bootstrap
into a row bootstrap and cost a published claim (`REPORT.md` C8j). The pack
command reports such a file as a `PROBLEM`.

**Leave `cache/` and `localiser_cache/` behind.** Gigabytes, and rebuilt from
the dataset by 4b and 4f.

### On the machine that receives it

```bash
python scripts/pack_handback.py --verify path/to/capstone_handback_v<version>.tar.gz
```

It unpacks beside the archive, recomputes every checksum, loads every
checkpoint again **with that machine's own PyTorch** -- the two machines will
not have the same build, and this is the test of whether that matters -- and
prints which files to pick in the app's **Models** dialog for each fold.

The app itself is one click: `start.bat` on Windows, `python -m app --open`
anywhere. `README.md`, "The app", has the rest.

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
- **Do not send files without packing them.** The August run's results exist
  today only as a report, because the files were never moved and nothing said
  which ones mattered. `scripts/pack_handback.py` is one command.
- **Do not `torch.save` a checkpoint yourself.** See section 6.
- **`artifacts/` is a different, superseded task.** Read its `SUPERSEDED.md`. Its
  pooled AUROC of 0.835 answers *"is an implant already present?"*, which is not
  this project's question. Never mix the two directories.

---

## 8. If you get stuck

`HANDOFF.md` is the full technical briefing and `REPORT.md` Part C is the
detailed log of the current task, including why each decision was made. Both
live alongside the code repository rather than inside it.
