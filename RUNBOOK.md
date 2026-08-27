# RUNBOOK — running this on a rented GPU

Written for someone who has **not** worked on this project and is running it on
a machine that is not the one it was developed on. Follow it top to bottom.

Everything before Step 4 is free and takes about fifteen minutes. **Do all of it
before you start paying for a GPU.**

---

## 0. What you are running

For each tooth position in a CBCT scan, the model predicts two things:

| Label | Question |
|---|---|
| `needs_implant` | Is this site missing a tooth that should be replaced? |
| `feasible` | Is there enough bone, clear of the nerve, to actually place one? |

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
| Disk | **120 GB free** | 28 GB dataset + ~52 GB cache + checkpoints and headroom |
| RAM | 32 GB | cache building holds whole volumes in memory |
| Python | 3.11 or 3.12 | both are gated in CI |

The cache is the surprise: **~100 MB per scan × 532 scans ≈ 52 GB**, because
this pipeline deliberately does *no* downsampling. Do not rent a box with a
50 GB disk.

---

## 2. Setup

```bash
git clone https://github.com/AGB2210/Adaptive-Multi-XAI-3D-ViT-for-Dental-Implant-Diagnosis-Support.git capstone-code
cd capstone-code
python -m venv .venv && source .venv/bin/activate
```

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

Expect `323 passed`.

```bash
python scripts/check_imports.py
```

Expect 49 modules.

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

**Expect exactly:**

```
6787 sites 486 patients
530 need an implant | 327 not feasible
```

If these differ, something changed upstream. Do not continue — report it.

### 4b. Build the cache — 2–4 h, CPU-bound, ~52 GB

```bash
python scripts/build_site_cache.py --config configs/sites.yaml
```

Resumable: re-running skips whatever already exists. Use `--force` only when you
actually intend to rebuild from scratch.

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

### 4d. Explainability — 1–2 h

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

---

## 5. How to tell whether it worked

`train.py` prints these before epoch 0. **A model that has learned nothing
scores exactly this**, so compare against it and never against zero:

```
BCE floor    1.2026
AUROC floor  0.500
AP floor     0.0952
```

| What you see | What it means |
|---|---|
| Val loss stuck near **1.2026** | learned nothing. Not a bug — a result. Report it |
| Macro AUROC CI **includes 0.500** | indistinguishable from chance. Say so plainly |
| AUROC above 0.5 with a CI that excludes it | a real signal |

The floor **moves with the label set**. The superseded three-label task's floor
was 1.0652, and quoting that here would be a category error.

`needs_implant` has 7.8% prevalence and `feasible` 11.2%. **Accuracy is
meaningless at this imbalance** — always quote AUROC and AP with their
confidence intervals.

---

## 6. What to send back

```
artifacts_sites/runs/cv_fold*/metrics.json
artifacts_sites/runs/cv_fold*/history.csv
artifacts_sites/cv_pooled_metrics.json
artifacts_sites/cv_predictions.csv
artifacts_sites/results_*.csv
artifacts_sites/xai_*.csv
artifacts_sites/figures/
```

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
