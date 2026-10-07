# Adaptive Multi-XAI 3D ViT for Dental Implant Diagnosis Support

A 3D Vision Transformer written from scratch in PyTorch, and a multi-method
explainability layer that **measures whether its own explanations are truthful**
rather than assuming they are.

The explainability framework is the contribution. The clinical task is the
vehicle it is demonstrated on.

## What the model answers

For each tooth position in the lower jaw:

| | |
|---|---|
| **needs_implant** | nothing occupies this position — no tooth, no crown, no bridge pontic, no existing implant |
| **available_height_mm** | how much bone there is, crest of the ridge down to the inferior alveolar canal |
| **ridge_width_mm** | how wide the ridge is below the crest |

Feasibility — *is there enough bone to place an implant safely, clear of the
nerve* — is **not** one of these. It is a rule applied to the two millimetre
outputs at inference, so a revised clinical threshold is a re-score rather than a
retrain. See "`feasible` is not a label" below.

Labels are derived geometrically from ToothFairy3's own voxel masks. Every row
stores the measured millimetres beside the verdict, so revising a clinical
threshold is a re-score of a CSV, not a reprocess of 28 GB.

## Status

| Stage | State |
|---|---|
| Label builder, arch fitting, site measurement | **Done** — validated against real anatomy |
| Site labels built | **Done** — 6,781 mandibular sites, 486 patients |
| Native-resolution cache builder, patch dataset, training wiring | **Done** — the full cache is built: 522 volumes, 26.4 GB, ~19 min on the rented box |
| Pipeline run end to end on real scans | **Done** — all five XAI stages |
| Test suite | **Green** on Python 3.12, ruff clean — the one version CI gates. Older is refused; newer is not tested |
| XAI stack on the site task | **Done.** Randomisation and faithfulness stand; **localisation is withdrawn pending a re-run** — its anatomy masks were cut 7.2 mm from the box the model was shown |
| Training on the site task | **All five folds done and pooled** on a rented RTX 4090. Pooled AUROC 0.9535, patient-clustered 95% CI [0.9424, 0.9636], over 6,781 sites from 486 patients, each scored once by the model that never saw it |
| Inference app | **Built and tested** — FastAPI server and browser page, started by `start.bat`. With a mask it is the measured pipeline; the image-only path waits on a trained localiser (see "Known limitation") |
| Baselines | A CNN was measured on fold 0 and is ahead there; architecture selection is outside this project's scope, so it is recorded and not pursued. The geometric estimator is written and never run on real scans |
| Guide sign-off on clinical thresholds | **Pending** |

**The headline result is a negative one, and it is stated here rather than
buried.** Feasibility agreement is 0.66–0.71 across all five folds, biased
toward calling sites feasible that are not — the unsafe direction beside a
nerve. Mean absolute error is 3.4 mm against a rule whose entire safety margin
is 2 mm, so this is arithmetic rather than an unlucky split. **What is
defensible is a screening aid that flags sites for human review, not a decision
tool.**

Both pipelines now produce a `CaseSet` (`src/xai/runner.py`), so the five XAI
scripts no longer care which task they are on: a case is a whole scan or a
`patient#tooth` pair, and `CaseSet.load` returns the right input either way.

The localisation metric changed with it. It used to ask *"does the explanation
point at the implant?"* — which metal passes trivially, and is how Integrated
Gradients scored 86x chance while failing the randomisation check. It now asks
*"the model says NOT feasible; does the explanation point at the inferior
alveolar canal?"* The canal is a dark tube inside bone, and it is **small**:
measured over the 485 scored patches its median share is **0.48%**, with the
middle 90% running 0.014% to 0.92% and a full spread of 0.0002% to 1.40%. An
edge detector cannot find it by accident — but a token spans 2.4 mm and the
canal is under 3 mm across, which is the resolution caveat §C8i.5 of `REPORT.md`
attaches to every enrichment figure.

## Quick start

```bash
export TOOTHFAIRY3_ROOT=/path/to/ToothFairy3     # never hard-coded

# 1. sites -> labels  (CPU, ~20 min, reads masks only)
python scripts/build_implant_labels.py --config configs/sites.yaml

# 2. scans -> native-resolution cache  (26.4 GB over 522 scans, ~19 min, rented box)
python scripts/build_site_cache.py --config configs/sites.yaml

# 3. sanity gate before any real run (2 min, CPU)
python scripts/train.py --config configs/synthetic.yaml --synthetic

# 4. train, one fold at a time
python scripts/train.py --config configs/sites.yaml --fold 0 --folds 5
```

Run the synthetic gate first. It plants a bright blob at a known site per label
— a task the model must be able to learn — so a failure there means the code is
broken rather than the problem being hard.

## The app

A browser app that does what the model is for, on one scan at a time.

**On Windows, double-click `start.bat`.** It finds Python, starts the server and
opens the page. The first time, if the packages are missing, it builds a `.venv`
in this folder and installs them -- reusing a PyTorch that is already installed
rather than downloading a second one, and taking the CUDA build when an NVIDIA
driver is present. **Ctrl+C, or closing its window, stops the server and frees
the port**, also in the middle of an explanation.

Anywhere else, or by hand:

```bash
pip install -r requirements-app.txt
python -m app --open                           # http://127.0.0.1:8000
```

If the app is already running on the port, either way opens the page on that
instance instead of failing. If another program holds the port, the next free
one is used and printed -- unless `--port` named it, which is refused rather
than quietly changed. Arguments after `start.bat` are passed through:
`start.bat --port 9000 --device cpu`.

1. **Models** — pick a site model's `.pt` in the browser, with any of its
   companions: `calibration.json` (temperature and the fitted confidence gate,
   written beside the checkpoint by `run_adaptive.py`), `best_val_metrics.json`
   (decision threshold and validation MAE), `cv_folds.json` (whether a patient
   was in that model's training data). No path is configured anywhere. A file
   is loaded and checked before it is kept, and refused with a reason if its
   architecture, heads or units do not add up.
2. **Scan** — a ToothFairy3-format CBCT (`.nii` / `.nii.gz`), with its mask if
   there is one. With a mask, sites are located and measured by the label
   builder's own `score_one`, and the measured values appear beside each
   prediction. Without one, a trained site localiser finds them (below).
3. **Result** — the fourteen lower sites on an axial projection and a tooth
   chart, each judged *no implant needed / feasible / not feasible / borderline*.
   Borderline means the prediction is within the model's own validation error of
   a threshold. Changing a threshold re-scores every site at once, because
   feasibility is a rule on the predicted millimetres, not a model output.
4. **Explain** — the adaptive layer on one site: the confidence gate routes it
   to attention rollout alone or to all four methods with agreement-weighted
   fusion, and the fusion weights, IG completeness error and SHAP standard
   error are shown with the maps.

A scan whose analysis failed, or was cut short because the server stopped, keeps
its uploaded files: the page says why it has no result and offers **Analyse
again**. **Remove scan** deletes a scan and everything computed from it.

**Nothing in the app is a second implementation.** An uploaded scan is prepared
by `src/data/scan.prepare_volume`, the function `build_site_cache.py` writes the
training cache with, and comes out byte-identical to the cached volume; patches
are `patch_centre` / `cut_patch`; outputs go through `to_report_units`;
feasibility is `derived_feasible`; the gate and fusion are `src/xai/adaptive`.
`tests/test_inference.py` pins each of those seams.

**Nothing in the app is sized for a machine.** Jobs run concurrently, every
site of a scan is predicted in one forward pass, and every model added stays
loaded on the device until it is removed. Two things still run one at a time,
and both are correctness rules: jobs on the *same scan* (they write the same
files), and work inside the *same model* (the attribution methods hook its
blocks and switch its attention capture on and off, so two of them in one model
object would read each other's activations). The XAI step counts and batch
sizes in `configs/app.yaml` are the research scripts' own, because those are
the settings the maps were measured at.

Checkpoints written by `train.py` now carry their architecture
(`model_config`). Older ones lack it, and `num_heads` cannot be read off the
weights, so for those the app takes it from a config and checks every other
field against the tensor shapes.

## Why patches, not whole heads

|  | detail | field of view | samples |
|---|---|---|---|
| 128³ @ 1.0 mm | 3.3× blurred | whole head | 532 |
| 256³ @ 0.5 mm | 1.7× blurred | whole head | 532 |
| **96³ @ 0.3 mm** | **native** | one tooth site | **6,781** |

The patch is sharper *and* 19× smaller than the 256³ volume. The structure the
model must respect is the inferior alveolar canal, 2–3 mm across: at 1.0 mm that
is two or three voxels, and no attribution method can point at something it
cannot resolve.

**Mind the conv stem.** It has stride 2, so a token spans `2 × patch_size` input
voxels, not `patch_size`. This was got wrong once — `patch_size: 8` was
documented as 2.4 mm per token and measured at **4.8 mm**, wider than the nerve
itself:

| | tokens | mm per token |
|---|---|---|
| patch 8 | 6³ = 216 | 4.8 mm — too coarse |
| **patch 4** | **12³ = 1,728** | **2.4 mm** ✓ |

Check `model.grid_size` rather than doing the arithmetic; `run_xai.py` prints it
from the checkpoint.

## Mandible only

Measured over all 522 usable scans, of the sites that need an implant:

```
mandible    884 needed,  826 measurable   93.4%
maxilla    2682 needed,   36 measurable    1.3%
```

98% of the unmeasurable maxillary sites have no bone voxels at all — after an
upper tooth is lost the ridge resorbs, the sinus pneumatises, and ToothFairy3's
`UpperJaw` mask does not cover the remnant. Training on them would reproduce an
annotation gap as a clinical verdict.

Nothing important is lost: the inferior alveolar canal is annotated in **every**
scan, and nerve clearance limits 376 of the 413 infeasible sites — which is both
the real clinical danger and a target an edge detector cannot fake, because the
canal is a dark tube inside bone rather than a bright edge.

## Versions

**Always take the latest tag.** No version number is written here as "the
current one", because that line goes stale the moment the next one is cut:

```bash
git clone https://github.com/AGB2210/Adaptive-Multi-XAI-3D-ViT-for-Dental-Implant-Diagnosis-Support.git
cd Adaptive-Multi-XAI-3D-ViT-for-Dental-Implant-Diagnosis-Support
git checkout "$(git tag -l 'v*' --sort=-v:refname | head -1)"
cat VERSION
```

`--sort=-v:refname` orders by version rather than push date, so `head -1` is the
highest tag and not merely the newest. The releases page marks the same one
"Latest".

The rest of this section is about which *older* tags exist and why none of them
may be used or quoted against another. That does not go stale.

**Do not use v1.0.0.** It predates an audit that found two faults in the label
builder, and the release tarball still contains them: occupancy matched teeth
across jaws (414 mandibular sites were held by a maxillary tooth), and ridge
width measured a single cortical plate wherever a tooth was present (a 6.00 mm
ridge measured 1.80 mm). Both change the labels, so nothing measured under
v1.0.0 can be compared with anything measured after it. See `REPORT.md` C8c.

| | v1.0.0 | v2.0.0 | v3.0.0 |
|---|---|---|---|
| `needs_implant` | 530 | 709 | 709 |
| feasibility | classified | classified | **regressed, thresholded at inference** |
| floors | BCE 1.2026 | BCE 0.9145 | BCE 0.3348 + MAE 6.91 / 3.58 mm |

**v3.0.0** replaces the `feasible` classification head with two millimetre heads.
`feasible` is now computed from the predictions and the config, so revising the
12 mm rule is a re-score rather than five folds of retraining -- which matters,
because that rule moves a third of the answers. Results are again incomparable
with what came before, and the config schema, `predict`, and `Trainer` all
changed signature.

Three majors in a day is not inflation. Each marks a point where a number
produced before it stops meaning the same thing as one produced after -- v2.0.0
changed the labels, v3.0.0 changed what the model predicts -- and the floors move
each time, which is the practical test of whether a comparison is legitimate.
**Every number in the paper must carry the tag it was measured under.**

## What the model predicts

| Head | Kind | Question |
|---|---|---|
| `needs_implant` | binary | Is this socket empty? |
| `available_height_mm` | **millimetres** | How much bone is there, crest to canal? |
| `ridge_width_mm` | **millimetres** | How wide is the ridge? |

**`feasible` is not a label.** It is a rule applied to the two measurements, and
it is applied at *inference*, from configuration:

```
feasible = available_height_mm >= 12.0 and ridge_width_mm >= 6.0
```

That matters because the threshold is the largest single lever in the project.
Over the 709 mandibular sites that need an implant:

```
height rule 10 mm -> 266 infeasible (37.5%)
height rule 12 mm -> 390 infeasible (55.0%)
height rule 14 mm -> 503 infeasible (70.9%)
```

A 2 mm revision moves a third of the answers. As a classifier that revision costs
five folds of retraining; predicting millimetres it costs a re-score, and
`train.py` prints the whole sweep as a result rather than a risk. This is the
project's own rule -- *thresholds are configuration, never code* -- applied to
the model and not only to the label builder.

It is also the more useful output: "14.3 mm of bone here" tells a clinician which
fixture will fit, and stops the pipeline silently assuming a 10 mm one.

`needs_implant` stays binary because it is occupancy, with no millimetre quantity
underneath. Expect it to be easy -- "is there a tooth in this patch" is a simple
visual task -- so a high AUROC there is a sanity check, not a finding.

## The three numbers to quote a result against

```
needs_implant     BCE floor 0.3348   AUROC floor 0.500   AP floor 0.1045
available_height  MAE floor 6.91 mm  RMSE floor 7.97 mm
ridge_width       MAE floor 3.58 mm  RMSE floor 4.40 mm
```

`train.py` prints the floor before the first epoch. **It moves with the label
set** — the superseded three-label task's floor was 1.0652, and quoting it here
would be a category error.

## Running this on another machine

**`RUNBOOK.md` is the instruction manual** — setup, the pre-flight gates, the
full command sequence, what the numbers must be at each step, and what to send
back. It is written for someone who has never seen this project. Start there.

## Before renting a GPU, run the smoke config

```bash
python scripts/build_site_cache.py --config configs/sites_smoke.yaml --limit 14
python scripts/train.py            --config configs/sites_smoke.yaml --synthetic
python scripts/train.py            --config configs/sites_smoke.yaml --num-workers 0
python scripts/run_xai.py          --config configs/sites_smoke.yaml --checkpoint artifacts_sites/runs/vit3d/best.pt
```

It shrinks the task until it runs on a 4 GB laptop. **Thirteen** integration
faults were found this way, none of which a unit test caught, because they all
lived in the seams between components — including one that would have silently
reduced a five-fold cross-validation to a single checkpoint. **Nothing measured under it is a result** — a
one-epoch model predicts near-constant, so its curves are flat. Read the exit
codes, not the tables.

## From scratch, deliberately

`src/models/vit3d.py` imports only `torch`, `torch.nn`, `torch.nn.functional`.
No `timm`, no MONAI, no `torchvision`, no pretrained weights. Every attribution
method is implemented here too — no `shap`, no `captum`. This is a project
requirement, not a gap to be filled.

## Layout

```
src/data/      preprocessing, caches, augmentation, splits
               implant_sites  bone height / ridge width / nerve clearance, in mm
               dental_arch    where a tooth site is when the tooth is gone
               site_dataset   one sample per site, patches at native resolution
               scan           scan -> model-frame volume (cache build AND app)
               site_scoring   every site of one scan, from its mask
src/models/    vit3d (from scratch), cnn3d baseline (one fold, not pursued),
               geometric (threshold-and-measure baseline, CPU only),
               localiser (finds the lower sites on a scan with no mask)
src/train/     training loop, metrics with bootstrap CIs, localiser metrics
src/xai/       rollout, IG, GradientSHAP, Grad-CAM, LIME (ablation only),
               faithfulness, localisation, calibration, adaptive fusion
src/inference/ the app's path: checkpoint loading, scan preparation, sites,
               prediction and verdicts, explanation
app/           FastAPI server and the browser page (python -m app)
start.bat      one click on Windows: environment, server, browser
scripts/       build_implant_labels, build_site_cache, train, evaluate,
               run_xai, run_faithfulness, run_localization, run_adaptive,
               run_geometric_baseline, pool_cv, make_figures,
               build_localiser_cache, train_localiser, eval_localiser
```

### The superseded detection pipeline

`scripts/build_labels.py`, `scripts/build_cache.py`, `src/data/dataset.py` and
`configs/default.yaml`'s `task.labels` implement the earlier task — *is an
implant, crown or bridge already present?* It is kept because it still runs and
the XAI scripts still support it through the same `CaseSet`. It is not the
project's question, and **nothing published here is measured on it any more** --
the randomisation values that were are gone, for the reason given under "the
finding that survives the pivot". `configs/preprocess_256.yaml` belongs to the
same path: a whole-head alternative, deliberately set aside.

## Datasets are configuration, never code

```yaml
data:
  datasets:
    toothfairy3:
      root: ToothFairy3
      volume: "imagesTr/{pid}_0000.nii.gz"
      labels: "labelsTr/{pid}.nii.gz"
  primary: toothfairy3
```

Roots are overridden per machine by `<NAME>_ROOT` env vars (`TOOTHFAIRY3_ROOT`),
which always win over the file. No path in this repo points at anyone's disk.

## Clinical thresholds live in the config

`configs/default.yaml` → `sites:`. The builder refuses to run if any is missing,
rather than inheriting a default nobody has reviewed.

```yaml
sites:
  min_height_mandible_mm: 12.0   # 10 mm implant + 2 mm nerve safety zone
  min_height_maxilla_mm:  10.0   # class A, 1996 Sinus Consensus Conference
  min_width_mm:            6.0   # 3.75-4 mm implant + >=1 mm each plate
```

Sources are cited in the config comments. **They are defaults, not clinical
sign-off** — the project guide reviews them before publication.

## Things that look like improvements and are not

- **Do not import a pretrained backbone.** From-scratch is the spec.
- **Do not trust the NIfTI affine for orientation.** ToothFairy3 reports LPS and
  the voxel data says the opposite. `superior_sign` reads it off the anatomy,
  using only cues validated at 100% agreement over 40 scans.
- **Do not add "lower teeth sit above the nerve" as an orientation cue.** It was
  measured at 82%, and jawbone-over-canal at 46%. The canal climbs toward the
  mandibular foramen.
- **Do not split sites at random.** 28 sites from one scan share anatomy, field
  of view and annotator. Split by patient.
- **Do not treat an unmeasurable site as infeasible.** That trains the model to
  reproduce our measurement failures as clinical findings.
- **Do not snap a site to the nearest spot with more bone.** It raises the
  feasible count and is tuning for a nicer number.
- **Do not delete the `store_attention` path** in `Attention` — the fused kernel
  never materialises the attention matrix, and Attention Rollout needs it.
- **Do not set `WEIGHT_METRIC == EVAL_METRIC`.** Fitting the fusion on the metric
  you then report is circular; `fuse()` raises to stop it.
- **Do not lower Integrated Gradients below 256 steps.** If memory is tight, cut
  the batch size — that costs time, not correctness.
- **Do not use a zero baseline for IG** on z-scored volumes: zero is a real
  tissue value, not absence.
- **Do not print test metrics** outside an explicit `--test` run.
- **Do not compare a loss to a floor computed for a different label set.**

## The finding that survives the pivot

A model-randomisation check (Adebayo et al. 2018) destroys the network stage by
stage and measures how much each explanation changes. A faithful map must
decorrelate; **Integrated Gradients changes least, and is therefore the least
faithful method here.** It produces nearly the same map from a trained network
and a randomised one, which is a property of the method rather than of the task.

**No table of values is published here on purpose.** An earlier version of this
section carried per-method numbers measured on the Part B *detection* model,
before a fault in the cascade's rebuild path was fixed. They described a
different network on a different task, and their ordering was not merely stale
but reversed. The current values belong to the site model and are reported with
their patient-clustered intervals in `RESULTS.md` §5 and `REPORT.md` §C9 — the
two documents that also carry the caveats they must be read with.

## Known limitation

Site positions are fitted to the ground-truth masks. That is legitimate for
training and for measuring this model, and on its own it is **not a deployable
pipeline** — a new patient arrives with an image and no segmentation.

`src/models/localiser.py` is the site-detection step that was missing: a 3D
U-Net predicting the fourteen lower sites, their in-view status and the scan's
orientation from the image alone, trained on the same folds as the site model
(RUNBOOK 4f). **It has not been trained on the cohort yet, so it has no measured
error.** Until `eval_localiser.py` has run, the image-only path in the app is
built and tested but unmeasured, and the mask path is the one to quote. Its
evaluation reports the number that matters: how much the site model's height
and width error grows when its patches are placed by the localiser instead of
the mask.

## CI

Seven gates on every push, cheapest first, on Python 3.12:

```
version is the single source of truth   VERSION and pyproject must agree
every module imports                    catches a script the tests never reach
ruff                                    rules pinned in pyproject, linter pinned in CI
pytest
every config resolves                   out_dir must sit under artifacts_dir
training runs end to end on synthetic    config -> loaders -> loss -> metrics -> checkpoint
predictions come back in millimetres     on a REAL checkpoint, not a mock
```

The last three are the expensive ones and the reason they exist: each catches a
fault that lived in a seam, produced a plausible number, and failed no unit
test. The millimetres gate asserts on a trained checkpoint that the binary head
reads 0.5 at logit zero and each millimetre head returns its training mean --
which is what a sigmoided millimetre head cannot do.
