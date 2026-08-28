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
| **feasible** | there is enough bone to place one safely, clear of the inferior alveolar nerve |

Labels are derived geometrically from ToothFairy3's own voxel masks. Every row
stores the measured millimetres beside the verdict, so revising a clinical
threshold is a re-score of a CSV, not a reprocess of 28 GB.

## Status

| Stage | State |
|---|---|
| Label builder, arch fitting, site measurement | **Done** — validated against real anatomy |
| Site labels built | **Done** — 6,787 mandibular sites, 486 patients |
| Native-resolution cache builder, patch dataset, training wiring | **Done** — builder verified on 14 scans; the full ~52 GB cache builds on the GPU box |
| Pipeline run end to end on real scans | **Done** — all five XAI stages |
| 380 tests | **Green**, ruff clean |
| XAI stack on the site task | **Done** — scores against the nerve canal |
| Training on the site task | **Not started** — needs a rented GPU |
| Guide sign-off on clinical thresholds | **Pending** |

Both pipelines now produce a `CaseSet` (`src/xai/runner.py`), so the five XAI
scripts no longer care which task they are on: a case is a whole scan or a
`patient#tooth` pair, and `CaseSet.load` returns the right input either way.

The localisation metric changed with it. It used to ask *"does the explanation
point at the implant?"* — which metal passes trivially, and is how Integrated
Gradients scored 86x chance while failing the randomisation check. It now asks
*"the model says NOT feasible; does the explanation point at the inferior
alveolar canal?"* The canal is a dark tube inside bone occupying 0.05-0.65% of a
patch, so an edge detector cannot find it by accident.

## Quick start

```bash
export TOOTHFAIRY3_ROOT=/path/to/ToothFairy3     # never hard-coded

# 1. sites -> labels  (CPU, ~20 min, reads masks only)
python scripts/build_implant_labels.py --config configs/sites.yaml

# 2. scans -> native-resolution cache  (~50 GB, rented box)
python scripts/build_site_cache.py --config configs/sites.yaml

# 3. sanity gate before any real run (2 min, CPU)
python scripts/train.py --config configs/synthetic.yaml --synthetic

# 4. train, one fold at a time
python scripts/train.py --config configs/sites.yaml --fold 0 --folds 5
```

Run the synthetic gate first. It plants a bright blob at a known site per label
— a task the model must be able to learn — so a failure there means the code is
broken rather than the problem being hard.

## Why patches, not whole heads

|  | detail | field of view | samples |
|---|---|---|---|
| 128³ @ 1.0 mm | 3.3× blurred | whole head | 532 |
| 256³ @ 0.5 mm | 1.7× blurred | whole head | 532 |
| **96³ @ 0.3 mm** | **native** | one tooth site | **6,787** |

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
scan, and nerve clearance limits 408 of the 478 infeasible sites — which is both
the real clinical danger and a target an edge detector cannot fake, because the
canal is a dark tube inside bone rather than a bright edge.

## Versions

**v3.0.0 is current.** Clone the tag, not just `main`, if you want a state that
matches this README exactly:

```bash
git clone --branch v3.0.0 https://github.com/AGB2210/Adaptive-Multi-XAI-3D-ViT-for-Dental-Implant-Diagnosis-Support.git
```

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
src/models/    vit3d (from scratch), cnn3d baseline (written, never trained)
src/train/     training loop, metrics with bootstrap CIs
src/xai/       rollout, IG, GradientSHAP, Grad-CAM, LIME (ablation only),
               faithfulness, localisation, calibration, adaptive fusion
scripts/       build_implant_labels, build_site_cache, train, evaluate,
               run_xai, run_faithfulness, run_localization, run_adaptive,
               pool_cv, make_figures
```

### The superseded detection pipeline

`scripts/build_labels.py`, `scripts/build_cache.py`, `src/data/dataset.py` and
`configs/default.yaml`'s `task.labels` implement the earlier task — *is an
implant, crown or bridge already present?* It is kept because it still runs, the XAI
scripts still support it through the same `CaseSet`, and the randomisation
findings below were measured on it. It is not the project's question. `configs/preprocess_256.yaml`
belongs to the same path: a whole-head alternative, deliberately set aside.

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

A corrected model-randomisation check (Adebayo et al. 2018), destroying 100% of
9.15M parameters across 12 stages:

| method | Spearman vs intact map, fully randomised |
|---|---|
| **gradcam** | **0.28** — the only method that passes |
| attention_rollout | 0.62 |
| gradient_shap | 0.71 |
| **integrated_gradients** | **0.94** — essentially unchanged |

Integrated Gradients produces almost the same map from a trained network and a
random one. It is an edge detector on this data. This is a property of the
method, not of the task, and it holds regardless of which labels are used.

## Known limitation

Site positions are fitted to the ground-truth masks. That is legitimate for
training and for measuring this model, and it is **not a deployable pipeline** —
a new patient arrives with an image and no segmentation. A fielded system needs
a site-detection step that does not exist here.

## CI

Four gates, cheapest first: version → imports → ruff → pytest.
