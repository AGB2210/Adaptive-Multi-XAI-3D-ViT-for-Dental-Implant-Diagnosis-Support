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
| Site labels built | **Done** — 6,990 mandibular sites, 503 patients |
| Native-resolution cache, patch dataset, training wiring | **Done** |
| 292 tests | **Green** |
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
| **96³ @ 0.3 mm** | **native** | one tooth site | **~7,000** |

The patch is sharper *and* 19× smaller than the 256³ volume. The structure the
model must respect is the inferior alveolar canal, 2–3 mm across: at 1.0 mm that
is two or three voxels, and no attribution method can point at something it
cannot resolve. At patch size 8 on native data, one token covers 2.4 mm.

## Mandible only

Measured over all 522 usable scans, of the sites that need an implant:

```
mandible    637 needed,  585 measurable   91.8%
maxilla    1682 needed,   72 measurable    4.3%
```

98% of the unmeasurable maxillary sites have no bone voxels at all — after an
upper tooth is lost the ridge resorbs, the sinus pneumatises, and ToothFairy3's
`UpperJaw` mask does not cover the remnant. Training on them would reproduce an
annotation gap as a clinical verdict.

Nothing important is lost: the inferior alveolar canal is annotated in **every**
scan, and nerve clearance limits 289 of the 361 infeasible sites — which is both
the real clinical danger and a target an edge detector cannot fake, because the
canal is a dark tube inside bone rather than a bright edge.

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
