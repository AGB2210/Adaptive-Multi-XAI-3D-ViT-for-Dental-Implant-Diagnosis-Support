# Adaptive Multi-XAI 3D ViT for Dental Implant Diagnosis Support

A 3D Vision Transformer written from scratch in PyTorch, and an adaptive
multi-method explainability layer that **measures whether its own explanations
are truthful** rather than assuming they are.

The explainability framework is the contribution. The classification task is the
vehicle it is demonstrated on.

## Status

Rebuilt on ToothFairy3, which ships expert voxel-level masks including
`Implant`, `Crown` and `Bridge` — the ground truth the previous cohorts lacked.

| Stage | State |
|---|---|
| Model, XAI stack, training loop, metrics | **Ported and green** — 156 tests |
| Dataset layer (config-driven) | **Done** |
| Mask → label builder | **Written and tested** |
| ToothFairy3 download | **Pending** |
| `task.class_indices` | **Empty on purpose** — read them off the dataset |
| Training / XAI runs | **Not started** |

The one thing that cannot be written ahead of the download is the mapping from
label name to mask class index. It is left empty rather than guessed: an index
wrong by one trains the model against a different anatomical structure, and every
metric downstream still looks entirely reasonable.

## Getting started once ToothFairy3 is in place

```bash
export TOOTHFAIRY3_ROOT=/path/to/ToothFairy3     # never hard-coded

# 0. fill task.labels and task.class_indices in configs/default.yaml
python scripts/build_labels.py --dataset toothfairy3   # masks -> labels CSV
python scripts/build_cache.py  --dataset toothfairy3   # volumes -> .npy cache
python scripts/train.py --config configs/synthetic.yaml --synthetic   # gate
python scripts/train.py --config configs/default.yaml
python scripts/run_xai.py --checkpoint artifacts/runs/vit3d/best.pt
```

Run the synthetic gate before any real training. It plants a bright blob at a
known site per label — a task the model must be able to learn — so a failure
there means the code is broken rather than the problem being hard. Two minutes
on CPU against an hour on real data.

## From scratch, deliberately

`src/models/vit3d.py` imports only `torch`, `torch.nn`, `torch.nn.functional`.
No `timm`, no MONAI, no `torchvision`, no pretrained weights. This is a project
requirement, not a gap to be filled.

## Layout

```
src/data/      preprocess, cache dataset, augmentation, splits, task definition
src/models/    vit3d (from scratch), cnn3d baseline
src/train/     training loop, metrics with bootstrap CIs
src/xai/       rollout, IG, GradientSHAP, Grad-CAM, LIME, faithfulness,
               calibration, adaptive fusion
scripts/       build_cache, train, evaluate, run_xai, run_faithfulness,
               run_adaptive, make_figures
```

## Datasets are configuration, never code

```yaml
data:
  datasets:
    toothfairy3:
      root: ToothFairy3
      volume: "{pid}/volume.nii.gz"
      labels: "{pid}/labels.nii.gz"
  primary: toothfairy3
```

Roots are overridden per machine by `<NAME>_ROOT` env vars (`TOOTHFAIRY3_ROOT`),
which always win over the file. No path in this repo points at anyone's disk.

## Things that look like improvements and are not

- **Do not import a pretrained backbone.** From-scratch is the spec.
- **Do not delete the `store_attention` path** in `Attention` — the fused kernel
  never materialises the attention matrix, and Attention Rollout needs it.
- **Do not set `WEIGHT_METRIC == EVAL_METRIC`.** Fitting the fusion on the metric
  you then report is circular; `fuse()` raises to stop it.
- **Do not lower Integrated Gradients below 256 steps.** Measured completeness
  error: 32 steps → 48%, 256 → 2.0%. If memory is tight, cut the batch size —
  that costs time, not correctness.
- **Do not use a zero baseline for IG** on z-scored volumes: zero is a real
  tissue value, not absence.
- **Do not change `fit_mode` to `resize`** — it makes a millimetre mean different
  things in different cohorts.
- **Do not print test metrics** outside an explicit `--test` run.
- **Do not compare a loss to a floor computed for a different label set.**

## CI

Four gates, cheapest first: version → imports → ruff → pytest.
