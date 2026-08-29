# Fold 0 results — the rows, not the summaries

Requested in `reply-reporting-path.md` §"What we still need", item 1: *"The report
gives medians; the intervals need the rows, and the patient-clustered bootstrap
cannot be run without them."*

These are those rows, pulled off the rented box. 1.1 MB of CSV, committed here
because `artifacts_*/` is gitignored and these are small, text, and diffable.
**Not a substitute for re-running anything** — they are one fold's outputs, kept
so the intervals and any re-analysis can be computed without a GPU.

---

## Provenance

| | |
|---|---|
| Model | fold 0, `best.pt` at epoch 41, early stopped at 53 of 60 |
| Code | **v3.0.0 (`ab623f4`)** for training and all XAI except where noted |
| Hardware | RTX 4090 24 GB, rented |
| Training | 4,510 s, ~83 s/epoch |
| XAI | ~3 h |

**`results_ablations.csv`, `results_pareto.csv` and `calibration.json` at the top
level were re-run on v3.1.0**, because their v3.0.0 versions are void — the
temperature came back NaN, so the confidence gate escalated nothing. The v3.0.0
originals are kept in `superseded-v3.0.0/` rather than deleted, so the before and
after can be compared; they must not be quoted.

The stale-file check from `RUNBOOK.md`: `ece_before <= 1` and a finite
`temperature`. The superseded file fails both.

```
superseded-v3.0.0/calibration.json    temperature NaN     ece_before 9.097
calibration.json                      temperature 1.2566  ece_before 0.0388
```

---

## What is in each file

| file | rows | what it holds |
|---|---|---|
| `results_randomization.csv` | 1,440 | Spearman vs the intact map, per method per cascade stage, 30 cases x 12 stages |
| `results_localization.csv` | 1,940 | enrichment, pointing hit, mass inside, IoU/Dice and the nerve-vs-jawbone ratio, 485 scored cases |
| `results_faithfulness.csv` | 800 | deletion and insertion AUC per method per case, 200 cases across 88 patients |
| `results_agreement.csv` | — | inter-method agreement |
| `results_ablations.csv` | 200 | adaptive fusion: per-case weights, routing decision, fused vs uniform vs best individual |
| `results_pareto.csv` | 7 | compute against faithfulness at seven ensemble fractions |
| `calibration.json` | — | temperature, ECE before and after |
| `xai_ig_completeness.csv` | 3 | the IG axiom, one row per head |
| `xai_runtime.csv` | 5 | measured seconds per volume per method |
| `xai_sanity.csv` | — | planted-signal enrichment |
| `cv_fold0/` | — | `metrics.json`, `history.csv`, `best_val_metrics.json` |
| `cv_folds.json` | — | **the fold partition.** Folds 1-4 must use this file or the cross-validation guarantee is void |

`cv_folds.json` is the one that would be expensive to lose: regenerating it puts
a patient in training for one round and testing for another under a different
partition, which silently destroys the "scored once by a model that never saw it"
property.

---

## Two things the CSVs settle that the report only stated

### The randomisation ordering is separated, the localisation ordering is not

Patient-clustered bootstrap, 4,000 resamples, medians, computed from
`results_randomization.csv` and `results_localization.csv`:

```
final cascade stage (cls_token)          enrichment vs the canal
attention_rollout  -0.335 [-0.362,-0.284]   1.10 [1.00, 1.19]
gradient_shap      +0.336 [+0.333,+0.342]   2.03 [1.93, 2.14]
gradcam            +0.630 [+0.526,+0.707]   2.22 [2.05, 2.35]
integrated_grad    +0.740 [+0.722,+0.759]   2.61 [2.23, 3.03]
```

**No two randomisation intervals overlap**, so that ordering is supported rather
than nominal. On localisation gradcam, gradient_shap and IG **do** overlap, so no
method can be called the best localiser; rollout's lower bound is exactly 1.00,
indistinguishable from chance. Two of four pointing-rate intervals include zero.

Resampling is by **patient**, not by row: a patient contributes up to fourteen
sites sharing anatomy, field of view, scanner and annotator.

### The Pareto sweep now has more than one point

```
ensemble    cost (s)    faithfulness
    0%       0.0073      -0.1212
   10%       0.4926      -0.1257
   30%       1.4632      -0.1245
  100%       4.8603      -0.1254
```

669x the compute buys nothing measurable. Under v3.0.0 every row read
`ensemble_fraction_actual = 0.0` because `nan >= nan` is `False`.

---

## What these files cannot tell you

- **No pooled or test-set result.** One fold. `pool_cv.py` needs all five.
- **No baseline.** Neither the CNN nor threshold-and-measure had been run when
  these were produced.
- **`results_faithfulness.csv` is the blur-baseline run.** Its flat spread is
  very likely an artefact of a 1.2 mm Gaussian preserving the crest-to-canal
  geometry that `available_height_mm` regresses. Do not read a conclusion out of
  it; the `mean`-baseline re-run is what settles it.
- **No achievable-ceiling control**, so the enrichment figures have no
  denominator. "Above chance and clinically unusable" is what they support.
