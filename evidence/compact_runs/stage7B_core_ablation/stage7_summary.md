# Stage 7 reviewer-experiment descriptive summary

**These comparisons are NEW post-confirmation analyses. They do not modify the original Stage 6B inference.**

Reference: `lgpb`; study folder: `results\stage7B_core_ablation`.

Samples with missing variants are excluded pairwise; inspect n_paired_seeds.

| Variant | Reference | Paired n | Accuracy delta (pp) | Geometric wall ratio | Partial q delta (pp) |
|---|---|---:|---:|---:|---:|
| freeze_prefix | adam | 6 | -0.685 | 0.908 | +35.47 |
| lgpb | adam | 6 | -0.108 | 0.915 | +35.93 |
| periodic_matched | adam | 6 | -0.068 | 0.920 | +35.99 |
| random_matched | adam | 6 | -0.103 | 0.917 | +35.99 |
| replay_prefix | adam | 6 | +0.070 | 0.920 | +35.83 |
| adam | lgpb | 6 | +0.108 | 1.093 | -35.93 |
| freeze_prefix | lgpb | 6 | -0.577 | 0.992 | -0.46 |
| periodic_matched | lgpb | 6 | +0.040 | 1.005 | +0.06 |
| random_matched | lgpb | 6 | +0.005 | 1.002 | +0.06 |
| replay_prefix | lgpb | 6 | +0.178 | 1.006 | -0.10 |

The CSV also contains exploratory paired 95% confidence intervals. Interpret intervals
as descriptive unless a separate Stage 7 analysis plan was frozen before running.
