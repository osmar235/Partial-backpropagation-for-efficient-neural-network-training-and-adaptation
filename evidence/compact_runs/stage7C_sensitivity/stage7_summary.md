# Stage 7 reviewer-experiment descriptive summary

**These comparisons are NEW post-confirmation analyses. They do not modify the original Stage 6B inference.**

Reference: `lgpb`; study folder: `results\stage7C_sensitivity`.

Samples with missing variants are excluded pairwise; inspect n_paired_seeds.

| Variant | Reference | Paired n | Accuracy delta (pp) | Geometric wall ratio | Partial q delta (pp) |
|---|---|---:|---:|---:|---:|
| no_forced_full | lgpb | 6 | -0.517 | 0.944 | +19.78 |
| prefix_stem | lgpb | 6 | +0.105 | 1.096 | -0.15 |
| prefix_stem_layer12 | lgpb | 6 | -0.823 | 0.950 | +0.33 |
| rho_090 | lgpb | 6 | -0.268 | 0.975 | -0.60 |
| rho_099 | lgpb | 6 | -0.095 | 0.984 | +3.67 |

The CSV also contains exploratory paired 95% confidence intervals. Interpret intervals
as descriptive unless a separate Stage 7 analysis plan was frozen before running.
