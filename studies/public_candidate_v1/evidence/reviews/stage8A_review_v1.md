# Stage 8A v1 development review

Four completed baseline runs pass source/configuration, step/epoch trace, LR schedule,
checkpoint SHA-256 and AdamW-counter checks: 600 epochs / 105,600 updates.
The readiness decision is correctly FAIL. No calibration, real freeze or evaluation outputs exist.

| LR | Seed | Final validation accuracy | Training seconds |
|---:|---:|---:|---:|
| 0.0003 | 1001 | 82.06% | 759.3 |
| 0.0003 | 1002 | 81.12% | 744.9 |
| 0.001 | 1001 | 82.92% | 750.0 |
| 0.001 | 1002 | 83.46% | 746.4 |

The selected LR 0.001 averages 83.19%, minimum 82.92%. Required: mean >=85%,
minimum >=83%. LR 0.0003 averages 81.59%, minimum 81.12%. Final epochs remain
the selection endpoint; diagnostic best epochs do not replace them.

This is an underperforming development recipe, not evidence about ViT LGPB performance.
Learning curves do not establish the cause. Smoothed/augmented training loss and
unsmoothed validation loss are not directly comparable. Retain all four runs.

Prospective v2: one fixed LR 0.001, 300 epochs and MixUp alpha 0.2, same split/seeds/model.
Readiness thresholds stay unchanged. This is a bundled development amendment,
not a causal ablation or independent confirmation; reuse of validation is disclosed.
If v2 fails readiness, stop again; no automatic search and no evaluation.

Preservation manifest: preserved_v1_sha256.csv. Raw v1 results and code are unmodified.
