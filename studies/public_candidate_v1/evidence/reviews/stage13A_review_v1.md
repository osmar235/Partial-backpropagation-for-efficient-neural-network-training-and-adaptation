# Stage13A completed review — 2026-10-23

The implementation/integrity audit passes. This fixed local-loss configuration
does not qualify for an efficacy pilot: the usefulness test is inconclusive
because the full-backpropagation reference fails its positive-improvement check,
and measured resource savings also miss both declared thresholds.

Do not interpret this as proof that local learning cannot work. The diagnostic
tested single zero-moment Adam steps at two seen head-warmup states, not a
training trajectory. No failed reference is relabeled as a valid learning test.

## Resource measurements

| Source seed | Full update | Local update | Time saving | Incremental memory saving | Absolute peak saving |
|---|---:|---:|---:|---:|---:|
| 1601 | 31.380 ms | 30.624 ms | 2.41% | 6.38% | 3.09% |
| 1602 | 33.000 ms | 29.913 ms | 9.36% | 6.26% | 3.03% |

The declared resource filter required at least10% lower mean update time OR
at least20% lower mean incremental allocated peak memory in each source state.
Neither state passes. Do not round9.36% upward to a10% pass or change the metric
to absolute memory after the fact. Absolute peaks are reported for context.

This measures device-ready update operations. State restoration, input
transforms, scoring and gradient diagnostics are outside the timer. Allocated
memory is not allocator-reserved VRAM. Blocks execute sequentially on one GPU;
this is not a test of asynchronous or multi-GPU local learning.

Auxiliary-head setup/calibration/checkpoint overhead was1.250 /0.980seconds.
The frozen optimistic time-only break-even counts are1,654 /318updates. They
assume stable per-step savings and ignore recovery/quality; they do not imply
useful learning or predict an end-to-end training speedup.

## Usefulness check: reference failed

Mean score-loss improvement is loss before minus loss after, so negative means
the update made the separate scoring batch worse:

| Mode | Seed1601 mean improvement | Seed1602 mean improvement | Positive measured updates |
|---|---:|---:|---:|
| Full final-loss backprop | -1.086947 | -1.369579 | 0 /24 |
| Connected auxiliary losses | -1.118643 | -1.450361 | 0 /24 |
| Detached local losses | -0.708084 | -0.893497 | 0 /24 |
| Suffix only | -0.371394 | -0.499430 | 0 /24 |

All96measured updates worsened their scoring loss, including every full-reference
update. The protocol explicitly required positive full-reference mean improvement
and declared failure of that reference uninformative for the utility screen.
We therefore cannot conclude that local learning failed to match a useful full
step, or that its smaller deterioration demonstrates successful learning.

Saved first-trial states reproduce these values directly. For example, source
1601's score loss starts at0.467498 and ends at1.360250 after full,1.475384 after
full_aux,1.231756 after local, and0.701971 after suffix. Thus the sign is not a
summary-arithmetic error. No optimizer updates were performed during rescoring.

The diagnostic deliberately resets ALL moments, including the trained final
head's moments, and uses only one fixed-size step. Source training used a
different optimizer history and many sequential steps. Reset behavior, step
size, feature disturbance and single-step noise are possible explanations;
this experiment does not identify a cause. Do not retrospectively lower LR,
lengthen the probe or change the source state to obtain a favorable result.

## Gradient measurements

Mean cosine with full final-loss gradients:

| Local block | Seed1601 | Seed1602 |
|---|---:|---:|
| Stem + layer1 + layer2 | 0.0365 | 0.0693 |
| layer3 | 0.1334 | 0.1052 |
| layer4 + final classifier | 1.0000 | 1.0000 |

Early local gradients have weak alignment with the final-loss gradients in this
configuration. Final-block agreement is an implementation invariant, not evidence
that the full local procedure is effective. Raw gradient cosine is descriptive;
it is not the Adam update direction, a predeclared acceptance threshold, or a
convergence result. The local heads solve different immediate objectives.

## Verification and preservation

- Re-ran the original frozen auditor:34scientific dependencies, two source
  checkpoints,96measured operations,256head-calibration steps and24warmup updates.
- Calibration/probe/score index streams reconstruct exactly and training pools
  are disjoint. Source backbones and buffers remain unchanged. Runtime checks
  cover every measured update; all eight saved first-trial update states and
  both calibrated states pass independent state/counter checks.
- Independently recomputed means, improvement signs/fractions, timing and memory
  ratios, gradient dot/norm/cosine arithmetic, calibration break-even and flags.
- Reloaded both calibrated states and all eight saved updated states on CUDA;
  rescored the exact first-trial training-only score batches. Before/after losses
  reproduce within the frozen audit tolerance. No parameter update, validation
  scoring or test scoring occurred during review.
- Verified1,138immutable files before/after review, including prior experiments,
  masters, source/pretrained checkpoints, preparation records and raw diagnostic
  outputs. New review files are separate; only current pointers are updated.

The score pools are disjoint from this diagnostic's calibration/update pools,
but come from the source training set. They are not an unseen generalization
holdout. Seen checkpoints, overlapping batches across trials, fixed BatchNorm,
96px resolution and zero-moment one-step operations limit interpretation.

## Decision

Park this local-loss implementation without a long training pilot or tuning
grid. Retain two distinct findings: modest resource gains below the threshold,
and an uninformative one-step usefulness test because its reference fails.
Neither establishes that all local-objective approaches are unsuitable.

The next queued idea is later feature/token compression. A future revisit of
local learning would need a new prospective diagnostic that first qualifies its
reference update or short trajectory; do not alter this completed experiment.
No new run command or real training is prepared in this review. Previous growth
and transfer tradeoffs remain preserved; publication stays on hold.
