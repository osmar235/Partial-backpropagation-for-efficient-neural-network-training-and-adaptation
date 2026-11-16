# AGRLS Stages 5A6–5A10 — current-batch subspace combination

## Scope

These stages used CIFAR-10 development seed 0 only. No untouched seed was used.
The purpose was to test whether current-batch forward information could combine
four historical exact-gradient-derived Adam displacements more effectively than
selecting one stale displacement.

## Stage 5A6: off-policy subspace-combination diagnostic

The live trajectory remained ordinary Adam. On each eligible batch:

1. The previous four Adam displacements formed a local subspace.
2. A one-sided probe estimated current-batch loss slopes along each direction.
3. The displacement Gram matrix and a ridge of `1e-4` defined a metric-aware
   steepest direction in that subspace.
4. Candidate parameter norm was fixed at 0.125 times the median norm of the four
   historical displacements.
5. A disjoint probe subset gated the candidate.
6. A full-batch forward supplied diagnostic truth only.

| fit/confirmation probe | true descent | gate acceptance | precision | recall | break-even | modeled saving |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 79.84% | 66.73% | 89.43% | 74.75% | 51.86% | 14.88% |
| 32 | 88.71% | 78.43% | 96.92% | 85.68% | 56.46% | 21.97% |

This was a strong diagnostic result and justified a live test with probe 32.

## Stage 5A7: live controller at candidate scale 0.125

The live controller allowed at most one skipped backward. After every accepted
subspace update, the next update was forced to obtain a fresh full-batch
gradient.

| metric | Adam | subspace controller |
|---|---:|---:|
| accepted updates | 500 | 500 |
| evaluation accuracy | 60.79% | 53.94% |
| backward passes | 500 | 286 |
| calibrated NN compute | 21.031 s | 16.603 s |
| direct measured NN time | 21.310 s | 17.464 s |
| wall time | 26.784 s | 38.714 s |
| reuse acceptance | — | 75.62% |

Paired effects:

- accuracy delta: **-6.85 percentage points**;
- backward reduction: **42.80%**;
- calibrated NN-compute reduction: **21.05%**;
- direct NN-time reduction: **18.05%**;
- wall-time reduction: **-44.54%** (controller slower).

The fixed-update criterion failed on accuracy. The wall-time penalty came from
the unoptimized full-parameter Gram and linear-combination bookkeeping.

## Stage 5A8: equal-calibrated-compute development

Because Stage 5A7 used only 78.95% of Adam's calibrated neural-network budget,
the observed ratio implied approximately 633 controller updates per 500 Adam
updates. The horizon was frozen at 630 before the run.

| metric | Adam | subspace controller |
|---|---:|---:|
| accepted updates | 500 | 630 |
| evaluation accuracy | 60.79% | 57.29% |
| backward passes | 500 | 357 |
| calibrated NN compute | 21.161 s | 21.613 s |
| wall time | 26.918 s | 48.013 s |

Paired effects:

- accuracy delta: **-3.50 percentage points**;
- backward reduction: **28.60%**;
- calibrated budget use: **102.14%**;
- direct NN-time budget use: **102.48%**;
- wall-time budget use: **178.37%**.

Additional small subspace updates narrowed but did not close the accuracy gap.

## Stage 5A9: frozen candidate-scale selection

The development scale rule was declared before the run:

> Select the largest candidate scale with confirmation precision >=90%, true
> full-batch descent >=75%, and acceptance above calibrated break-even.

All scales shared one ordinary Adam trajectory.

| candidate scale | true descent | gate acceptance | precision | recall | break-even | median accepted full-batch gain | passes |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.125 | 89.92% | 78.23% | 96.65% | 84.08% | 57.76% | 0.015077 | yes |
| 0.25 | 85.89% | 73.59% | 96.16% | 82.39% | 57.76% | 0.026396 | yes |
| 0.5 | 72.18% | 63.91% | 88.33% | 78.21% | 57.76% | 0.035631 | no |
| 1.0 | 40.73% | 38.91% | 70.98% | 67.82% | 57.76% | 0.020403 | no |

The frozen rule selected **0.25**.

## Stage 5A10: live selected-scale controller

| metric | Adam | scale-0.25 controller |
|---|---:|---:|
| accepted updates | 500 | 500 |
| evaluation accuracy | 60.79% | 52.74% |
| backward passes | 500 | 291 |
| calibrated NN compute | 21.732 s | 20.068 s |
| direct measured NN time | 21.308 s | 17.289 s |
| wall time | 27.059 s | 37.018 s |
| reuse acceptance | — | 72.8% |

Paired effects:

- accuracy delta: **-8.05 percentage points**;
- backward reduction: **41.80%**;
- calibrated NN-compute reduction: **7.66%**;
- direct NN-time reduction: **18.86%**;
- wall-time reduction: **-36.80%**.

The selected larger scale increased diagnostic full-batch gain but worsened live
accuracy.

## Interpretation

The main finding is the gap between an off-policy diagnostic and a live
controller. Along an ordinary Adam trajectory, current-batch subspace candidates
are frequently true descent steps and a disjoint probe predicts them accurately.
Once those candidates replace exact-gradient updates, however:

- the parameter trajectory changes;
- Adam moments receive fewer exact-gradient refreshes;
- the direction bank is generated from a different optimization history;
- individually descending small steps do not reproduce the learning progress of
  the skipped Adam updates;
- diagnostic precision measured on the full-Adam trajectory no longer predicts
  final live accuracy.

This is more informative than another probe-size result: future mechanism tests
must obtain evidence on the trajectory induced by the mechanism, rather than
relying only on candidate quality along an Adam reference trajectory.

## Decision

Pause the current four-direction, one-sided-fit subspace controller. Do not:

- move Stages 5A7 or 5A10 to untouched CIFAR-10 seeds;
- run 1,500-update versions;
- tune more scales, probe sizes, ridges, or bank lengths on seed 0;
- claim optimizer improvement from the Stage 5A6 or Stage 5A9 diagnostics.

The branch did establish a useful mechanistic result: cheap current-batch probes
can construct descent directions in a historical exact-gradient subspace. The
remaining problem is maintaining learning progress and optimizer state when
those directions replace backward passes.

A future branch should change that state-transition mechanism. Plausible options
include updating optimizer state from a reconstructed subspace gradient,
performing partial/blockwise backward refreshes, or defining a controller whose
development evidence is measured on-policy from the beginning. Each option
needs a cost model and a fresh development protocol before more training runs.

The projected-gradient state-transition branch is specified in
`README_STAGE5B_PROJECTED_ADAM.md`.

## Reproduce

```cmd
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
run_stage5a6_cifar10_subspace_combination_seed0.cmd
run_stage5a7_cifar10_subspace_controller_seed0.cmd
run_stage5a8_cifar10_subspace_equal_compute_seed0.cmd
run_stage5a9_cifar10_subspace_scale_diagnostic_seed0.cmd
```

Stage 5A10 used the Stage 5A7 script with `--candidate-scale 0.25` and output
folder `stage5a10_cifar10_subspace_scale025_controller_seed0`.
