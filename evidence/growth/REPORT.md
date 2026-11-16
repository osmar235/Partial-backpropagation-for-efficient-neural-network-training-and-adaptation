# Stage11B completed review — 2026-10-22

The fixed midpoint-growth recipe fails the predeclared both-seed screen. Growth
saves complete-run time in both seeds but meets the final accuracy margin in
only one. Retain this as mixed development evidence, not a successful method.
The integrity audit passes; that is separate from the efficacy decision.

| Seed | Full final accuracy | Growth final accuracy | Difference | Full time | Growth time | Time saving | Frozen screen |
|---|---:|---:|---:|---:|---:|---:|---|
| 1501 | 82.46% | 81.06% | -1.40 pp | 534.83 s | 408.91 s | 23.54% | Fail: quality |
| 1502 | 82.74% | 82.62% | -0.12 pp | 525.11 s | 409.64 s | 21.99% | Pass |

Both require accuracy within0.20percentage points of the best ordinary control
and at least5% less complete-run time than full and the fastest quality-qualified
control. Full is the best and only quality-qualified control in each seed. Full
and growth both meet sustained80% readiness. Thus this is an interpretable
quality failure in one seed, not a baseline-readiness failure.

## All controls

| Arm | Seed1501 accuracy | Seed1502 accuracy | Mean accuracy | Mean complete-run time |
|---|---:|---:|---:|---:|
| Full, uninterrupted | 82.46% | 82.74% | 82.60% | 529.97 s |
| Small, uninterrupted | 80.60% | 81.60% | 81.10% | 297.40 s |
| Full, Adam reset at50 | 82.16% | 82.46% | 82.31% | 527.39 s |
| Small, Adam reset at50 | 80.70% | 81.50% | 81.10% | 294.19 s |
| Growth + Adam reset at50 | 81.06% | 82.62% | 81.84% | 409.27 s |

Growth improves final accuracy over uninterrupted small by0.46 /1.02points,
and over small with the same Adam reset by0.36 /1.12points. These paired observed
contrasts support some benefit from changing capacity beyond merely resetting
Adam, but two development seeds cannot establish a general effect. Growth costs
more than remaining small, and its mean deficit to full is0.76points. Averaging
does not replace the frozen per-seed decision. Restarting full Adam reduces
final accuracy by0.30 /0.28points in these runs.

## Recovery observations, explicitly post hoc

The frozen audit verifies the exact expansion map and unchanged predictions at
the transition probe. The following declines occur AFTER one full epoch of
training, not at the mapping operation:

| Seed | Growth epoch50 accuracy | Growth epoch51 accuracy | Change |
|---|---:|---:|---:|
| 1501 | 78.08% | 76.18% | -1.90 pp |
| 1502 | 77.68% | 76.38% | -1.30 pp |

Small-reset controls instead move to79.48% /77.80% at epoch51. This makes
post-expansion optimization a plausible issue to investigate if growth is
revisited; it does not identify a unique cause. Gradient/Adam geometry, dropout
changes and finite training budget are not isolated by this pilot. We cannot
infer that an earlier transition, longer run or different optimizer would fix it.

Growth reaches sustained80% at observed350.89 /294.06seconds versus full251.09 /
278.04seconds:39.75% /5.76% slower. These are the frozen descriptive post-warmup
observed clocks, distinct from the complete-run primary clock. Growth does not
sustain82% in seed1501; seed1502 confirms82% only at epoch100. These observations
reinforce the recovery concern but do not substitute a new primary endpoint.

## Verification and limits

- Ten100-epoch runs,176,000 updates. Frozen source/evidence/configuration,
  step LR/MixUp/dropout schedules, batch sizes, midpoint map, Adam state/counters,
  final states and clocks all verified by the original read-only auditor.
- Matched groups have bitwise-identical pre-transition losses, validation
  histories, weights and moments; data RNG streams match across all arms.
- Independently recomputed thresholds, sustained crossings, censoring, control
  qualification, ratios and per-seed decisions from saved traces and summaries.
- Reloaded all ten final checkpoints and rescored the original5,000-example
  development validation split on CUDA: every accuracy reproduced exactly and
  loss matched within1e-6. No training or test-set evaluation performed.
- Verified941 immutable files before/after review, including prior science,
  masters, preparation records and all real Stage11B outputs. New review files
  live separately. Only current pointers are intentionally updated and tracked.

The primary clock includes setup/warmup, training, validation, transition,
logging/telemetry and checkpoints, excluding common dataset load and post-run
audit/sealing. Two seeds, repeatedly reused validation and strict FP32/math
attention limit generalization and optimized-deployment conclusions. No claim
of statistical significance, novelty or submission readiness follows.

## Decision

Close this fixed epoch50 recipe as failed under its frozen rule. Do not extend
epochs, relax the margin, select seed1502 or start a transition grid. This does
not rule out growth in general; mechanics and cost headroom passed, while
reliable quality recovery remains unresolved.

Continue the agreed one-idea-at-a-time sequence. The next queued direction is
pretrained adaptation with selective updates, to be specified separately before
new experiments. Any later growth revisit should first test a concrete recovery
mechanism in a small diagnostic and declare its controls/cost accounting. No new
training has been launched or new run command prepared in this review.
Publication remains on hold.
