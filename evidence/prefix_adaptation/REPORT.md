# Prefix adaptation review — 2026-10-28

Close the tested cut3/r4 adaptation recipe as FAILED under its predeclared joint
quality/complete-cost criteria. Its complete-time savings are consistent, but
source1502 misses the 0.30-percentage-point quality margin in BOTH scoring modes.
Frozen cut3 fails quality on both sources. No threshold change, best-epoch
substitution, additional epochs or automatic boundary/refresh rescue sweep.
Publication remains on hold; qualified full BF16/automatic remains the baseline.

This is an informative speed–accuracy tradeoff, not an implementation failure:
periodic refresh improves on frozen-prefix accuracy, but does not preserve full
accuracy within the promised margin across both checkpoints. It does not disprove
every selective-gradient method or establish novelty for periodic scheduling.

## Integrity and baseline readiness

The frozen auditor passes all36 trajectories/31680 training updates/108 discarded
warmups, with six counterbalanced sessions, two original trained sources and no
fragmented sessions. It reconstructs data/augmentation/MixUp/LR/dropout streams,
global refresh cadence, initial states, final weights and all AdamW ages. Prefix
ages are880 for full, absent for frozen,220 for periodic; suffix age880 throughout.
Frozen prefix weights are bitwise unchanged; final moments finite; inactive
gradient absence is checked at execution. Both final evaluation modes were rerun
on all36 saved checkpoints: all72 rescores are bitwise exact.

Independent CPU analysis recomputed all252 saved-logit loss/accuracy arrays using
log-add-exp, verified31680 step records, reconstructed24 paired comparisons,
complete-time component sums, all readiness/quality decisions and both conditional
timing tests. Both practical-screen verdicts reproduce as false. No training,
backward pass or optimizer update occurred during review. Frozen-auditor inference
is disclosed and excluded from the original measured adaptation time.

The full reference reproduces the previously qualified BF16/automatic final
weights and optimizer state exactly in all12 source/session runs. Common-FP32
accuracy is84.58%/83.94%; own-policy accuracy84.68%/83.88%, for1501/1502. Both source
readiness checks pass. This failure cannot be attributed to the old baseline
readiness problem. All schedules' recorded learning repeats exactly across the
six timing sessions. Those repeats are not six additional independent quality seeds.

## Prospective results

| Schedule | Source | Complete-time saving, median (range) | Common-FP32 accuracy | Own-policy accuracy | Deficit common / own | Quality |
|---|---:|---:|---:|---:|---:|---|
| Frozen cut3 | 1501 | 26.27% (25.15–31.93%) | 83.86% | 83.78% | 0.72 / 0.90 points | FAIL |
| Frozen cut3 | 1502 | 26.95% (25.03–29.09%) | 83.18% | 83.18% | 0.76 / 0.70 points | FAIL |
| Periodic cut3/r4 | 1501 | 20.84% (15.75–25.56%) | 84.42% | 84.46% | 0.16 / 0.22 points | PASS |
| Periodic cut3/r4 | 1502 | 19.69% (18.16–22.90%) | 83.50% | 83.52% | 0.44 / 0.36 points | FAIL |

Deficits compare each candidate with paired full in the SAME evaluation mode.
The allowed deficit is0.30 points in both modes, every source/session. Periodic
1501 passes (0.16/0.22); periodic1502 fails (0.44/0.36). Frozen1501 fails (0.72/0.90),
as does frozen1502 (0.76/0.70). Final scores are constant across repeated sessions;
this is a two-checkpoint screen, not a population noninferiority result. Do not
reinterpret the 0.44-point deficit as acceptable by widening the margin after
seeing it. A different acceptable tradeoff would be a separately defined question.

Both partial schedules are faster in all12 paired sessions. Per-source timing
sign p=.015625; the predeclared maximum-source then two-policy adjustment gives
p=.03125 for each. Thus both pass the conditional direction screen but fail the
joint scientific criterion. Shared-machine dependence limits timing inference.
Reference-first and candidate-first subsets retain positive median savings;
none of the actual paired complete-time differences reverses sign.

Median complete times are16.57/16.53s for full;12.13/12.04s for frozen;13.20/13.34s
for periodic. Medians of paired saving percentages are not ratios of separate
median times. Training-only savings are29.96%/29.66% frozen and22.57%/22.99% periodic.
Unlike the preceding cost screen, the primary clock here includes data/model
setup, initial validation, profiled warmup/restoration, training/augmentation/
routing, all epoch/final validations and final checkpoint/logit/stream/CSV writing.
Artifact-size differences from absent frozen-prefix moments are charged. Imports,
preflight/provenance checks, post-trajectory telemetry/state checks and final
checkpoint audit inference are excluded. No test-set evaluation or extra training.

## What periodic refresh contributes

Against frozen cut3, periodic refresh improves common-FP32 accuracy by0.56 points
for1501 and0.32 for1502; own-policy gains are0.68/0.34 points. Its median complete
time is8.83%/8.96% longer than freezing. These are descriptive matched outcomes,
not superiority inference based on repeated identical seeds. They support a
learning benefit from refreshing here, but not enough to meet the quality screen
on source1502. The frozen rule for refresh being needed to pass this screen is
not met because periodic itself does not qualify on both checkpoints.

Validation cross-entropy also remains worse than full. Common-FP32 loss increases
are0.012649/0.022198 frozen and0.003756/0.004971 periodic. Own-policy increases are
0.012789/0.022396 frozen and0.003806/0.005062 periodic. Report these differences;
loss was not substituted for the predeclared final-accuracy criterion. All epoch
scores are retained; no best-epoch rescue is used.

## Interpretation and decision

The cost-screen premise survived actual complete adaptation measurement: skipping
prefix updates saves time under optimized BF16/automatic execution. The missing
piece remains learning preservation. Both this adaptation result and earlier
Stage9E show why compute headroom alone is insufficient, while involving different
training regimes, gates and budgets. Neither result should be overwritten by the
other. Current evidence supports a scoped tradeoff and negative qualification.

Keep full BF16/automatic as the qualified practical baseline. Close this exact
cut3/r4 recipe and retain all positive speed and negative quality evidence for
the paper/discussion. Do not automatically try cut1/cut5, r2/r8, new LR/epochs or
a relaxed margin as a rescue. Any later mechanism needs a distinct rationale
and a new prospective plan; none is selected, prepared or launched here.

These results concern adaptation from two trained checkpoints on one architecture,
one GPU and reused training-derived validation. They establish neither faster
pretraining nor generality. Including archived source-training time gives only
about0.65%/0.62% descriptive source-plus-adaptation accounting savings for periodic;
that is not a new measured pretraining workflow and quality still fails. Do not
add current percentages to earlier43% BF16 or6.34–6.40% QKV results: their clocks,
comparators and quality properties differ. Earlier gates remain unchanged.

No new run command. Updated project masterv7.49/paper memoryv38 and an organized
results/portable-analysis addendum preserve the decision and evidence. No external
publication or release. Frozen scientific sources and all previous records remain
intact, with navigation-only edits tracked separately.
