# BF16 matched adaptation review — 2026-10-27

Both BF16 policies pass the predeclared practical quality-and-complete-cost
screen. BF16/automatic saves 43.25% and 42.78% of complete adaptation time in the
two tested source conditions, with final quality inside the 0.30-percentage-point
margin in both evaluation modes and every session. BF16/math saves about 18.3%.
Use BF16/automatic as the new strongest qualified baseline for this local setup.
The gain is attributable to the tested conventional precision/backend recipe;
it is not evidence of an incremental gain from the original gradient method.

## Integrity and scope

Six unfragmented sessions, 36 five-epoch trajectories, 31680 training updates
and 108 discarded warmup updates. Two archived full ViT checkpoints, 45000 CIFAR10
training examples and the fixed 5000-example validation split. No test-set use.
All parameters remain trainable FP32, with fresh AdamW state shared across arms,
matched data/augmentation/MixUp/LR/dropout-seed schedules, reference attention
and explicit FP32 cross-entropy outside autocast. Three policies are compared:
reference FP32/math, BF16/math and BF16/automatic. Repeated sessions reproduce
the same two source-specific trajectories; they are not extra independent seeds.

The frozen auditor rechecked sources, manifests, all streams and optimizer
states, and re-evaluated all final checkpoints in common FP32 and own-policy
arithmetic. All 72 inference rescores are bitwise exact. Learning results within
each source/policy are exact across all six timing sessions, including automatic
BF16 in this run. A separate CPU implementation rescored all 252 saved validation
arrays with log-add-exp, reconstructed all 24 paired timing/quality comparisons,
checked component sums and independently reproduced both decisions. No training,
backward pass or optimizer step was rerun during review. Inference audit is
excluded from original measured adaptation time. Both audits pass.

## Quality and complete adaptation cost

The FP32 reference finishes at 84.70% / 83.88% validation accuracy for source
1501 / 1502, matching the historical qualified full-adaptation accuracies and
exceeding both source accuracies (82.46% / 82.74%). All baseline readiness checks
pass. Candidate accuracies below are constant across the six repeated sessions.

| Execution | Source | Complete-time saving, median (range) | Faster pairs | FP32-scored accuracy | Own-policy accuracy |
|---|---:|---:|---:|---:|---:|
| BF16 / math | 1501 | 18.26% (11.60–20.39%) | 6/6 | 84.60% | 84.62% |
| BF16 / math | 1502 | 18.31% (14.62–19.01%) | 6/6 | 83.92% | 83.92% |
| BF16 / automatic | 1501 | 43.25% (40.22–45.83%) | 6/6 | 84.58% | 84.68% |
| BF16 / automatic | 1502 | 42.78% (41.26–43.63%) | 6/6 | 83.94% | 83.88% |

The worst common-FP32 accuracy deficits are 0.10 points for BF16/math and
0.12 points for BF16/automatic, both at source 1501. Under the own-policy final
evaluation, the worst deficits are 0.08 and 0.02 points respectively. Source 1502
has small gains or equal accuracy. All fall inside the predeclared 0.30-point
screen in every session; this is not a statistical population noninferiority
result. No best epoch or favorable session subset was selected.

Validation cross-entropy changes are also retained. At source 1501 common-FP32
loss rises by 0.002106 (BF16/math) and 0.002091 (BF16/automatic); at source 1502
it falls by 0.002386 and 0.002446. Own-policy loss changes have the same directions.
The frozen quality decision is based on accuracy, not a post hoc loss threshold.

All 12 paired sessions are faster for each policy. Each six-session/source
one-sided sign p is 0.015625. The frozen both-source test takes their maximum,
then adjusts for two candidate policies: p=0.03125 for each. Both pass. These
conditional direction tests assume independent session effects; correlated
shared-machine drift remains a limitation. The observed positive direction
holds when the candidate runs before or after the reference. Ranges are
descriptive, not confidence intervals or evidence of additional seed diversity.

Median complete FP32 times are 28.77 / 29.54 seconds per five-epoch trajectory;
BF16/automatic medians are 16.60 / 16.75 seconds. Reported percentage savings
are medians of paired percentages, not percentages derived from these separately
computed median times. Training-only paired savings are 46.63% / 46.25% for
automatic and 19.18% / 19.97% for math. Training contributes most of the absolute
gain. The primary clock additionally charges data loading, model setup, profiled
warmup/restoration, augmentation, all epoch validations, final own-policy
validation and final checkpoint/logit/trace writing. Imports, preflight,
provenance checks and post-run audits are excluded.

Peak allocated memory medians are about 2140.99 MiB (FP32), 1714.39 MiB (BF16/math)
and 1531.67 MiB (automatic) for both sources. This includes the common-FP32
validation work and is secondary. Profiled warmup dispatch is math for both math
arms and efficient SDPA for automatic (72 recorded forward calls per policy).
No flash dispatch was observed in the profiled warmups. Dispatch outside those
profiled updates was not continuously measured. Hardware: RTX 5080, PyTorch
2.11.0+cu128, CUDA 12.8, Python 3.14.3. No compilation or environment installation.

## What this changes, and what it preserves

This establishes a qualified faster full-gradient adaptation baseline in the
tested configuration. The result concerns conventional mixed precision and
backend selection with explicit FP32 loss; it does not establish a new gradient
approximation, generality across architectures/hardware, or long-run quality.
Neither two existing checkpoints nor repeated timings establish publication
novelty. Any proposed method now needs an incremental benefit over this baseline.

The earlier BF16 parameter-distance gate stays failed. This prospective study
uses a different question, longer trajectories and explicit FP32 loss, whereas
earlier gradients included their original autocast loss arithmetic. The success
therefore does not prove the older failed trajectories were harmless or justify
waiving those gates. Key-bias functional findings remain unchanged. The original
scoped 6.34–6.40% FP32 QKV result and failed 10% screen remain intact. Do not add
that percentage to the present 43% saving: their implementations and comparators
differ, and the present reference already uses the established attention layout.

The 43% applies to marginal five-epoch adaptation, not training from scratch.
If archived source-training seconds are simply added to the measured adaptation
costs, descriptive savings become about 2.26% / 2.33% for automatic and
0.92% / 0.98% for math. This is accounting context using historical source clocks,
not a newly measured complete pretraining-plus-adaptation workflow.

## Decision and proposed next step

Close this prospective screen as passed for both policies. Prefer BF16/automatic
as the strongest qualified local baseline. Keep the report, raw checkpoints,
frozen design and reproducible analysis together in the paper-support record.

Recommended next is a bounded cost-feasibility check of remaining prefix/suffix
backward savings under this optimized baseline, before another quality-training
campaign. Compare full backward with predeclared suffix-only boundaries, charging
any routing/refresh overhead. Such a check measures available compute savings
only; it cannot establish the quality of skipping gradients. If little useful
headroom remains, avoid rebuilding a method whose costs cannot compete. If
headroom remains, select one mechanism and freeze its quality/cost study. This
is a proposed next test, not a prepared or launched experiment. No new run
command is issued. Publication remains on hold pending a distinct contribution.

## Evidence and organization

Raw results: results/bf16_continuation_v1, unchanged.
Audit/analysis: frozen_audit_recheck.json, independent_audit.json, review_data.json,
group_summary.json, validation_scores.json and source_metadata.json.
Organized results and portable aggregation: paper_support/bf16_continuation_results_v1.
Frozen source capsule: paper_support/bf16_continuation_addendum_v1, unchanged.
Current master v7.45; paper discussion memory v34. All previous science retained.
