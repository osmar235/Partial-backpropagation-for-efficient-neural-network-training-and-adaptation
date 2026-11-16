# Attention/copy profile review — 2026-10-28

The saved-results audit PASSES. This diagnostic does not establish a reliable,
learning-preserving speedup from removing the compatibility copy. Close this
bounded diagnostic and park copy removal as a candidate method. Keep the qualified
full-backpropagation BF16/automatic baseline. No new experiment is prepared or run.

The explicit forward compatibility copy occupies only 0.475–0.533% of linked GPU
self time in the compatible profiles (median 0.509%; 43.51–53.86 microseconds per
active update). That is useful evidence about this specific operation. It is NOT
the total cost of layout, a wall-time percentage, or a bound on downstream and
backward consequences. Plain timing changes cannot be explained solely by this
number. No equivalence or quality conclusion follows from copy accounting.

## Integrity, execution and scope

The frozen auditor was rerun against saved results, followed by independent
standard-library analysis. Neither review performs inference, training, backward
passes or optimizer updates. The frozen sources, protocol, traces and raw results
are unchanged. All 9,467 previously protected immutable files pass SHA-256 checks.

There are four distinct process sessions, no retries, 64 primary blocks with 4,096
full updates, 16 event blocks with 1,024 updates, 16 short plain blocks with 64
updates and 16 profiler blocks with 64 updates. There are 16 traces, each with one
warmup and three active updates. Common anchors add 128 updates and discarded arm
warmups add 512: 5,888 diagnostic updates in the original experiment, zero during
review. The 288 observed active attention forward calls use the efficient SDPA
backend; automatic policy does not imply flash attention.

Within each arm, event/plain and profile/short-plain losses, final model, optimizer
and RNG identities agree exactly. Counts, finite-loss records, optimizer ages,
paired savings, instrumentation overhead and linked operator totals reproduce.
Finite-state checks were performed live; saved final hashes are not full archived
checkpoints. Review therefore does not independently reconstruct final tensors.

This is the same six-block ViT, two trained sources, batches 256/200 and one fixed,
once-augmented/MixUp training batch per case. All parameters receive full gradients.
The direct-layout control uses pinned upstream attention inside the same mapped
model; it is not a separate complete upstream architecture. Original and disabled
dropout both use training mode. No held-out accuracy or test-set result is measured.

The primary clock covers RNG/zero, forward/FP32 loss/readback, backward and AdamW.
It excludes loading, fixed-input preparation, anchor/warmup/restoration and artifact
work. Results concern operation blocks, not complete adaptation or pretraining.

## Ordinary paired timing

Positive saving means the direct native-layout control is faster than compatible.
Each cell has four counterbalanced process repeats. These are descriptive repeats,
not four independent quality seeds; no post hoc significance or new 5%/10% gate.

| Source | Batch | Dropout | Median paired saving | Range | Faster pairs | Exact learning across layouts |
|---|---:|---|---:|---:|---:|---|
| 1501 | 256 | original | 5.39% | 4.46 to 21.22% | 4/4 | No |
| 1501 | 256 | disabled | 3.51% | 0.55 to 8.61% | 4/4 | Yes |
| 1501 | 200 | original | 6.51% | -45.97 to 10.59% | 3/4 | No |
| 1501 | 200 | disabled | 7.07% | -0.15 to 24.29% | 3/4 | Yes |
| 1502 | 256 | original | 2.90% | -2.97 to 22.96% | 3/4 | No |
| 1502 | 256 | disabled | -5.32% | -16.50 to 24.28% | 1/4 | Yes |
| 1502 | 200 | original | 0.65% | -24.41 to 29.63% | 2/4 | No |
| 1502 | 200 | disabled | 14.70% | 2.30 to 21.39% | 4/4 | Yes |

Overall 24/32 pairs are positive, with a range of -45.97% to +29.63%. The pooled
median is 5.28%, but mixing conditions with reversals cannot establish a general
5% speedup. In particular, source 1502/batch 256/dropout disabled is faster only
once in four comparisons and has a negative median. Other cells are promising
descriptively, but they do not justify selecting a favorable cell as a method.
Per-order subsets and every pair are retained in group_summary/review_data.

All 16 original-dropout comparisons differ in recorded learning across layouts.
Different strides can map dropout randomness to different elements even with the
same RNG state. All 16 dropout-disabled comparisons have exact recorded identities,
but their timings remain mixed. Neither finding measures final adaptation quality.
The control is not qualified as equivalent original-dropout training.

## What the traces do and do not show

The frozen collector groups operators using CPU parent chains. Autograd operators
on worker threads often lack the main-thread backward range as an ancestor. Much
of their GPU work is consequently labeled `outside`. This is an attribution gap,
not evidence that backward is almost free or that outside work is disposable.

A separate post hoc reconstruction maps raw GPU activities to CPU launch operators
by External id, then places each complete CPU operation within a unique CPU
user_annotation phase interval, allowing another CPU thread. GPU annotation ranges
are excluded from that host interval lookup. This recovers 14,304 cross-thread
activities. It does not change any frozen results or ordinary timing.

Across all 16 traces, reconstructed GPU-duration shares are 28.93–33.18% for
forward/loss, 61.81–68.73% for backward, and 1.59–3.04% for optimizer. Residual
unresolved shares range from 0 to 5.27%. There are 23,131 GPU activities, of which
355 remain unresolved. Unique CPU-linked GPU duration totals reproduce the frozen
linked self-time totals to below 0.1 microsecond per trace. Raw GPU durations not
linked to a unique exported CPU op are retained separately, not discarded.
Forward compatibility-copy totals also reconcile independently with raw traces.

These are sums of recorded activity durations in heavily instrumented runs, not
a decomposition of ordinary elapsed wall time; they do not establish removable
cost. Overlapping work, host gaps and synchronization require different accounting.
Efficient-attention backward, normalization backward and matrix operations are
prominent. Expensive required operations are not automatically optimization targets.
Broad copy/cast totals contain other work and must not be labeled compatibility copy.

Event-wall changes versus matched plain range from -25.44% to +52.34%. Negative
overhead reflects timing variability/order effects, not negative instrumentation
cost. Profiler session overhead versus four short plain updates is +278.15% to
+432.96% (about 3.78–5.33 times total wall). Do not rank implementations from
profiled or event wall time. Boundary telemetry is preserved, but it does not
identify a proven cause for each timing outlier.

An internal first reconstruction draft mistakenly included both CPU and GPU
annotation ranges, making some phase matches ambiguous. It is preserved under
draft_phase_reconstruction1 for review history only. The finalized reconstruction
uses CPU ranges exclusively. Draft phase counts are superseded; they are not
included in the public results addendum. This is a review-analysis correction,
not a changed experiment, excluded run or retry.

## Project decision and next step

Close the bounded profiling question with a scoped negative qualification: direct
forward copy cost is small, cross-layout learning differs with normal dropout,
and ordinary timings do not establish a robust general saving. Park copy removal;
do not claim all layout optimization is impossible or that no further speed exists.
No single new implementation target is justified by these traces alone.

Next, reassess the paper's distinct contribution before spending on another training
campaign. Assemble the strongest positive and negative findings around one precise
research question, distinguish conventional execution improvements from an original
gradient-method result, and decide whether that evidence supports a scoped empirical
paper or whether a new mechanism is needed. Any new mechanism requires its own
prospective rationale, comparison, quality criterion and complete-cost measurement.
No genetic/frequency hybrid or automatic cut/refresh rescue is selected here.

The full BF16/automatic baseline remains qualified locally; its earlier 43.25% /
42.78% complete adaptation savings are conventional execution results. Prior cut3/r4
quality failure, Stage9E failures, the original QKV 10% gate failure and earlier
BF16 numerical-distance failures remain in the record. This diagnostic has no new
threshold and cannot repair those outcomes. Do not add percentages across studies.
Publication remains on hold pending a defensible distinct contribution.

## Evidence and organization

Current master: v7.51. Paper discussion memory: v40. Source capsule remains
paper_support/attention_copy_profile_addendum_v1. Reviewed evidence and portable
aggregation are in paper_support/attention_copy_profile_results_v1. The portable
standard-library command reproduces group_summary.json exactly without torch/GPU.
Raw traces and source checkpoints remain at original project locations; source
provenance and preservation manifests identify them. Full experiment execution
still contains local provenance paths and needs portability work before release.

Internal audit/preservation scripts remain in protocol/attention_copy_profile;
draft review outputs remain in the internal results folder. No external publication,
scientific rerun or next run command. Final preservation and package checks are
recorded in project_update_verification.json and review_artifacts_sha256.json.
