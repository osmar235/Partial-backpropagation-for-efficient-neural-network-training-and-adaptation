# Independent Oxford Pet evaluation — final review — 2026-10-30

The prespecified periodic-versus-full primary comparison does not meet the joint
accuracy/time criterion. The prespecified frozen-prefix secondary comparison
does: it saves approximately 35% of complete adaptation time while satisfying
the fixed 0.30 pp noninferiority margin. These conclusions use the frozen
analysis and all 24 evaluation seeds; no development outcomes were pooled.

## Final results

| Method | Test accuracy | Difference vs full (95% interval) | Mean complete time | Geometric complete-time saving (95% interval) | Prespecified joint criterion |
|---|---:|---:|---:|---:|---|
| Full BF16 | 90.1358% | Reference | 20.872 s | Reference | Reference |
| Periodic r4 (primary) | 89.5782% | -0.558 pp [-0.721, -0.394] | 15.481 s | 25.83% [25.44, 26.21] | FAIL: accuracy criterion |
| Frozen prefix (secondary) | 90.0223% | -0.114 pp [-0.262, +0.035] | 13.575 s | 34.96% [34.66, 35.25] | PASS |

Accuracy uses final official-test predictions in common FP32 with TF32 disabled.
All training used the same optimized BF16 execution. Intervals are two-sided
paired-t 95% intervals; their endpoints also provide the prespecified one-sided
97.5% bounds. Timing intervals are computed on log paired complete-time ratios
and transformed to savings. Do not interpret geometric ratios as ratios of
arithmetic mean times or pool these timings with prior experiments.

The frozen-prefix lower accuracy bound is -0.262 pp, above -0.30 pp. Its upper
log-time-ratio bound is negative. It therefore passes both components of the
prespecified secondary joint claim. This means bounded accuracy loss within the
specified margin, not identical accuracy or an accuracy improvement.

Periodic r4 has a mean deficit of 0.558 pp and an accuracy interval entirely
below -0.30 pp (-0.721 to -0.394 pp). Its timing component passes, but its joint
claim fails. This is not merely a wide interval straddling the allowed margin.
Keep the failed primary comparison explicit; do not relabel freezing as the
original primary method or tune a replacement cadence on the opened test set.

The two composite method claims each use intersection-union testing at .025,
with Bonferroni across the two claims bounding family error at .05 under the
specified assumptions. No extra confirmatory claim is made for accuracy
superiority, own-BF16 scoring, or frozen-versus-periodic superiority.

Own-BF16 scoring is consistent descriptively: frozen-minus-full -0.143 pp
[-0.292, +0.006], periodic-minus-full -0.575 pp [-0.738, -0.412]. This is a
secondary robustness analysis, not an additional multiplicity-free claim.

## Precision, timing and scope

The attained primary accuracy half-width is 0.163783 pp, meeting the original
0.20 pp precision target despite the conservative development projections.
Observed test paired SD is 0.387870 pp, versus 0.641615 pp in development. The
frozen-prefix half-width is 0.148580 pp. Retain the historical planning warning:
the target was not guaranteed at preparation; it was achieved in the final data.

Both methods saved time in all 24 paired runs. Complete-time ranges: frozen
34.04–37.09%, periodic 24.59–29.13%. Arithmetic mean paired savings: 34.96% and
25.82%, respectively. Mean training-only savings: 42.73% and 31.71%. These
results describe a single interleaved session on the NVIDIA RTX 5080; the 24
seeds are not 24 independent timing sessions, GPUs, datasets or pretrainings.
Intervals quantify paired seed variability conditional on the shared official
test sample and public backbone. Shared timing drift and different hardware
remain generalizability limitations; the prescribed balanced order was retained.

Complete adaptation time includes cache/model loading, warmup/restoration,
training, internal validation and checkpoint/log writing under the frozen clock.
It excludes independent official-test assessment, preflight/imports/audits,
pretraining and one-time dataset preparation. Official-test cache preparation
took 11.231 s. Mean separate test-assessment times were 1.884 s (full), 1.901 s
(periodic) and 1.882 s (frozen). All 72 adaptation clocks sum to 1198.273 s and
all test-assessment clocks sum to 136.027 s; neither sum is the full study's
wall-clock duration because audits and other exclusions are not included.
Do not describe these as raw-image or pretraining-inclusive speedups.

## Integrity

All 72 trajectories completed in one unfragmented launch. The review verified
the source/runtime/data freeze, complete markers, original training gate, final
analysis seal, pretrained weights, official-test annotation/image/cache hashes,
test label order and trainval/test-id disjointness. Training audits reconstructed
optimizer ages, frozen-prefix identity, BatchNorm buffers, initialization and
data/crop/flip/LR streams. Independent NumPy scoring reproduced every saved
accuracy/loss and the frozen paired analysis.

All 144 repeated development-validation checkpoint rescores and all 144 new
official-test checkpoint rescores matched saved logits exactly. No training
was rerun and every original scientific result file remained byte-identical.
The source enforces all-training-complete before official-test access; saved
gate/marker evidence agrees. These are integrity/provenance checks, not an
independent observer's guarantee about activity outside this workflow.

## Paper interpretation and next action

This independent dataset/resolution workload supports useful prefix omission
under competitive execution, while showing that periodic refresh is not always
the best policy. Simple freezing, an established strategy, is the successful
control here; do not present it as a newly invented algorithm. The result does
not establish general superiority of the original adaptive controller or of
periodic updating. It strengthens the empirical comparison of omission policy,
optimizer semantics and realized cost alongside the earlier project results.

Next: integrate this completed, locked evaluation into manuscript v11 and its
evidence map, preserving the primary/secondary distinction, the earlier CNN/ViT
results, and the known execution/precision scope. Add the protocol and compact
reproduction capsule to the eventual public supplement. No further Pet tuning
or evaluation expansion is prescribed by this review. Manuscript v10 remains
the current document until that editorial update; all original drafts are kept.

## All paired evaluation outcomes

| Seed | Frozen accuracy difference | Periodic accuracy difference | Frozen complete-time saving | Periodic complete-time saving |
|---|---:|---:|---:|---:|
| 2801 | -0.354 pp | -0.872 pp | 37.09% | 29.13% |
| 2802 | -0.136 pp | -0.164 pp | 34.18% | 25.05% |
| 2803 | +0.027 pp | -0.382 pp | 34.85% | 25.87% |
| 2804 | +0.164 pp | -0.736 pp | 34.04% | 25.96% |
| 2805 | +0.491 pp | -0.245 pp | 35.14% | 25.79% |
| 2806 | +0.218 pp | -0.763 pp | 35.88% | 26.95% |
| 2807 | -0.164 pp | -0.600 pp | 34.74% | 26.03% |
| 2808 | +0.136 pp | -0.818 pp | 35.48% | 26.08% |
| 2809 | -0.327 pp | -0.763 pp | 34.83% | 25.28% |
| 2810 | -0.273 pp | -1.008 pp | 35.56% | 26.38% |
| 2811 | -0.790 pp | -0.572 pp | 34.75% | 26.09% |
| 2812 | -0.382 pp | -0.382 pp | 34.90% | 25.05% |
| 2813 | -0.164 pp | -0.218 pp | 34.56% | 26.01% |
| 2814 | +0.354 pp | -0.300 pp | 34.76% | 25.11% |
| 2815 | -0.382 pp | -0.600 pp | 34.12% | 25.54% |
| 2816 | -0.273 pp | -0.300 pp | 35.37% | 26.08% |
| 2817 | -0.491 pp | -0.382 pp | 34.48% | 25.28% |
| 2818 | +0.055 pp | -0.109 pp | 35.57% | 24.59% |
| 2819 | -0.572 pp | -1.336 pp | 35.48% | 26.07% |
| 2820 | -0.382 pp | -0.872 pp | 34.37% | 26.07% |
| 2821 | +0.191 pp | -1.063 pp | 34.32% | 24.60% |
| 2822 | +0.709 pp | +0.518 pp | 35.19% | 25.90% |
| 2823 | -0.191 pp | -0.790 pp | 34.16% | 25.19% |
| 2824 | -0.191 pp | -0.627 pp | 35.10% | 25.57% |
