# Oxford Pet independent evaluation — frozen design

Prepared 2026-10-30 after six-seed development, before any official-test scoring.
This evaluates fresh adaptation seeds conditional on one ImageNet-pretrained
ResNet-18, one dataset split and one GPU. It is not independent pretraining or
cross-hardware replication. Public pretraining overlap has not been audited.

## Fixed design and access sequence

Use 24 seeds 2801–2824, disjoint from baseline and development seeds. Each has
full BF16, frozen-prefix and periodic-r4 arms. Repeat the six development arm
orders four times, giving each order four occurrences. Keep the same 2944 train
examples and 736 development-validation examples; do not retrain on all trainval.
This preserves the 690-update, 15-epoch recipe, augmentation stream convention,
LR schedule and adaptation clock. Evaluation seed streams/heads are new. The
official test sample is separate from development, not 24 independent datasets.

All original execution settings remain: BF16 autocast, FP32 weights/moments/loss,
channels-last, fused AdamW, fixed BatchNorm buffers, deterministic algorithms,
public pretrained backbone, 224px crops from the cached 256px input convention.
Prefix: stem + layer1 + layer2; suffix: layer3 + layer4 + classifier. Refresh
global steps 0,4,8,... (173 of 690). Omitted prefix gradients are None, pausing
optimizer state and decay. No sweep, quality-based stopping or checkpoint choice.
Only final epoch counts. All unfavorable results are retained.

The runner first completes all 72 adaptation trajectories and their integrity
audit using trainval only. Only after those checkpoints are fixed and audited
does it read test.txt and decode/cache the 3669 official test images. Both source
preparation freeze and run freeze must verify before that access. Test ids must
be disjoint from all trainval ids. Test images use the same shorter-side resize
to 256 and 256 center crop, followed by center 224 crop and normalization.
No archive download is required. Cached test inputs have their own hashes and
recorded preparation duration. Never use test outcomes to revise this recipe.

Official test scoring saves final logits in common FP32 with TF32 disabled
(primary) and BF16 (secondary robustness). All 72 arms must finish scoring and
their file/logit audit before aggregate test results or claims are printed.
Partial files are preserved; completed trajectories/scorings are verified and
skipped on restart. Interrupted adaptation timings are flagged fragmented and
cannot receive a combined efficiency claim without review. Pure scoring restarts
do not make already-complete training fragmented. No replacement seed or favorable
attempt is selected automatically.

## Outcomes and inference

Periodic-minus-full is primary; frozen-minus-full is a prespecified secondary
comparison. Keep this ordering despite the favorable freezing development result.
For each method and seed compute exact final-test accuracy difference in percentage
points from saved logits and log(method complete time / paired full complete time).
Use n=24 paired seeds, with no pooling of baseline/development outcomes.

Report mean accuracy, paired mean differences, sample SD, two-sided paired-t 95%
intervals, individual seed results and attained half-width. For timing report
geometric mean ratio and its t interval on log ratios, converted to saving bounds;
also report arithmetic mean paired savings, ranges and training-only times.
Timing inference describes this single session on this GPU; adjacent/interleaved
runs may share drift. Seed replication is not independent timing-session replication.

Define two composite claims: each method preserves quality AND saves complete
adaptation time. For each, require a one-sided 97.5% lower paired-t accuracy bound
strictly greater than -0.30 pp AND a one-sided 97.5% upper paired-t log-time-ratio
bound strictly below zero. Each is an intersection-union test at alpha .025;
Bonferroni across the two claims bounds family-wise error by .05 under the test
assumptions. Neither criterion alone warrants the combined claim. A failed NI
test means noninferiority not established, not proof of inferiority. No accuracy
superiority claim is specified. Frozen-versus-periodic and own-BF16 scoring are
descriptive/exploratory, with no additional confirmatory claim.

Use paired t bounds conditional on the observed test sample; they quantify
adaptation-seed variability, not uncertainty from drawing a different test set.
Normal/approximately normal paired differences and log ratios are assumptions.
Keep this specified analysis even if the outcome is unfavorable; describe unusual
distributions or measurement problems as limitations, not reasons to select a test.

## Precision and timing scope

The development periodic paired SD is 0.641615 pp. Its one-sided 95% chi-square
upper SD is 1.340499 pp using five degrees of freedom. A two-sided t interval with
desired half-width 0.20 pp would require 42 or 175 pairs, respectively. Respect
the practical cap of 24; projected half-widths are 0.270930 and 0.566043 pp.
These projections may not transfer to official-test accuracy. Explicitly disclose
that the original precision target is not assured. Do not widen the .30 pp margin.

The primary timing endpoint is the same adaptation-complete clock as development:
cache/model loading, warmup/restoration, all training, epoch development validation,
final development BF16 validation and checkpoint/log writing. Fresh matched full
controls are measured alongside candidates. Preflight/imports, archive download,
one-time cache preparation, source pretraining and independent audits are excluded.
Official-test scoring happens afterward for scientific assessment, is recorded
separately (including cache/model/checkpoint loading and logit writing), and is
not silently added to or removed from individual arms' adaptation clocks.
Report this as cached-input adaptation savings, not raw-data/pretraining-inclusive
or complete study wall-clock savings. Post-training scoring order follows the
same arm order; its duration does not drive a speedup claim.

## Preparation and integrity

Only artificial CPU/native CUDA workflows are exercised during preparation.
Freeze source hashes, exact runtime, development evidence/precision analysis and
this specification before enabling real execution. The scientific helpers remain
unchanged. Historical filename development_freeze.json inside training/ is retained
to reuse the audited training implementation; it contains this evaluation's 24
seeds and new fingerprint. Final analysis checks expected optimizer ages, exact
frozen prefix and BN state, data/LR streams, saved-logit accuracy, source/input/
checkpoint hashes and checkpoint-scoring provenance. Synthetic test fixtures
exercise access gating, endpoint inference and complete no-retrain restart.
