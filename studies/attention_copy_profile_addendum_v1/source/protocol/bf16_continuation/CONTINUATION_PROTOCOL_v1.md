# BF16 matched checkpoint adaptation — 2026-10-27

New prospective study of practical quality and complete adaptation cost. The
earlier BF16 parameter-distance gate remains failed, and the original QKV 10%
screen remains failed. No coordinate is exempted. This study makes no claim
that BF16 is a novel method or preserves exact parameter trajectories.

## Fixed design

Use the two archived Stage11B full checkpoints (1501/1502), with all parameters
trainable in FP32. Use pinned reference attention in all three arms: FP32/math,
BF16/math and BF16/automatic SDPA. BF16 uses autocast only for model forwards;
cross-entropy receives logits.float() with autocast disabled in every arm.
Optimizer moments remain FP32; no GradScaler, gradient clipping, key-bias
removal, selective-precision modification or compilation. Preserve original
dropout. Matched dropout seeds do not assert identical masks across precisions.

Use the established five-epoch adaptation recipe, with fresh AdamW in every arm,
common parameter groups, LR schedule, augmentation, MixUp and sampling streams.
This is adaptation from trained weights, not exact continuation of the original
17600-step optimizer history. The original optimizer options are retained but
moments/counters are reset. Warmup updates are discarded and weights/optimizer
restored exactly before training. Six sessions cover all six arm permutations
for each seed; reverse seed order on alternate sessions. Each session is a new
process. Full scope: 36 trajectories, 31680 training updates plus 108 discarded
warmup updates. Same fixed random streams across sessions measure timing
repeatability and numerical repeatability, not 12 independent model seeds.

Use only the established 45000/5000 CIFAR10 training/validation split. Evaluate
every epoch in common FP32/math to isolate training effects. Also evaluate final
weights once in the arm's own policy to measure deployment precision effects;
charge that extra evaluation to every arm, including FP32. Save both logits.
No test-set access, best-epoch selection, tuning, extra epochs or automatic run.

## Qualification and prospective decision

The FP32 reference must meet 80% final validation accuracy, lose no more than
0.30 percentage points against its source, and be within 0.30 points of the
previous qualified full adaptation reference, in every session and both seeds.
Source common-FP32 logits must match archived source logits within 1e-5, with
identical predictions. Source mapping, all trainable gradients, finite states,
optimizer age, update schedule and RNG streams must pass integrity checks.

For a BF16 policy to qualify, both its final common-FP32 validation accuracy and
its final own-policy accuracy must be within 0.30 points of the paired FP32
reference, in every session and both seeds. This is a fixed-checkpoint screening
margin, not a statistical noninferiority claim about a model population. Report
all loss/accuracy differences and session variation, including unfavorable ones.

Primary time is complete adaptation wall time: per-arm data loading, checkpoint
and model setup, profiled warmup/restoration, training including augmentation,
all validations and final checkpoint/logit/trace writing. Imports, idle preflight,
frozen-provenance verification and post-run audits are excluded. Component times,
training-only time and allocated/reserved memory are secondary. Original source
training cost is reported separately; do not claim faster pretraining.

For each candidate policy, require positive paired complete-time saving in all
six sessions in each source, with no fragmented sessions. Also report an exact
one-sided sign test per source (zero savings count as nonpositive); combine the
two source tests by their maximum p-value for the both-source claim, then apply
Bonferroni for the two candidate policies. Require adjusted p <= 0.05. Six/six
positive in each source gives 0.03125 adjusted p. This conditional direction
screen assumes independent session effects; shared-machine drift limits it.
Report median/min/max savings rather than treating repeated models as extra
quality replicates. No new minimum percentage saving is selected. A small
positive saving is reported at its observed magnitude, not called transformative.

An integrity PASS differs from a quality/timing qualification. Any baseline,
quality, direction, fragmentation or availability failure blocks the practical
claim; no automatic follow-up, tolerance change or favorable-subset exclusion.

## Execution and evidence

Profile SDPA dispatch during the first charged warmup update; preserve settings
and count actual backend calls. Automatic mode may choose efficient rather than
flash; do not infer dispatch from its name. Unexpected/nonfinite errors abort,
retain the attempted files and a failure record. CPU smoke explicitly tests only
FP32; CUDA smoke exercises all three policies on artificial tiny data/models.

Save per-step loss/LR/MixUp/dropout seeds, epoch permutations and final stream
states, all validation logits, checkpoints, component times and telemetry.
Audit source/freeze/completion hashes, regenerate data-stream bookkeeping,
check optimizer/parameter states and rescore saved logits. Re-evaluate each final
checkpoint in common FP32 and its own policy outside measured time; require
maximum absolute logit error <= 1e-5 and identical predicted classes. Report
exactness as well as tolerance. Auditor performs inference, not training.
Completed trajectories are verified and skipped on restart. Interrupted attempts
are retained; any fragmented session precludes the primary timing claim.

After results are reviewed, broader seeds/architectures/hardware and publication
scope remain separate decisions. This screen alone establishes neither generality
nor novelty. Existing research evidence and frozen source packages stay intact.
