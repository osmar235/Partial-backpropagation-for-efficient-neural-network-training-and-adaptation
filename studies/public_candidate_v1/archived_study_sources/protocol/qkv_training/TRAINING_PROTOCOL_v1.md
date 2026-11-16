# QKV real-data adaptation confirmation v1 — 2026-10-26

The author approved testing whether the confirmed artificial update-time benefit
transfers to real training. This follow-up was selected after the positive QKV
confirmation. Preserve discovery, its failed >=10% target, and confirmation intact.

## Fixed question and design

Compare native versus unchanged `qkv_unbind.py` on the existing full RGB32 ViT
checkpoints, seeds 1501/1502. Use the established Stage15A full-adaptation recipe:
five epochs, all 45,000 development training examples, batch 256 (last batch 200),
5,000 validation examples, eval batch 512, fresh AdamW, LR 1e-4, one warmup epoch,
cosine to 0.1 times LR, weight decay .05, MixUp .2, smoothing .1, original random
stream offsets and dropout. All parameters train. FP32, TF32-off, strict
determinism and math attention remain fixed. No new layout, precision or optimizer.
Use training-derived validation only; do not evaluate CIFAR-10 test data.

Eight fresh sessions each run both arms for both source seeds: 32 trajectories,
28,160 training updates plus 96 discarded hardware-warmup updates. Seed order
alternates; arm order is balanced within each source across sessions. No tuning,
outlier removal, favorable-run selection, speed-dependent stopping or extra repeats.
Sessions repeat two fixed learning trajectories, not 16 independent training seeds.

Each trajectory starts in a separate output attempt, independently loads the data
and source checkpoint, and restores model/fresh optimizer after three warmup updates.
Data may be in OS cache; this is repeated application-workflow timing, not a guaranteed
cold disk/cache startup study. No cache purging. Preserve all interrupted attempts;
fragmented sessions cannot support the primary confirmation conclusion.

## Cost endpoints

Primary: one synchronized wall clock starting immediately before per-arm data
loading and ending after the final checkpoint, predictions, curves, training trace
and input-order/RNG evidence are saved. Include data preprocessing/device transfer,
source load and adapter installation, warmup/restore, initial validation, all five
training epochs with augmentation/MixUp and LR scheduling, five epoch validations,
ordinary logging and artifact serialization. Same clock and outputs for both arms.
Checkpoint time means completion of the torch.save call, not a forced durable-disk
fsync guarantee. Measure complete_workflow_seconds directly; never reconstruct it
by adding selective GPU timings or a shared-data allocation.

Exclude Python startup/imports, preflight, immutable provenance hashing, post-run
correctness audits, and final checksum manifests. Report data/setup/warmup/training/
validation/artifact times separately plus the residual clock time. No profiler or
per-update CUDA timing instrumentation. Record per-step loss, LR and MixUp lambda,
full epoch batch orders and final RNG states; their collection/serialization cost
is included equally in the primary endpoint. No standalone inference benchmark.
Record end-of-workflow peak allocated/reserved device memory, including resident
data. The original source-training cost is a sunk cost and reported separately;
this is adaptation runtime, not a faster original 100-epoch pretraining claim.

## Correctness and reference gates

Verify source/data hashes before scientific execution. Native runs must reproduce
the archived Stage15A full learning trajectory: every training loss within 1e-6,
all epoch logits within 1e-5, unchanged predicted labels and accuracy. Initial
predictions match the archived source within 1e-5. Require native final accuracy
>=80% and no >0.30 percentage-point decline from source, unchanged readiness rules.

For paired native/unbind runs, require identical every-step losses and recorded
learning streams, every-epoch validation logits, final model/Adam state hashes and
final RNG states. Require deterministic repeats across sessions. This prospective
identity gate is strict; no post-hoc tolerance relaxation. Any failure withholds
confirmation even when speed improves. Scientific gate failure is reported separately
from evidence-integrity failure. Nonfinite training or corrupted sources abort.
Retain native references and both trained checkpoints for independent rescoring.

## Analysis

For each source seed, compute eight paired candidate/native complete-workflow time
ratios (one per session). Report all savings and their median. Primary directional
test: one-sided exact sign test against no positive median saving, Bonferroni across
two source seeds, adjusted p<=.05 in both. Eight positive pairs per seed are needed
(adjusted p=.0078125). This is not a new 5% or 10% practical target. Numerical and
reference gates, unfragmented sessions and the two timing tests must all pass.
Report [min,max] session savings as conservative exact median intervals: marginal
99.21875% and simultaneous >=98.4375% coverage assuming independent identically
distributed continuous session effects. Same-machine drift limits the assumption.
Order-stratified results, component timing and memory are descriptive secondary
endpoints. Never pool with artificial benchmarks or treat epochs/updates as replicates.

A pass supports measured complete adaptation savings at preserved quality for this
model, two source states, dataset and execution policy. It does not establish novelty,
other architectures/hardware, optimized AMP/compile baselines or publication readiness.
Keep publication on hold pending independent review and broader contribution assessment.

## Preparation and execution

Validate with small artificial CPU/CUDA models, including eval-mode identity,
optimizer state mapping, real augmentation/MixUp machinery, stream reconstruction,
checkpoint reload/scoring, audit tamper rejection and immutable replay. Smoke uses
two sessions, two artificial source seeds, two epochs, batch 3/eval 4; never qualifies.
Freeze sources, data/checkpoint fingerprints and environment before real execution.
Full real-data training is user-run via scripts/run_qkv_training.cmd. No automatic
additional experiment is launched after this study. Return results/qkv_training_v1.
