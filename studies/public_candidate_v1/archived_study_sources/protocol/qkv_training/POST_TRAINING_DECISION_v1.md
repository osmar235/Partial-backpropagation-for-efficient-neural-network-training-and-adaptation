# Post real-data confirmation decision — 2026-10-26

## Current decision: QKV complete real-data adaptation saving confirmed — 2026-10-26

Both source conditions pass the prospectively frozen real-data study. Median
paired complete-workflow time reductions are6.34% (seed1501) and6.40% (seed1502),
with8/8 positive pairs per source and16/16 overall. Adjusted sign-test p=.0078125
per source. Median absolute paired saving is about1.96seconds per five-epoch
adaptation. Native/changed median times are30.925/28.784s and31.111/28.849s;
the declared percentage uses paired ratios, not the ratio of those medians.

The measured workflow includes data loading/preprocessing/device transfer,
source/model setup, warmup/restore, training/augmentation, initial plus five
validations and final checkpoints/predictions/evidence. Interpreter/import startup,
preflight, provenance hashes and post-run audits are excluded. Training/augmentation
alone is faster in every pair, with median reductions6.97% and7.24%. Native-first
and candidate-first strata remain positive for both sources. No outlier removal.

Every recorded training loss, epoch validation-logit array, final model/Adam/RNG
state and learning stream matches exactly across paired arms and repeated sessions.
Both native references reproduce archived Stage15A losses/logits with zero error.
Both arms finish at84.70% /83.88% validation accuracy. All32 final checkpoints
were reloaded and reproduced their saved predictions exactly. Independent raw
arithmetic and stdlib NPY scoring agree. Review performed inference, not training.

There are32 trajectories,28,160 updates,96 discarded warmups,eight distinct
processes and no fragmented/replacement attempts. Absolute peak allocated memory
is identical within every pair (2,399,243,264bytes). Incremental allocations vary
with starting baselines by +/-68,157,440bytes; reserved peaks by +/-2,097,152bytes.
No systematic memory saving is established. Idle preflights pass; sparse snapshots
do not certify interference-free execution throughout the measured blocks.

Conservative median intervals [5.25,13.21]% /[3.24,8.65]% and p-values assume
independent identically distributed session effects. Repetitions share one GPU
and two fixed learning trajectories; do not count them as extra model/data seeds.
This supports complete measured adaptation savings at preserved learning outcomes
for the tested ViT/CIFAR-development/FP32/math/deterministic configuration, not
faster original pretraining or a universal hardware/architecture claim.

Retain this as a substantive systems implementation result for the paper. Attribute
it to shared QKV split/autograd processing; prior approximate-gradient outcomes
remain unchanged. Original10% screen stays failed; separate positive-direction
confirmation and real-data confirmation pass. Broader workloads/architectures,
competitive optimized execution settings and prior-art/novelty remain open before
broader publication claims. No next experiment/command is prepared or launched.
Publication remains on hold. All prior/raw evidence is preserved unchanged.

Review: results/qkv_training_review_v1/REPORT.md. Master v7.34 and paper memory
v23 are current; earlier statuses and commands are historical.

