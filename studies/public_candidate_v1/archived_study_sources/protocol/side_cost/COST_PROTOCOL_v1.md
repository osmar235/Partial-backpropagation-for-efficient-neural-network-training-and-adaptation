# Side-network operation-cost feasibility — 2026-10-25

Bounded artificial timing check requested by the author. No real data/checkpoint,
task accuracy, parameter-count shortcut to speed, or full-study efficiency claim.
Keep the selected prototype unchanged. No width, tap or hyperparameter sweep.

Four arms: native full training; native head-only training with frozen body;
final-feature side control; multi-depth side candidate. The original six-block
width-192 backbone and all 65 tokens remain. Both side models use three width-48
blocks, 125,002 trainable parameters, and identical initial states. Backbone
dropout follows train mode for all arms. Head-only relies on autograd having no
trainable body/input path. Each side forward includes native backbone execution,
all feature-capture hooks, detached features, side processing and classifier.
There is no cached-feature reuse or auxiliary loss.

Two arbitrary artificial model/input seeds (99031/99032), full batch 256 and
last-batch size 200, eight rounds per condition. For each seed/shape, cyclically
rotate the four arms each round; reverse the base order on alternating seed/shape
combinations. Every arm occupies every position twice. Full is measured in every
round, allowing paired ratios. Log per-round GPU telemetry outside timed blocks.

Each trial freshly constructs its model/AdamW on CPU, transfers to the GPU,
then measures three discarded warmup updates followed by exact weight restoration
and a new empty-state AdamW. Charge construction/transfer/setup and warmup/restore
separately. Record the first post-restore update with lazy moment allocation;
then time one block of eight updates. Every update includes RNG seeding, gradient
zeroing, full forward, smoothed CE, scalar readback, backward and AdamW. Input
creation/transfer is shared outside operation timings. No augmentation is timed.
Use synchronized wall-clock boundaries; do not substitute CUDA-event arithmetic
or FLOP estimates. Data are arbitrary, with nonzero artificial classifier weights.

One model is resident at a time; delete old models/optimizers and release cached
GPU allocation outside each trial. Memory reset/boundary instrumentation is
outside update timers. Record baseline, absolute peak, incremental peak and
reserved allocation for setup, warmup/restore, first update and steady block.
Frozen-body hash and absent-gradient checks, finite trainable gradients, parameter
changes, and exact Adam age nine are checked outside the update timers. Source
initialization hashes must reproduce after discarded warmup. No checkpoint saving
or validation is timed. Cold setup means a fresh model in an already running,
initialized process/context; it is not OS/process/GPU cold-start latency.

Primary gate, declared before execution: median of eight paired steady-block
multi_side/full ratios <=0.90 in EACH of the four seed/batch conditions. No memory
alternative and no pooled result that hides a failed condition. Report the ratio
range, quartiles, round counts meeting 0.90, absolute operation times, first-step
ratios and setup/warmup costs. Timings are repeated observations on one GPU, not
independent quality/learning replications or statistical significance.

Full and head costs are descriptive feasibility bounds. Head-only and final-side
accuracy are unknown, so passing this gate does not imply beating the fastest
quality-qualified control. The head-warmup/full strategy combines these phase
types but needs its own later complete-run timing; no synthetic composite is
substituted for it here.

An optional transparent 880-update arithmetic projection uses 875 batch-256 and
five batch-200 operations plus per-trial setup/warmup and first-state allocation.
It is not measured end-to-end adaptation time and is not a decision criterion.
The actual efficacy screen still requires <=0.30-point loss versus strong controls
and >=10% complete adaptation saving in both seeds, if prepared later.

All 128 trials are retained. Interrupted executions restart the bounded check
in a new attempt with old partial records preserved; no selective rerun of bad
timings. A completed rerun only verifies/audits. The source/spec/environment freeze
is written before the target CUDA benchmark. CPU smoke uses reduced artificial
shapes and cannot pass the target feasibility gate. Prior evidence is hash-protected.

Gate failure closes this fixed side architecture for the current time objective.
Gate success permits real-protocol preparation; it does not launch real training,
establish useful quality, demonstrate novelty or lift the publication hold.
