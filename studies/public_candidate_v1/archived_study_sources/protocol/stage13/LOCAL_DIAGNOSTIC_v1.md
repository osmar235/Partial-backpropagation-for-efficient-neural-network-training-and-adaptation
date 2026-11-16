# Stage13A: three-block local objectives, mechanics and cost

Prepared2026-10-22 before diagnostic outcomes. The author approved idea3:
divide credit assignment into smaller learning problems. This first step is
a fixed-state diagnostic, not an efficacy training run or a novel method claim.

## Prior art and question

Decoupled Greedy Learning (https://arxiv.org/abs/1901.08164) and local error
signals (https://proceedings.mlr.press/v97/nokland19a.html) establish this general
direction. We implement a simple local supervised cross-entropy variant, not
their full recipes or a claimed reproduction. Local gradients are separated;
representations still depend on preceding blocks. No independent-process or
parallel-GPU speedup is assumed. All blocks run sequentially on one GPU.

Question: at the already learned ImageNet features / head-warmup states, do
local objectives give useful update directions and measurable resource headroom
relative to full backpropagation, once auxiliary-head costs are visible?

## Fixed sources and architecture

Use preserved Stage12A lp_full warmup_checkpoint.pt files for seeds1601/1602:
epoch2,704head updates, no backbone adaptation. These are seen diagnostic source
states, not fresh efficacy seeds. Use their exact full hashes and run provenance.
All source weights and buffers remain unchanged on disk. Source learning and
external ImageNet pretraining are common sunk costs, never claimed as free
end-to-end training. Earlier Stage11/12 outcomes remain unchanged.

ResNet-18 at96px, three blocks:

1. Stem + layer1 + layer2, with global-average-pool / linear10-class auxiliary head.
2. layer3, with another global-average-pool / linear10-class auxiliary head.
3. layer4 + existing final classifier.

Auxiliary widths128/256. Fixed BatchNorm running statistics in every mode,
including full; affine weights train with their block. No dropout, feature cache,
feedback predictor, random/genetic optimization or automatic boundary search.

## Four matched modes

| Mode | Objective and gradient flow |
|---|---|
| full | Final CE through the entire network; auxiliary heads unused |
| full_aux | Sum of both auxiliary CEs and final CE, all gradients connected |
| local | Same three CEs, but detach between blocks; each trains only its block/head |
| suffix | Final CE updates layer4/classifier only; preceding blocks frozen |

Local forward/backward runs block by block, releasing each block's backward
graph. All parameter updates occur together AFTER all three backward calls;
later blocks consume pre-update features. Full_aux isolates the effect of extra
supervision from cutting gradient flow. All local CE weights are1, no division
by three. Ordinary full does not compute unused auxiliary heads; their tiny
parameters are present in the shared model container for matched state restoration.

## Training-only pools and auxiliary calibration

Keep the original45,000/5,000 training/validation split, but this diagnostic uses
ONLY the45,000training examples. No test examples; no validation scoring. From
each class, deterministically take200calibration,128update-probe and128score
examples from a seeded permutation. Pools are mutually disjoint. Source models
have already seen training data; the score pool is independent of diagnostic
head calibration and update probes, not an unseen generalization holdout.

Initialize the two auxiliary heads independently of the pretrained backbone.
Train ONLY those heads for128fixed minibatches of128, sampled with replacement
from the2,000calibration examples. Use native32px crop/flip, resize96 and
ImageNet normalization as Stage12. Backbone and final classifier stay fixed.
Auxiliary AdamW lr.001/wd.01; no LR selection, early stopping or reuse of score
labels for calibration. Log both head losses and save states/counters.

Measure auxiliary initialization, calibration, copies and calibrated-checkpoint
serialization together as calibration_overhead_seconds. Do not hide this
setup cost behind the full baseline. No source-backbone updates accumulate.

## Matched one-step operations

Twelve trials per source, four modes per trial,96measured updates total.
Draw128examples without replacement within each probe and score minibatch
(different trials can overlap). Probe batches receive deterministic augmentation;
score batches use deterministic resize/normalization. Each mode sees the same
probe/score pair and starts from the same calibrated model weights. Rotate mode
order and reverse it in alternate groups of four trials.

Use fresh-zero Adam states for each operation: backbone lr1e-4, final/auxiliary
heads lr.001, wd.01. This diagnoses first-step Adam directions, NOT continuation
of the source's mature head optimizer. Inactive parameters have grad=None and
no optimizer state/weight decay; all active counters are exactly1 afterward.

First warm each mode with three independent reset operations. Cold timings are
reported separately. Reuse an allocated all-zero moment template after warmup
for measured operations; model/optimizer restoration is outside the timer.
Report that this is warmed-kernel, zero-moment per-operation cost, not a training
trajectory. Future pilots must charge their actual continuation/reset policy.

Timer includes forward, all local/global backward calls, small-loss readback,
zeroing gradients, Adam step and CUDA synchronization. It excludes common
batch transformation, state restore, scoring, gradient-copy/diagnostic arithmetic,
checkpoint IO and source loading. These exclusions limit the speed claim to
device-ready update operations. No end-to-end speedup is inferred here.

GPU memory: record allocated baseline, peak allocated during the update and
incremental peak above baseline. Reset peak counters per measured operation.
Baseline includes resident data, model and that mode's optimizer buffers;
incremental peak is gradient/activation/workspace allocation, not total VRAM or
allocator reserved memory. Report absolute peaks too. CPU memory is unavailable
and encoded as0 in artificial tests only; never interpreted as a measured saving.

After each disposable update, measure final-classifier CE change on its separate
score batch; restore before every mode/trial. Also record per-block dot products,
norms and cosine between local/control gradients and full-loss gradients at the
same initial state. Zero-norm cosines remain undefined. Final-block gradients
must agree across modes because activations and final loss are unchanged before
the synchronous update. Save first-trial model/Adam states for every mode.

## Predeclared review flags

These are engineering filters for discussing a bounded pilot, not statistical
tests or evidence of convergence. For EACH source state:

- Full mean score-loss improvement must be positive. Local mean improvement
  must also be positive and at least half of full's mean improvement.
- Local must have either at least10% lower mean update time than full OR at
  least20% lower mean incremental allocated peak memory.
- All source/state, finite-gradient, detachment, optimizer and pool checks pass.

Both source states must meet the filters to qualify for pilot review. The
half-utility threshold is deliberately a coarse prospective diagnostic filter,
not an acceptable final-accuracy margin. Correlated/overlapping batches and
two seen checkpoints provide no inference about generalization or significance.
A failed full-utility positivity check makes the utility screen uninformative;
do not tune the probe step size or move the source checkpoint after seeing it.
Gradient alignment is descriptive and has no extra acceptance threshold.

If local time is lower, report ceil(calibration_overhead / per-step saving) as
an OPTIMISTIC calibration break-even step count; otherwise leave undefined.
It ignores recovery dynamics and is not a predicted training speedup. Report
all four modes even if review flags fail. No automatic efficacy run, longer
calibration, hyperparameter sweep or fallback implementation.

## Provenance and execution

Freeze sources/spec/tests/launcher/protocol, previous evidence, exact source
checkpoints and pretrained input before real output exists. CPU/CUDA unit tests
and complete artificial workflows precede this freeze. Preserve pool indices,
calibration traces, gradients' aggregate statistics, utility/time/memory rows,
cold timings, calibrated states, first-step checkpoints and input hashes.
Independent audit reconstructs indices, arithmetic, modes/counters and flags.
Completed rerun verifies output without writes. An interrupted diagnostic starts
a new numbered attempt; completed original attempts remain untouched. Review
the diagnostic before deciding whether idea3 warrants an efficacy pilot.
