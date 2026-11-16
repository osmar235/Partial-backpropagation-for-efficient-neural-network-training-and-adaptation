# Stage 8 second architecture study

Prospective plan dated 2026-10-18. No substantive Stage 8 CIFAR training has run.
Machine-readable settings: `study_spec_v1.json`. This is a new study, not an
amendment to Stage 6 inference or a replacement for Stage 7 results.

## Scientific question and scope

Does bounded prefix-gradient omission retain a useful accuracy/time trade-off
in an attention-based architecture distinct from ResNet-18-GN? Are its outcomes
competitive with a development-calibrated periodic schedule and displacement
replay? All four arms and all seeds will be reported, including negative or
inconclusive results. Another architecture does not by itself establish that
loss-informed scheduling is uniquely beneficial.

Use the native torchvision `VisionTransformer` base class, randomly initialized,
without downloaded/pretrained weights or architecture changes selected from
new evaluation outcomes. Images are 32 x 32, patches 4 x 4 (64 patches plus a
class token), six encoder blocks, hidden width 192, three attention heads, MLP
width 768, LayerNorm, dropout 0.1 and attention dropout zero. This is a compact
ViT configuration, not a claim to reproduce a named pretrained ViT benchmark.
Prefix: patch projection, class token, positional embedding and first two
encoder blocks. Suffix: remaining four blocks, final LayerNorm and classifier.
The boundary is fixed structurally, not selected for favorable new accuracy.

The baseline is full-gradient **AdamW**, making explicit the change from the
earlier proposed Adam sketch. The established Stage 7E result motivates this
fixed-recipe extension; no new partial-Adan implementation is implied. This
study does not replace or pool the old Adam/Adan comparisons.

## Development before evaluation

1. Use only CIFAR-10's 50,000 training images. A fixed stratified split reserves
   500 examples per class (5,000 total) for validation and 45,000 for training,
   with split seed 80401. Index hashes are recorded. Development code does
   not construct the test dataset.
2. On development seeds **1001 and 1002**, run full AdamW with each of learning
   rates **0.0003 and 0.001**: four complete 150-epoch runs. Reverse the LR
   execution order on the second seed. No pruning or selection of best epochs.
3. Common recipe: batch 256, AdamW defaults and decoupled weight decay 0.05,
   cross-entropy label smoothing 0.1, padded crop/flip augmentation, fixed
   CIFAR-10 normalization, 150 epochs, five epochs of linear LR warm-up from
   lr/warmup_updates to the base rate, then cosine decay to a 1% floor.
   This is a credible baseline candidate, not an assertion of optimized/SOTA
   transformer performance. Longer training than Stage 7 is deliberate.
4. Select the LR by mean **final validation accuracy** over both development
   seeds, with a tie going to the smaller rate. Readiness requires mean final
   validation accuracy at least 85% and neither seed below 83%. These are
   prospective operational baseline-quality thresholds, not statistical gates
   or universal publication standards. If they fail, retain the screen,
   **do not open evaluation seeds**, and document any subsequent recipe change
   as a prospective amended development plan. No automatic unbounded search.
5. At that LR, run LGPB on the same two development seeds/split for two more
   complete runs. Their mean per-epoch partial fractions supply the periodic
   control profile. No gate parameter or boundary is tuned. Development total
   is **six training runs** if baseline readiness passes.
6. Review development for implementation/quality failures, then create the
   immutable freeze JSON from completed records and source hashes. Freezing
   refuses incompatible prior freezes or existing evaluation output without
   its prospective freeze. Substantive evaluation cannot start without this
   record. No manual editing of rates, seeds, profiles or criteria after freeze.

The new architecture's initialization/training seeds are new; CIFAR-10 test
images have been used elsewhere in the project. Do not claim an untouched
evaluation dataset or independent laboratory replication.

## Four-arm frozen evaluation

Reserve **1101–1106** for the evaluation. Each uses the full 50,000 training
images and 10,000 test images, without any new hyperparameter selection.
There are **24 evaluation runs**, each 150 epochs/29,400 updates. Within each
seed, initialization, minibatch ordering, augmentation and dropout random
streams are matched. Dropout is reset independently of algorithm warm-up.

- `full_adamw`: fresh gradients for all parameters at every update.
- `lgpb`: original loss-before-EMA comparison, rho 0.95, full after partial,
  exact fresh suffix gradients, explicit zero prefix gradients, ordinary AdamW
  state/step/decoupled-weight-decay continuation.
- `periodic_matched`: loss-blind nonadjacent schedule with exactly
  round(196 q_development_epoch) partials per epoch. First update of every epoch
  is full, guaranteeing no adjacency across epochs. Other state handling is
  the same as LGPB. This matches independent development rates, not each new
  test run's loss-gate decisions; realized rates may differ and must be shown.
- `replay_prefix`: same endogenous loss gate; prefix gradients are None during
  partial updates, so prefix AdamW state and decoupled decay are skipped, and
  the preceding full-step displacement is reapplied. True cloning/replay cost
  is timed. Later decisions may diverge as weights/losses change.

Orders are specified in the JSON as three order/reversal pairs. **Every pair
of methods runs first in three seeds each**, and each method occurs in the
early half of the order in three seeds and late half in three. This improves
on relying solely on cyclic positions. Do not replace interrupted/slow seeds.
Resume restarts an unfinished variant from initialization and skips completed
matching variants; incompatible code/settings/environment fail explicitly.

## Prospective inference

The single new primary comparison is LGPB/full AdamW on the six evaluation
seeds. Success requires both: a paired two-sided 95% t lower accuracy-difference
bound above **-1.0 percentage point**, and an exponentiated paired 95% t upper
log-wall-ratio bound below **1.0**. Both use five degrees of freedom. This is
a newly declared benchmark-specific margin motivated by comparability with
the original operating-point question, not retrospective reuse of Stage 6's
confirmatory designation. Six seeds are a bounded replication budget, not a
guarantee of sufficient precision or power. Report estimates and uncertainty
even when either gate fails; do not add seeds after inspecting results.

Report all four arms against full AdamW and LGPB against periodic/replay.
Mechanism intervals are descriptive and unadjusted for multiple comparisons;
no superiority/equivalence declaration is inferred from nonsignificance.
Accuracy effects are variant minus reference; wall ratios variant/reference.
Accuracy is the **final** epoch, not the best test epoch.

Predeclare target accuracies **70%, 80%, 85%**. Time-to-target means cumulative
timed training-only wall through the first end-of-epoch evaluation crossing;
warm-up, evaluation and checkpoint time are excluded, and there is no
within-epoch interpolation. Unreached targets remain unestimated. Report all
targets and paired reach counts rather than choosing a favorable threshold.

## Timing and state audit

Use the local RTX 5080, FP32, no AMP/compile/TF32. Algorithm-specific 196-step
hardware warm-up uses a separate RNG stream; model parameters and optimizer
state are reset before the timed trajectory. Periodic warm-up alternates full
and partial paths; it does not contribute scientific decisions or metrics.

The timed epoch includes device indexing, CPU shuffle bookkeeping, augmentation,
forward, scalar loss extraction, gate, backward, optimizer and per-step record
construction. All arms perform exactly **one training loss `.item()` per
update**, including full AdamW. There are eight explicit CUDA sync calls per
update (four timer start/end pairs) and two per epoch inside the training region.
Scalar extraction can synchronize implicitly; evaluation has additional calls
outside that region. Record full/partial operation timers and wall minus their
sum; do not call that residual GPU compute savings.

Sample nvidia-smi before warm-up and before/after each training epoch, outside
the training timer. Records include timestamp, driver, clocks, power/limit,
temperature, utilization and pstate. A three-sample median above 20% utilization
at study startup blocks the launcher so other GPU work can finish first.
This is a workload guard, not an excuse to discard data after seeing outcomes.
Telemetry is evidence about conditions, not an automatic correction for drift
or an energy measurement. Report any drift and retain all seeds.

Record training, evaluation, hardware-warm-up and checkpoint durations separately.
Per-run elapsed time includes telemetry, warm-up, evaluation and checkpoint
operations; initial dataset loading/model construction are excluded from that
per-run elapsed measure. Do not describe training-only time as total experiment
time. Save final model **and optimizer** states with hashes, all epoch/step logs,
initialization hash and configurations. Audit checkpoints for correct per-block
step counters: all updates for continuation; full updates only for replay prefix.

All old code, outcome directories and release artifacts stay unchanged. No new
architecture result is assumed in the v6 manuscript until this study is run.
Publication also requires the separately identified supplement, code/data release
and author-reviewed manuscript updates; this plan does not authorize publishing
an unreviewed package or invent a public DOI/submission status.

## Primary-source rationale

- [Vision Transformer](https://arxiv.org/abs/2010.11929) establishes the patch-token
  transformer family; this compact randomly initialized configuration is new
  to our benchmark rather than a claim of matching its large-scale results.
- [Torchvision documentation](https://docs.pytorch.org/vision/main/models/vision_transformer.html)
  describes the native model family; the installed class/source is hashed.
- [Vision Transformer for Small-Size Datasets](https://arxiv.org/abs/2112.13492)
  motivates taking development baseline quality seriously. We do not claim to
  implement its shifted-patch/locality-attention extensions.
